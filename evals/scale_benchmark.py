"""SystemOne throughput benchmark: time and cost at scale.

Runs the SystemOne classification stage (theme mapping + detail detection in
one batched pass) at increasing response counts to measure how wall time,
token usage and cost scale. Responses are resampled with replacement from a
real dataset, which preserves the length distribution that drives request
size. Resampled duplicates are fine for throughput measurement but carry no
ground truth, so this benchmark reports no accuracy metrics; use
compare_systemone.py for those.

Usage:
    uv run python evals/scale_benchmark.py --dry-run          # estimate only
    uv run python evals/scale_benchmark.py                    # 100, 1k, 10k
    uv run python evals/scale_benchmark.py --sizes 1000       # one size

Environment: TYPESAFE_API_KEY.
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from compare_systemone import JEV_INPUT_PRICE_PER_M, JEV_OUTPUT_PRICE_PER_M  # noqa: E402
from datasets import DatasetConfig, load_local_mapping_data  # noqa: E402
from systemone_diagnostics import build_sample_request  # noqa: E402

from themefinder import SystemOne, classify_responses_systemone  # noqa: E402
from themefinder.systemone import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
)

RESAMPLE_SEED = 20260917


def scale_responses(base_df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Resample the base responses with replacement up to the requested size."""
    rng = random.Random(RESAMPLE_SEED)
    texts = base_df["response"].tolist()
    return pd.DataFrame(
        {
            "response_id": range(1, size + 1),
            "response": [rng.choice(texts) for _ in range(size)],
        }
    )


def estimate(responses_df, question, themes_df, batch_size: int) -> dict:
    """Estimate tokens and cost for one size from a sample request payload."""
    sample = build_sample_request(responses_df, question, themes_df, batch_size)
    chars_per_request = len(json.dumps(sample))
    requests = -(-len(responses_df) // batch_size)
    # ~4 chars per token is a serviceable estimate for English + JSON.
    input_tokens = requests * chars_per_request // 4
    return {
        "requests": requests,
        "estimated_input_tokens": input_tokens,
        "estimated_cost_usd": input_tokens * JEV_INPUT_PRICE_PER_M / 1_000_000,
    }


async def run_size(
    size: int,
    base_df: pd.DataFrame,
    question: str,
    themes_df: pd.DataFrame,
    client: SystemOne,
    batch_size: int,
    concurrency: int,
) -> dict:
    """Run the classification stage at one scale and measure it."""
    responses_df = scale_responses(base_df, size)
    usage_before = (client.usage.input_tokens, client.usage.output_tokens)
    rate_limits_before = client.rate_limit_hits

    start = time.perf_counter()
    classified_df, unprocessable_df = await classify_responses_systemone(
        responses_df,
        client,
        question=question,
        refined_themes_df=themes_df,
        batch_size=batch_size,
        concurrency=concurrency,
    )
    seconds = time.perf_counter() - start

    input_tokens = client.usage.input_tokens - usage_before[0]
    output_tokens = client.usage.output_tokens - usage_before[1]
    cost = (
        input_tokens * JEV_INPUT_PRICE_PER_M + output_tokens * JEV_OUTPUT_PRICE_PER_M
    ) / 1_000_000
    return {
        "responses": size,
        "requests": -(-size // batch_size),
        "seconds": round(seconds, 2),
        "responses_per_second": round(size / seconds, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
        "cost_per_1k_responses_usd": round(cost / size * 1000, 4),
        "unprocessable": len(unprocessable_df),
        "rate_limit_hits": client.rate_limit_hits - rate_limits_before,
    }


def print_results(rows: list[dict], concurrency: int, batch_size: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="SystemOne classification at scale",
        caption=(
            f"batch size {batch_size}, concurrency {concurrency}. Responses "
            "resampled from real data (throughput only, no accuracy)."
        ),
    )
    for column, justify in [
        ("Responses", "right"),
        ("Requests", "right"),
        ("Wall time", "right"),
        ("Resp/sec", "right"),
        ("Input tokens", "right"),
        ("Cost", "right"),
        ("Cost / 1k resp", "right"),
        ("429s", "right"),
        ("Dropped", "right"),
    ]:
        table.add_column(column, justify=justify)
    for row in rows:
        table.add_row(
            f"{row['responses']:,}",
            str(row["requests"]),
            f"{row['seconds']:.1f} s",
            f"{row['responses_per_second']:,.0f}",
            f"{row['input_tokens']:,}",
            f"${row['cost_usd']:.4f}",
            f"${row['cost_per_1k_responses_usd']:.4f}",
            str(row.get("rate_limit_hits", 0)),
            str(row.get("unprocessable", 0)),
        )
    console.print(table)

    if rows:
        best = max(rows, key=lambda row: row["responses"])
        throughput = best["responses_per_second"]
        cost_per_1k = best["cost_per_1k_responses_usd"]
        console.print(
            f"Projection at this rate: 100,000 responses ≈ "
            f"{100_000 / throughput / 60:.1f} min, ${cost_per_1k * 100:.2f}"
        )


async def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Benchmark SystemOne classification throughput at scale"
    )
    parser.add_argument("--dataset", default="gambling_XS")
    parser.add_argument(
        "--question", type=int, default=1, help="Question part to draw responses from"
    )
    parser.add_argument(
        "--sizes",
        default="100,1000,10000",
        help="Comma-separated response counts to test",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--model", default=None, help="SystemOne model override (default jev-latest)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request plan and estimated cost without calling the API",
    )
    args = parser.parse_args()
    sizes = [int(size) for size in args.sizes.split(",")]

    config = DatasetConfig(dataset=args.dataset, stage="mapping")
    items = [
        item
        for item in load_local_mapping_data(config)
        if f"part_{args.question}" in item["metadata"]["question_part"]
    ]
    if not items:
        sys.exit(f"No question part {args.question} in dataset {args.dataset}")
    item = items[0]
    base_df = pd.DataFrame(item["input"]["responses"])
    question = item["input"]["question"]
    themes_df = pd.DataFrame(item["input"]["topics"])[["topic_id", "topic"]]

    if args.dry_run:
        print(f"Plan (batch size {args.batch_size}, concurrency {args.concurrency}):")
        total_cost = 0.0
        for size in sizes:
            plan = estimate(
                scale_responses(base_df, min(size, args.batch_size * 3)),
                question,
                themes_df,
                args.batch_size,
            )
            requests = -(-size // args.batch_size)
            tokens = plan["estimated_input_tokens"] // plan["requests"] * requests
            cost = tokens * JEV_INPUT_PRICE_PER_M / 1_000_000
            total_cost += cost
            print(
                f"  {size:>7,} responses → {requests:>4} requests, "
                f"~{tokens / 1000:,.0f}k input tokens, ~${cost:.3f}"
            )
        print(f"  Total estimated cost: ~${total_cost:.3f}")
        return

    client = SystemOne.from_env(model=args.model)
    rows = []
    for size in sizes:
        print(f"\nRunning {size:,} responses...")
        rows.append(
            await run_size(
                size,
                base_df,
                question,
                themes_df,
                client,
                args.batch_size,
                args.concurrency,
            )
        )

    print()
    print_results(rows, args.concurrency, args.batch_size)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = (
        results_dir / f"scale_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "question_part": item["metadata"]["question_part"],
                "batch_size": args.batch_size,
                "concurrency": args.concurrency,
                "timestamp": datetime.now().isoformat(),
                "runs": rows,
            },
            indent=2,
        )
    )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
