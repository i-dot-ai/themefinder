"""Full SystemOne benchmark: end-to-end totals, stage accuracy, and scale.

One command, three phases:

1. **End-to-end** — runs the complete `find_themes` (LLM) and
   `find_themes_hybrid` (LLM + SystemOne) pipelines on the same responses,
   for the overall wall time and cost picture. Each run generates its own
   themes, so this phase reports descriptive output statistics only.
2. **Stage-level** — re-runs the classification stages (theme mapping,
   detail detection) on the dataset's fixed reference themes, where ground
   truth exists, for a real accuracy comparison (F1, evidence accuracy,
   AUC, agreement).
3. **Scale** — reruns the SystemOne classification at increasing response
   counts (default 100 and 1,000, resampled from real responses) to measure
   throughput and cost per 1k responses. Rate-limit pauses are reported
   separately so raw latency is not obscured by waiting. No accuracy:
   resampled data has no ground truth.

Finishes with a combined verdict and writes everything to one JSON file.

Usage:
    uv run python evals/systemone_benchmark.py --llm-model gpt-4o-mini
    uv run python evals/systemone_benchmark.py --llm-model gpt-4o-mini --limit 20
    uv run python evals/systemone_benchmark.py --dry-run     # scale cost estimate
    uv run python evals/systemone_benchmark.py --skip-e2e --skip-stages  # scale only

Environment: as evals/compare_systemone.py (TYPESAFE_API_KEY plus the LLM
variables; the LLM is not needed when both LLM phases are skipped).
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from compare_e2e import print_comparison, run_pipeline  # noqa: E402
from compare_systemone import (  # noqa: E402
    JEV_INPUT_PRICE_PER_M,
    JEV_OUTPUT_PRICE_PER_M,
    build_llm,
    compare_question_part,
)
from datasets import DatasetConfig, load_local_mapping_data  # noqa: E402
from systemone_diagnostics import (  # noqa: E402
    build_sample_request,
    print_pipeline_flow,
    save_sample_request,
)

from themefinder import SystemOne, classify_responses_systemone  # noqa: E402
from themefinder.systemone import (  # noqa: E402
    DEFAULT_ASSIGNMENT_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_DETAIL_THRESHOLD,
)

RESAMPLE_SEED = 20260917


# ---------------------------------------------------------------- verdict


def _mean_metric(question_parts: dict, backend: str, key: str) -> float | None:
    """Average one metric for one backend across question parts."""
    values = [
        run["metrics"][key]
        for part in question_parts.values()
        for run in part["runs"]
        if run["backend"] == backend and key in run["metrics"]
    ]
    return sum(values) / len(values) if values else None


def print_verdict(
    e2e_runs: list[dict], question_parts: dict, scale_rows: list[dict]
) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    lines = []

    e2e_by_name = {run["name"]: run for run in e2e_runs}
    if "llm" in e2e_by_name and "hybrid" in e2e_by_name:
        llm_run, hybrid_run = e2e_by_name["llm"], e2e_by_name["hybrid"]
        time_change = (hybrid_run["seconds"] - llm_run["seconds"]) / llm_run["seconds"]
        cost_change = (
            (hybrid_run["cost_usd"] - llm_run["cost_usd"]) / llm_run["cost_usd"]
            if llm_run["cost_usd"]
            else 0
        )
        lines.append(
            f"End-to-end: hybrid {hybrid_run['seconds']:.1f}s vs LLM "
            f"{llm_run['seconds']:.1f}s ({time_change:+.0%}), "
            f"${hybrid_run['cost_usd']:.4f} vs ${llm_run['cost_usd']:.4f} "
            f"({cost_change:+.0%})"
        )

    for label, key in [
        ("Mapping F1", "map_f1_score"),
        ("Evidence accuracy", "detail_accuracy"),
    ]:
        llm_value = _mean_metric(question_parts, "llm", key)
        s1_value = _mean_metric(question_parts, "systemone", key)
        if llm_value is not None and s1_value is not None:
            winner = "SystemOne" if s1_value >= llm_value else "LLM"
            lines.append(
                f"{label} (mean, vs ground truth): SystemOne {s1_value:.3f} vs "
                f"LLM {llm_value:.3f} — {winner} ahead"
            )

    if scale_rows:
        biggest = max(scale_rows, key=lambda row: row["responses"])
        lines.append(
            f"Scale: {biggest['responses']:,} responses in "
            f"{biggest['seconds']:.1f}s "
            f"({biggest['responses_per_second_active']:,.0f}/s active, "
            f"{biggest['rate_limit_hits']} rate-limit hits), "
            f"${biggest['cost_per_1k_responses_usd']:.4f} per 1k responses"
        )

    if lines:
        console.print(Panel("\n".join(lines), title="Verdict", expand=False))


# ---------------------------------------------------------------- scale phase


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


def print_scale_plan(
    sizes: list[int],
    base_df: pd.DataFrame,
    question: str,
    themes_df: pd.DataFrame,
    batch_size: int,
) -> None:
    """Estimate scale-phase tokens and cost from a sample request payload."""
    sample = build_sample_request(
        scale_responses(base_df, batch_size), question, themes_df, batch_size
    )
    # ~4 chars per token is a serviceable estimate for English + JSON.
    tokens_per_request = len(json.dumps(sample)) // 4
    print(f"Scale phase plan (batch size {batch_size}):")
    total_cost = 0.0
    for size in sizes:
        requests = -(-size // batch_size)
        tokens = tokens_per_request * requests
        cost = tokens * JEV_INPUT_PRICE_PER_M / 1_000_000
        total_cost += cost
        print(
            f"  {size:>7,} responses → {requests:>4} requests, "
            f"~{tokens / 1000:,.0f}k input tokens, ~${cost:.3f}"
        )
    print(f"  Total estimated cost: ~${total_cost:.3f}")


async def run_scale_size(
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
    pause_before = client.rate_limit_pause_seconds

    start = time.perf_counter()
    _, unprocessable_df = await classify_responses_systemone(
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
    pause_seconds = min(client.rate_limit_pause_seconds - pause_before, seconds)
    active_seconds = max(seconds - pause_seconds, 1e-9)
    return {
        "responses": size,
        "requests": -(-size // batch_size),
        "seconds": round(seconds, 2),
        "rate_limit_pause_seconds": round(pause_seconds, 2),
        "active_seconds": round(active_seconds, 2),
        "responses_per_second_active": round(size / active_seconds, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
        "cost_per_1k_responses_usd": round(cost / size * 1000, 4),
        "unprocessable": len(unprocessable_df),
        "rate_limit_hits": client.rate_limit_hits - rate_limits_before,
    }


def print_scale_results(rows: list[dict], concurrency: int, batch_size: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="SystemOne classification at scale",
        caption=(
            f"batch size {batch_size}, concurrency {concurrency}. Responses "
            "resampled from real data (throughput only, no accuracy). "
            "Resp/sec is computed on active time, i.e. wall time minus 429 "
            "pauses, so it reflects raw latency rather than waiting."
        ),
    )
    for column in [
        "Responses",
        "Requests",
        "Wall time",
        "429 pause",
        "Resp/sec (active)",
        "Input tokens",
        "Cost",
        "Cost / 1k resp",
        "429s",
        "Dropped",
    ]:
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            f"{row['responses']:,}",
            str(row["requests"]),
            f"{row['seconds']:.1f} s",
            f"{row['rate_limit_pause_seconds']:.1f} s",
            f"{row['responses_per_second_active']:,.0f}",
            f"{row['input_tokens']:,}",
            f"${row['cost_usd']:.4f}",
            f"${row['cost_per_1k_responses_usd']:.4f}",
            str(row.get("rate_limit_hits", 0)),
            str(row.get("unprocessable", 0)),
        )
    console.print(table)

    if rows:
        best = max(rows, key=lambda row: row["responses"])
        throughput = best["responses_per_second_active"]
        cost_per_1k = best["cost_per_1k_responses_usd"]
        console.print(
            f"Projection at the active rate (excludes rate-limit pauses): "
            f"100,000 responses ≈ {100_000 / throughput / 60:.1f} min, "
            f"${cost_per_1k * 100:.2f}. Sustained runs at this scale will be "
            "paced by the account's rate limit."
        )


# ---------------------------------------------------------------- aggregation


def _mean_std(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    std = (
        (sum((value - mean) ** 2 for value in values) / (n - 1)) ** 0.5
        if n > 1
        else 0.0
    )
    return {"mean": mean, "std": std, "n": n}


STAGE_AGGREGATE_METRICS = [
    ("map_f1_score", "Mapping F1"),
    ("detail_accuracy", "Evidence acc"),
    ("detail_auc", "Evidence AUC"),
]


def aggregate_runs(runs: list[dict]) -> dict:
    """Collect per-run values into mean/std summaries across repeats."""
    aggregate: dict = {"e2e": {}, "stage": {}, "scale": {}}

    e2e_values: dict = {}
    for run in runs:
        for pipeline in run["e2e"]:
            entry = e2e_values.setdefault(pipeline["name"], {"seconds": [], "cost_usd": []})
            entry["seconds"].append(pipeline["seconds"])
            entry["cost_usd"].append(pipeline["cost_usd"])
    aggregate["e2e"] = {
        name: {key: _mean_std(values) for key, values in entries.items()}
        for name, entries in e2e_values.items()
    }

    stage_values: dict = {}
    for run in runs:
        for part, part_data in run["question_parts"].items():
            for stage_run in part_data["runs"]:
                key = (part, stage_run["backend"])
                entry = stage_values.setdefault(
                    key, {"seconds": [], "cost_usd": [], "metrics": {}}
                )
                entry["seconds"].append(stage_run["seconds"])
                entry["cost_usd"].append(stage_run["cost_usd"])
                for metric, value in stage_run["metrics"].items():
                    entry["metrics"].setdefault(metric, []).append(value)
    aggregate["stage"] = {
        f"{part}/{backend}": {
            "seconds": _mean_std(entry["seconds"]),
            "cost_usd": _mean_std(entry["cost_usd"]),
            "metrics": {
                metric: _mean_std(values)
                for metric, values in entry["metrics"].items()
            },
        }
        for (part, backend), entry in stage_values.items()
    }

    scale_values: dict = {}
    for run in runs:
        for row in run["scale"]:
            entry = scale_values.setdefault(
                row["responses"],
                {"seconds": [], "responses_per_second_active": [], "rate_limit_hits": []},
            )
            entry["seconds"].append(row["seconds"])
            entry["responses_per_second_active"].append(
                row["responses_per_second_active"]
            )
            entry["rate_limit_hits"].append(row["rate_limit_hits"])
    aggregate["scale"] = {
        str(size): {key: _mean_std(values) for key, values in entries.items()}
        for size, entries in scale_values.items()
    }

    return aggregate


def print_aggregate(aggregate: dict, n_runs: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    def cell(stats: dict, fmt: str = "{:.3f}") -> str:
        return f"{fmt.format(stats['mean'])} ± {fmt.format(stats['std'])}"

    if aggregate["stage"]:
        table = Table(title=f"Aggregate over {n_runs} runs — stage-level")
        table.add_column("Part / backend")
        for _, label in STAGE_AGGREGATE_METRICS:
            table.add_column(label, justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Cost (USD)", justify="right")
        for key, entry in sorted(aggregate["stage"].items()):
            row = [key]
            for metric, _ in STAGE_AGGREGATE_METRICS:
                stats = entry["metrics"].get(metric)
                row.append(cell(stats) if stats else "—")
            row.append(cell(entry["seconds"], "{:.1f}"))
            row.append(cell(entry["cost_usd"], "{:.4f}"))
            table.add_row(*row)
        console.print(table)

    if aggregate["e2e"]:
        table = Table(title=f"Aggregate over {n_runs} runs — end-to-end")
        table.add_column("Pipeline")
        table.add_column("Time (s)", justify="right")
        table.add_column("Cost (USD)", justify="right")
        for name, entry in sorted(aggregate["e2e"].items()):
            table.add_row(
                name, cell(entry["seconds"], "{:.1f}"), cell(entry["cost_usd"], "{:.4f}")
            )
        console.print(table)

    if aggregate["scale"]:
        table = Table(title=f"Aggregate over {n_runs} runs — scale")
        table.add_column("Responses", justify="right")
        table.add_column("Wall time (s)", justify="right")
        table.add_column("Resp/sec (active)", justify="right")
        table.add_column("429s (mean)", justify="right")
        for size, entry in sorted(
            aggregate["scale"].items(), key=lambda item: int(item[0])
        ):
            table.add_row(
                f"{int(size):,}",
                cell(entry["seconds"], "{:.1f}"),
                cell(entry["responses_per_second_active"], "{:.0f}"),
                f"{entry['rate_limit_hits']['mean']:.1f}",
            )
        console.print(table)


# ---------------------------------------------------------------- one run


async def run_once(
    args,
    items: list[dict],
    config: DatasetConfig,
    llm,
    systemone_client: SystemOne,
    scale_sizes: list[int],
    first_item: dict,
    first_responses: pd.DataFrame,
    first_question: str,
    first_themes: pd.DataFrame,
    show_detail: bool = True,
) -> dict:
    """Run the three phases once; show_detail limits repeated boilerplate."""
    # Phase 1: end-to-end totals on one question part (both pipelines share
    # the generative stages, so one part suffices for the overall picture).
    e2e_runs: list[dict] = []
    if not args.skip_e2e:
        e2e_part = first_item["metadata"]["question_part"]
        print(f"\n{'=' * 20} Phase 1: end-to-end ({e2e_part}) {'=' * 20}")
        responses_df = first_responses
        if args.limit:
            responses_df = responses_df.head(args.limit)
        e2e_runs.append(
            await run_pipeline("llm", responses_df, first_question, llm, None)
        )
        e2e_runs.append(
            await run_pipeline(
                "hybrid", responses_df, first_question, llm, systemone_client
            )
        )
        print_comparison(e2e_part, e2e_runs)

    # Phase 2: stage-level accuracy against ground truth, all selected parts.
    question_parts: dict = {}
    if not args.skip_stages:
        print(f"\n{'=' * 20} Phase 2: stage-level accuracy {'=' * 20}")
        seen_caveats: set = set()
        for i, item in enumerate(items):
            question_parts[item["metadata"]["question_part"]] = (
                await compare_question_part(
                    item,
                    config,
                    llm,
                    systemone_client,
                    limit=args.limit,
                    llm_concurrency=args.concurrency,
                    mapping_threshold=args.mapping_threshold,
                    detail_threshold=args.detail_threshold,
                    systemone_concurrency=args.systemone_concurrency,
                    batch_size=args.systemone_batch_size,
                    show_structure=(show_detail and i == 0),
                    seen_caveats=seen_caveats,
                )
            )

    # Phase 3: throughput at scale, resampled from the first question part.
    scale_rows: list[dict] = []
    if not args.skip_scale:
        print(f"\n{'=' * 20} Phase 3: scale {'=' * 20}")
        for size in scale_sizes:
            print(f"Running {size:,} responses...")
            scale_rows.append(
                await run_scale_size(
                    size,
                    first_responses,
                    first_question,
                    first_themes,
                    systemone_client,
                    args.systemone_batch_size,
                    args.systemone_concurrency,
                )
            )
        print_scale_results(
            scale_rows, args.systemone_concurrency, args.systemone_batch_size
        )

    print()
    print_verdict(e2e_runs, question_parts, scale_rows)
    return {
        "e2e": [{**run, "seconds": round(run["seconds"], 2)} for run in e2e_runs],
        "question_parts": question_parts,
        "scale": scale_rows,
    }


# ---------------------------------------------------------------- main


async def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="End-to-end, stage-level and scale SystemOne benchmark"
    )
    parser.add_argument("--dataset", default="gambling_XS")
    parser.add_argument(
        "--question",
        type=int,
        default=None,
        help="Specific question part (default: e2e/scale on part 1, stages on all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Subsample to N responses (phases 1-2)"
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT"),
        help="OpenAI model/deployment name",
    )
    parser.add_argument("--llm-api", choices=["chat", "responses"], default=None)
    parser.add_argument(
        "--model", default=None, help="SystemOne model override (default jev-latest)"
    )
    parser.add_argument("--mapping-threshold", type=float, default=DEFAULT_ASSIGNMENT_THRESHOLD)
    parser.add_argument("--detail-threshold", type=float, default=DEFAULT_DETAIL_THRESHOLD)
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent LLM calls")
    parser.add_argument("--systemone-concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--systemone-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--scale-sizes",
        default="100,1000",
        help=(
            "Comma-separated response counts for the scale phase (10,000+ "
            "currently exceeds the sustainable rate; expect 429 pauses)"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run every phase N times and report mean ± std across runs",
    )
    parser.add_argument("--skip-e2e", action="store_true", help="Skip the end-to-end phase")
    parser.add_argument("--skip-stages", action="store_true", help="Skip the stage-level phase")
    parser.add_argument("--skip-scale", action="store_true", help="Skip the scale phase")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scale-phase request plan and estimated cost, then exit",
    )
    args = parser.parse_args()
    scale_sizes = [int(size) for size in args.scale_sizes.split(",")]

    config = DatasetConfig(dataset=args.dataset, stage="mapping")
    items = load_local_mapping_data(config)
    if args.question is not None:
        items = [
            item
            for item in items
            if f"part_{args.question}" in item["metadata"]["question_part"]
        ]
    if not items:
        sys.exit(f"No matching question parts in dataset {args.dataset}")

    first_item = items[0]
    first_responses = pd.DataFrame(first_item["input"]["responses"])
    first_question = first_item["input"]["question"]
    first_themes = pd.DataFrame(first_item["input"]["topics"])[["topic_id", "topic"]]

    if args.dry_run:
        print_scale_plan(
            scale_sizes,
            first_responses,
            first_question,
            first_themes,
            args.systemone_batch_size,
        )
        return

    needs_llm = not (args.skip_e2e and args.skip_stages)
    llm = None
    if needs_llm:
        if not args.llm_model:
            sys.exit(
                "No LLM model configured: pass --llm-model or set "
                "AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT (or skip the LLM phases)."
            )
        llm = build_llm(args.llm_model, args.llm_api)
    systemone_client = SystemOne.from_env(model=args.model)

    print_pipeline_flow(
        n_themes=len(first_item["input"]["topics"]),
        batch_size=args.systemone_batch_size,
    )

    all_runs: list[dict] = []
    for repeat in range(args.repeats):
        if args.repeats > 1:
            print(f"\n{'#' * 24} Run {repeat + 1} of {args.repeats} {'#' * 24}")
        run_results = await run_once(
            args,
            items,
            config,
            llm,
            systemone_client,
            scale_sizes,
            first_item,
            first_responses,
            first_question,
            first_themes,
            show_detail=(repeat == 0),
        )
        all_runs.append(run_results)

    aggregate = None
    if args.repeats > 1:
        print()
        aggregate = aggregate_runs(all_runs)
        print_aggregate(aggregate, args.repeats)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_path = results_dir / f"systemone_sample_request_{timestamp}.json"
    save_sample_request(
        first_item, args.systemone_batch_size, sample_path, limit=args.limit
    )
    print(f"\nSample SystemOne request written to {sample_path}")
    output_path = results_dir / f"systemone_benchmark_{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "llm_model": args.llm_model,
                "mapping_threshold": args.mapping_threshold,
                "detail_threshold": args.detail_threshold,
                "batch_size": args.systemone_batch_size,
                "repeats": args.repeats,
                "timestamp": datetime.now().isoformat(),
                "runs": all_runs,
                "aggregate": aggregate,
            },
            indent=2,
        )
    )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
