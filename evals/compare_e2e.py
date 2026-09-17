"""End-to-end comparison: full LLM pipeline vs the SystemOne hybrid.

Runs the complete theme analysis pipeline twice on the same responses —
`find_themes` (all five stages on the LLM) and `find_themes_hybrid`
(generative stages on the LLM, classification on SystemOne) — and reports
wall time, token usage, cost and descriptive output quality for each.

Unlike evals/compare_systemone.py (which fixes the theme set and scores the
classification stages against ground truth), each end-to-end run generates
its own themes, so mapping labels are not directly comparable to ground
truth or between pipelines. This eval therefore reports descriptive
statistics (theme count, coverage, evidence-rich rate) plus the generated
themes themselves for human inspection; use compare_systemone.py for
ground-truth accuracy.

Usage:
    uv run python evals/compare_e2e.py --llm-model gpt-4o-mini
    uv run python evals/compare_e2e.py --pipeline hybrid --limit 20

Environment: as evals/compare_systemone.py.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from compare_systemone import (  # noqa: E402
    JEV_INPUT_PRICE_PER_M,
    JEV_OUTPUT_PRICE_PER_M,
    build_llm,
    cost_usd,
    llm_prices,
)
from datasets import DatasetConfig, load_local_mapping_data  # noqa: E402

from themefinder import SystemOne, find_themes, find_themes_hybrid  # noqa: E402


def output_stats(result: dict) -> dict:
    """Descriptive statistics of one pipeline run's output."""
    themes_df = result["themes"]
    mapping_df = result["mapping"]
    detailed_df = result["detailed_responses"]
    stats = {
        "themes": len(themes_df),
        "responses_mapped": len(mapping_df),
        "unprocessable": len(result["unprocessables"]),
    }
    if not mapping_df.empty:
        topic_ids = set(themes_df["topic_id"])
        on_theme = mapping_df["labels"].apply(
            lambda labels: any(label in topic_ids for label in labels)
        )
        stats["coverage"] = float(on_theme.mean())
        stats["labels_per_response"] = float(mapping_df["labels"].apply(len).mean())
    if not detailed_df.empty:
        stats["evidence_rich_rate"] = float(
            (detailed_df["evidence_rich"] == "YES").mean()
        )
    return stats


async def run_pipeline(
    name: str,
    responses_df: pd.DataFrame,
    question: str,
    llm,
    systemone_client: SystemOne | None,
) -> dict:
    """Run one full pipeline, measuring time, tokens and cost."""
    print(f"\n--- Running {name} pipeline on {len(responses_df)} responses ---")
    llm_before = (llm.usage.input_tokens, llm.usage.output_tokens)
    s1_before = (
        (systemone_client.usage.input_tokens, systemone_client.usage.output_tokens)
        if systemone_client
        else (0, 0)
    )

    start = time.perf_counter()
    if systemone_client is None:
        result = await find_themes(responses_df, llm, question, verbose=False)
    else:
        result = await find_themes_hybrid(
            responses_df, llm, systemone_client, question, verbose=False
        )
    seconds = time.perf_counter() - start

    llm_input = llm.usage.input_tokens - llm_before[0]
    llm_output = llm.usage.output_tokens - llm_before[1]
    cost = cost_usd(llm_input, llm_output, llm_prices())
    s1_input = s1_output = 0
    if systemone_client:
        s1_input = systemone_client.usage.input_tokens - s1_before[0]
        s1_output = systemone_client.usage.output_tokens - s1_before[1]
        cost += cost_usd(
            s1_input, s1_output, (JEV_INPUT_PRICE_PER_M, JEV_OUTPUT_PRICE_PER_M)
        )

    return {
        "name": name,
        "seconds": seconds,
        "llm_input_tokens": llm_input,
        "llm_output_tokens": llm_output,
        "systemone_input_tokens": s1_input,
        "systemone_output_tokens": s1_output,
        "cost_usd": cost,
        "stats": output_stats(result),
        "themes": result["themes"]["topic"].tolist()
        if not result["themes"].empty
        else [],
    }


def print_comparison(question_part: str, pipeline_runs: list[dict]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    rows = [
        ("Time (s)", "seconds", "{:.1f}"),
        ("LLM input tokens", "llm_input_tokens", "{:,.0f}"),
        ("LLM output tokens", "llm_output_tokens", "{:,.0f}"),
        ("SystemOne input tokens", "systemone_input_tokens", "{:,.0f}"),
        ("Cost (USD)", "cost_usd", "${:.4f}"),
        ("Themes generated", ("stats", "themes"), "{:.0f}"),
        ("Responses mapped", ("stats", "responses_mapped"), "{:.0f}"),
        ("Theme coverage", ("stats", "coverage"), "{:.3f}"),
        ("Labels per response", ("stats", "labels_per_response"), "{:.2f}"),
        ("Evidence-rich rate", ("stats", "evidence_rich_rate"), "{:.3f}"),
        ("Unprocessable", ("stats", "unprocessable"), "{:.0f}"),
    ]

    table = Table(title=f"End-to-end pipeline comparison — {question_part}")
    table.add_column("Metric")
    for run in pipeline_runs:
        table.add_column(run["name"], justify="right")
    for label, key, fmt in rows:
        cells = []
        for run in pipeline_runs:
            value = (
                run["stats"].get(key[1]) if isinstance(key, tuple) else run.get(key)
            )
            cells.append(fmt.format(value) if value is not None else "—")
        if any(cell != "—" for cell in cells):
            table.add_row(label, *cells)
    console.print(table)

    for run in pipeline_runs:
        console.print(f"\n[bold]{run['name']} themes:[/]")
        for topic in run["themes"]:
            console.print(f"  • {topic}")


async def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="End-to-end comparison of the LLM and SystemOne hybrid pipelines"
    )
    parser.add_argument("--dataset", default="gambling_XS")
    parser.add_argument(
        "--question", type=int, default=1, help="Question part number to run"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Subsample to N responses"
    )
    parser.add_argument(
        "--pipeline",
        choices=["llm", "hybrid", "both"],
        default="both",
        help="Which pipeline(s) to run",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="OpenAI model/deployment name (default: AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT)",
    )
    parser.add_argument("--llm-api", choices=["chat", "responses"], default=None)
    parser.add_argument(
        "--model", default=None, help="SystemOne model override (default jev-latest)"
    )
    args = parser.parse_args()

    llm_model = args.llm_model or os.getenv("AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT")
    if not llm_model:
        sys.exit(
            "No LLM model configured: pass --llm-model or set "
            "AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT (the generative stages always "
            "need an LLM, even in the hybrid pipeline)."
        )
    llm = build_llm(llm_model, args.llm_api)

    config = DatasetConfig(dataset=args.dataset, stage="mapping")
    items = [
        item
        for item in load_local_mapping_data(config)
        if f"part_{args.question}" in item["metadata"]["question_part"]
    ]
    if not items:
        sys.exit(f"No question part {args.question} in dataset {args.dataset}")
    item = items[0]
    question_part = item["metadata"]["question_part"]
    responses_df = pd.DataFrame(item["input"]["responses"])
    if args.limit:
        responses_df = responses_df.head(args.limit)
    question = item["input"]["question"]

    pipeline_runs = []
    if args.pipeline in ("llm", "both"):
        pipeline_runs.append(
            await run_pipeline("llm", responses_df, question, llm, None)
        )
    if args.pipeline in ("hybrid", "both"):
        systemone_client = SystemOne.from_env(model=args.model)
        pipeline_runs.append(
            await run_pipeline("hybrid", responses_df, question, llm, systemone_client)
        )

    print_comparison(question_part, pipeline_runs)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = (
        results_dir / f"e2e_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "question_part": question_part,
                "n_responses": len(responses_df),
                "llm_model": llm_model,
                "timestamp": datetime.now().isoformat(),
                "runs": [
                    {**run, "seconds": round(run["seconds"], 2)}
                    for run in pipeline_runs
                ],
            },
            indent=2,
        )
    )
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
