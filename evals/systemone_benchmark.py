"""Full SystemOne benchmark: end-to-end totals, then stage-level accuracy.

One command, two phases:

1. **End-to-end** — runs the complete `find_themes` (LLM) and
   `find_themes_hybrid` (LLM + SystemOne) pipelines on the same responses,
   for the overall wall time and cost picture. Each run generates its own
   themes, so this phase reports descriptive output statistics only.
2. **Stage-level** — re-runs the classification stages (theme mapping,
   detail detection) on the dataset's fixed reference themes, where ground
   truth exists, for a real accuracy comparison (F1, evidence accuracy,
   AUC, agreement).

Finishes with a combined verdict and writes everything to one JSON file.

Usage:
    uv run python evals/systemone_benchmark.py --llm-model gpt-4o-mini
    uv run python evals/systemone_benchmark.py --llm-model gpt-4o-mini --limit 20

Environment: as evals/compare_systemone.py (TYPESAFE_API_KEY plus the LLM
variables).
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from compare_e2e import print_comparison, run_pipeline  # noqa: E402
from compare_systemone import build_llm, compare_question_part  # noqa: E402
from datasets import DatasetConfig, load_local_mapping_data  # noqa: E402
from systemone_diagnostics import (  # noqa: E402
    print_pipeline_flow,
    save_sample_request,
)

from themefinder import SystemOne  # noqa: E402
from themefinder.systemone import (  # noqa: E402
    DEFAULT_ASSIGNMENT_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_DETAIL_THRESHOLD,
)


def _mean_metric(question_parts: dict, backend: str, key: str) -> float | None:
    """Average one metric for one backend across question parts."""
    values = [
        run["metrics"][key]
        for part in question_parts.values()
        for run in part["runs"]
        if run["backend"] == backend and key in run["metrics"]
    ]
    return sum(values) / len(values) if values else None


def print_verdict(e2e_runs: list[dict], question_parts: dict) -> None:
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

    if lines:
        console.print(Panel("\n".join(lines), title="Verdict", expand=False))


async def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="End-to-end plus stage-level SystemOne benchmark"
    )
    parser.add_argument("--dataset", default="gambling_XS")
    parser.add_argument(
        "--question",
        type=int,
        default=None,
        help="Specific question part (default: e2e on part 1, stages on all parts)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Subsample to N responses"
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
        "--skip-e2e", action="store_true", help="Only run the stage-level phase"
    )
    parser.add_argument(
        "--skip-stages", action="store_true", help="Only run the end-to-end phase"
    )
    args = parser.parse_args()

    if not args.llm_model:
        sys.exit(
            "No LLM model configured: pass --llm-model or set "
            "AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT."
        )
    llm = build_llm(args.llm_model, args.llm_api)
    systemone_client = SystemOne.from_env(model=args.model)

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

    print_pipeline_flow(
        n_themes=len(items[0]["input"]["topics"]),
        batch_size=args.systemone_batch_size,
    )

    # Phase 1: end-to-end totals on one question part (both pipelines share
    # the generative stages, so one part suffices for the overall picture).
    e2e_runs: list[dict] = []
    if not args.skip_e2e:
        e2e_item = items[0]
        e2e_part = e2e_item["metadata"]["question_part"]
        print(f"\n{'=' * 20} Phase 1: end-to-end ({e2e_part}) {'=' * 20}")
        responses_df = pd.DataFrame(e2e_item["input"]["responses"])
        if args.limit:
            responses_df = responses_df.head(args.limit)
        question = e2e_item["input"]["question"]
        e2e_runs.append(
            await run_pipeline("llm", responses_df, question, llm, None)
        )
        e2e_runs.append(
            await run_pipeline(
                "hybrid", responses_df, question, llm, systemone_client
            )
        )
        print_comparison(e2e_part, e2e_runs)

    # Phase 2: stage-level accuracy against ground truth, all selected parts.
    question_parts: dict = {}
    if not args.skip_stages:
        print(f"\n{'=' * 20} Phase 2: stage-level accuracy {'=' * 20}")
        for item in items:
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
                )
            )

    print()
    print_verdict(e2e_runs, question_parts)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_path = results_dir / f"systemone_sample_request_{timestamp}.json"
    save_sample_request(
        items[0], args.systemone_batch_size, sample_path, limit=args.limit
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
                "timestamp": datetime.now().isoformat(),
                "e2e": [
                    {**run, "seconds": round(run["seconds"], 2)} for run in e2e_runs
                ],
                "question_parts": question_parts,
            },
            indent=2,
        )
    )
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
