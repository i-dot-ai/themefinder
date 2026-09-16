"""Compare the LLM pipeline against SystemOne (jev) on classification stages.

Runs theme mapping and detail detection through the regular LLM stages, and
through the SystemOne implementation (both stages batched into one request
per group of responses), and reports speed, token usage, cost, accuracy
against ground truth, and agreement between the two.

Ground truth comes from the local eval datasets (e.g. evals/data/gambling_XS),
the same data the existing mapping eval uses.

Usage:
    uv run python evals/compare_systemone.py --llm-model gpt-4o-mini
    uv run python evals/compare_systemone.py --skip-llm          # SystemOne only
    uv run python evals/compare_systemone.py --limit 10          # subsample

Environment:
    TYPESAFE_API_KEY                 SystemOne (TypeSafe) API key
    AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT  LLM deployment name (as in existing evals)
    LLM_GATEWAY_URL                  LLM endpoint base URL
    CONSULT_EVAL_LITELLM_API_KEY     LLM API key
    LLM_INPUT_PRICE_PER_M            Override LLM input price (USD per 1M tokens)
    LLM_OUTPUT_PRICE_PER_M           Override LLM output price (USD per 1M tokens)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from datasets import DatasetConfig, load_local_mapping_data  # noqa: E402
from metrics import calculate_mapping_metrics  # noqa: E402

from themefinder import (  # noqa: E402
    OpenAILLM,
    SystemOne,
    classify_responses_systemone,
    detail_detection,
    theme_mapping,
)
from themefinder.systemone import (  # noqa: E402
    DEFAULT_ASSIGNMENT_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_DETAIL_THRESHOLD,
)

# USD per 1M tokens. jev pricing from docs.typesafe.ai (jev-1.12, September 2026):
# input $0.042/1M, output free. LLM default assumes GPT-4.1; override via env.
JEV_INPUT_PRICE_PER_M = 0.042
JEV_OUTPUT_PRICE_PER_M = 0.0
DEFAULT_LLM_INPUT_PRICE_PER_M = 2.00
DEFAULT_LLM_OUTPUT_PRICE_PER_M = 8.00


@dataclass
class StageRun:
    """Measurements for one stage run through one backend."""

    backend: str
    stage: str
    seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metrics: dict

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "stage": self.stage,
            "seconds": round(self.seconds, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "metrics": self.metrics,
        }


def llm_prices() -> tuple[float, float]:
    return (
        float(os.getenv("LLM_INPUT_PRICE_PER_M", DEFAULT_LLM_INPUT_PRICE_PER_M)),
        float(os.getenv("LLM_OUTPUT_PRICE_PER_M", DEFAULT_LLM_OUTPUT_PRICE_PER_M)),
    )


def cost_usd(input_tokens: int, output_tokens: int, prices: tuple[float, float]) -> float:
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def load_detail_ground_truth(config: DatasetConfig, question_part: str) -> dict[int, str]:
    """Load expected evidence_rich labels ({response_id: "YES"/"NO"})."""
    outputs_dir = config.local_path / "outputs" / "mapping"
    date_dirs = sorted(outputs_dir.iterdir(), reverse=True)
    detail_path = date_dirs[0] / question_part / "detail_detection.jsonl"
    if not detail_path.exists():
        return {}
    df = pd.read_json(detail_path, lines=True)
    return dict(zip(df["response_id"].astype(int), df["evidence_rich"]))


def mapping_accuracy_metrics(
    result_df: pd.DataFrame, expected: dict[str, list[str]]
) -> dict:
    """Score predicted theme labels against the expected mapping."""
    if result_df.empty or "response_id" not in result_df.columns:
        return {}
    df = result_df.copy()
    df["expected"] = df["response_id"].astype(str).map(expected)
    df = df[df["expected"].notna()]
    if df.empty:
        return {}
    metrics = calculate_mapping_metrics(df, column_one="expected", column_two="labels")
    return {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }


def mapping_agreement(llm_df: pd.DataFrame, systemone_df: pd.DataFrame) -> dict:
    """Score the two backends' theme labels against each other."""
    if (
        llm_df.empty
        or systemone_df.empty
        or "response_id" not in llm_df.columns
        or "response_id" not in systemone_df.columns
    ):
        return {}
    merged = llm_df[["response_id", "labels"]].merge(
        systemone_df[["response_id", "labels"]],
        on="response_id",
        suffixes=("_llm", "_systemone"),
    )
    if merged.empty:
        return {}
    metrics = calculate_mapping_metrics(
        merged, column_one="labels_llm", column_two="labels_systemone"
    )
    return {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }


def detail_accuracy_metrics(result_df: pd.DataFrame, expected: dict[int, str]) -> dict:
    """Score predicted evidence_rich labels against the expected labels."""
    if not expected or result_df.empty or "response_id" not in result_df.columns:
        return {}
    df = result_df.copy()
    df["expected"] = df["response_id"].astype(int).map(expected)
    df = df[df["expected"].notna()]
    if df.empty:
        return {}
    accuracy = float((df["evidence_rich"] == df["expected"]).mean())
    kappa = float(cohen_kappa_score(df["expected"], df["evidence_rich"]))
    metrics = {"accuracy": accuracy, "cohen_kappa": kappa}
    # AUC (threshold-free) shows whether probabilities rank responses
    # correctly even when the threshold classifies them all one way.
    if "evidence_probability" in df.columns and df["expected"].nunique() > 1:
        expected_yes = df["expected"] == "YES"
        probabilities = df["evidence_probability"]
        metrics["auc"] = float(roc_auc_score(expected_yes, probabilities))
        # Diagnostic threshold sweep: where should the cut-off actually sit?
        best_threshold, best_accuracy = max(
            (
                (candidate, float((expected_yes == (probabilities >= candidate)).mean()))
                for candidate in sorted(probabilities.unique())
            ),
            key=lambda pair: pair[1],
        )
        metrics["best_threshold"] = float(best_threshold)
        metrics["best_thr_accuracy"] = best_accuracy
    return metrics


async def run_llm_stages(
    llm: OpenAILLM,
    responses_df: pd.DataFrame,
    question: str,
    topics_df: pd.DataFrame,
    expected_mapping: dict[str, list[str]],
    expected_detail: dict[int, str],
    concurrency: int,
) -> tuple[list[StageRun], pd.DataFrame]:
    """Run mapping and detail detection through the LLM, measuring as we go."""
    prices = llm_prices()
    runs = []

    before = (llm.usage.input_tokens, llm.usage.output_tokens)
    start = time.perf_counter()
    mapping_df, unprocessable = await theme_mapping(
        responses_df=responses_df[["response_id", "response"]],
        llm=llm,
        question=question,
        refined_themes_df=topics_df[["topic_id", "topic"]],
        concurrency=concurrency,
    )
    seconds = time.perf_counter() - start
    if not unprocessable.empty:
        print(f"  Warning: {len(unprocessable)} responses unprocessable (LLM mapping)")
    input_tokens = llm.usage.input_tokens - before[0]
    output_tokens = llm.usage.output_tokens - before[1]
    runs.append(
        StageRun(
            backend="llm",
            stage="mapping",
            seconds=seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(input_tokens, output_tokens, prices),
            metrics=mapping_accuracy_metrics(mapping_df, expected_mapping),
        )
    )

    before = (llm.usage.input_tokens, llm.usage.output_tokens)
    start = time.perf_counter()
    detail_df, _ = await detail_detection(
        responses_df=responses_df[["response_id", "response"]],
        llm=llm,
        question=question,
        concurrency=concurrency,
    )
    seconds = time.perf_counter() - start
    input_tokens = llm.usage.input_tokens - before[0]
    output_tokens = llm.usage.output_tokens - before[1]
    runs.append(
        StageRun(
            backend="llm",
            stage="detail_detection",
            seconds=seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(input_tokens, output_tokens, prices),
            metrics=detail_accuracy_metrics(detail_df, expected_detail),
        )
    )

    return runs, mapping_df


async def run_systemone_stage(
    client: SystemOne,
    responses_df: pd.DataFrame,
    question: str,
    topics_df: pd.DataFrame,
    expected_mapping: dict[str, list[str]],
    expected_detail: dict[int, str],
    threshold: float,
    detail_threshold: float,
    concurrency: int,
    batch_size: int,
) -> tuple[StageRun, pd.DataFrame]:
    """Run the combined SystemOne classification, measuring as we go."""
    before = (client.usage.input_tokens, client.usage.output_tokens)
    start = time.perf_counter()
    classified_df, unprocessable = await classify_responses_systemone(
        responses_df=responses_df[["response_id", "response"]],
        client=client,
        question=question,
        refined_themes_df=topics_df[["topic_id", "topic"]],
        threshold=threshold,
        detail_threshold=detail_threshold,
        concurrency=concurrency,
        batch_size=batch_size,
    )
    seconds = time.perf_counter() - start
    if not unprocessable.empty:
        print(f"  Warning: {len(unprocessable)} responses unprocessable (SystemOne)")
    input_tokens = client.usage.input_tokens - before[0]
    output_tokens = client.usage.output_tokens - before[1]

    metrics = {
        **{
            f"map_{key}": value
            for key, value in mapping_accuracy_metrics(
                classified_df, expected_mapping
            ).items()
        },
        **{
            f"detail_{key}": value
            for key, value in detail_accuracy_metrics(
                classified_df, expected_detail
            ).items()
        },
    }
    run = StageRun(
        backend="systemone",
        stage="mapping+detail",
        seconds=seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(input_tokens, output_tokens, (JEV_INPUT_PRICE_PER_M, JEV_OUTPUT_PRICE_PER_M)),
        metrics=metrics,
    )
    return run, classified_df


def print_summary(question_part: str, runs: list[StageRun], agreement: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"LLM vs SystemOne — {question_part}")
    for column in [
        "Stage",
        "Backend",
        "Time (s)",
        "Input tok",
        "Output tok",
        "Cost (USD)",
        "Quality",
    ]:
        table.add_column(column)

    for run in runs:
        quality = ", ".join(f"{key}={value:.3f}" for key, value in run.metrics.items())
        table.add_row(
            run.stage,
            run.backend,
            f"{run.seconds:.1f}",
            str(run.input_tokens),
            str(run.output_tokens),
            f"${run.cost_usd:.4f}",
            quality or "n/a",
        )
    console.print(table)

    if agreement:
        agreement_summary = ", ".join(
            f"{key}={value:.3f}" for key, value in agreement.items()
        )
        console.print(f"LLM vs SystemOne mapping agreement: {agreement_summary}")


async def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compare LLM vs SystemOne classification stages"
    )
    parser.add_argument("--dataset", default="gambling_XS")
    parser.add_argument(
        "--question", type=int, default=None, help="Specific question part number"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Subsample to N responses"
    )
    parser.add_argument(
        "--mapping-threshold",
        type=float,
        default=DEFAULT_ASSIGNMENT_THRESHOLD,
        help="Probability threshold for theme assignment",
    )
    parser.add_argument(
        "--detail-threshold",
        type=float,
        default=DEFAULT_DETAIL_THRESHOLD,
        help=(
            "Probability threshold for evidence-rich classification. Tune "
            "using the best_threshold diagnostic in the results."
        ),
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrent LLM calls"
    )
    parser.add_argument(
        "--systemone-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Concurrent SystemOne calls",
    )
    parser.add_argument(
        "--systemone-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Responses per SystemOne request",
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Only run the SystemOne stage"
    )
    parser.add_argument(
        "--skip-systemone", action="store_true", help="Only run the LLM stages"
    )
    parser.add_argument(
        "--model", default=None, help="SystemOne model override (default jev-latest)"
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT"),
        help=(
            "OpenAI model/deployment name for the LLM stages "
            "(default: AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT env var)"
        ),
    )
    parser.add_argument(
        "--llm-api",
        choices=["chat", "responses"],
        default=None,
        help=(
            "OpenAI API to use for the LLM stages. Default: 'responses' for "
            "gpt-5* models (which are Responses-API only), otherwise 'chat'."
        ),
    )
    args = parser.parse_args()

    config = DatasetConfig(dataset=args.dataset, stage="mapping")
    items = load_local_mapping_data(config)
    if args.question is not None:
        items = [
            item
            for item in items
            if f"part_{args.question}" in item["metadata"]["question_part"]
        ]

    llm = None
    if not args.skip_llm:
        if not args.llm_model:
            sys.exit(
                "No LLM model configured: pass --llm-model or set the "
                "AUTO_EVAL_4_1_SWEDEN_DEPLOYMENT env var (or use --skip-llm)."
            )
        is_gpt5_family = args.llm_model.startswith("gpt-5")
        use_responses_api = (
            args.llm_api == "responses" if args.llm_api else is_gpt5_family
        )
        # gpt-5 family models reject explicit temperature settings.
        request_kwargs = {} if is_gpt5_family else {"temperature": 0}
        llm = OpenAILLM(
            model=args.llm_model,
            request_kwargs=request_kwargs,
            use_responses_api=use_responses_api,
            base_url=os.getenv("LLM_GATEWAY_URL"),
            api_key=os.getenv("CONSULT_EVAL_LITELLM_API_KEY"),
        )
    systemone_client = None
    if not args.skip_systemone:
        systemone_client = SystemOne.from_env(model=args.model)

    all_results = {
        "dataset": args.dataset,
        "mapping_threshold": args.mapping_threshold,
        "detail_threshold": args.detail_threshold,
        "batch_size": args.systemone_batch_size,
        "timestamp": datetime.now().isoformat(),
        "question_parts": {},
    }

    for item in items:
        question_part = item["metadata"]["question_part"]
        print(f"\n=== {question_part} ===")
        responses_df = pd.DataFrame(item["input"]["responses"])
        if args.limit:
            responses_df = responses_df.head(args.limit)
        question = item["input"]["question"]
        topics_df = pd.DataFrame(item["input"]["topics"])
        expected_mapping = item["expected_output"]["mappings"]
        expected_detail = load_detail_ground_truth(config, question_part)

        runs: list[StageRun] = []
        llm_mapping_df = systemone_df = None

        if llm is not None:
            llm_runs, llm_mapping_df = await run_llm_stages(
                llm,
                responses_df,
                question,
                topics_df,
                expected_mapping,
                expected_detail,
                args.concurrency,
            )
            runs.extend(llm_runs)

        if systemone_client is not None:
            systemone_run, systemone_df = await run_systemone_stage(
                systemone_client,
                responses_df,
                question,
                topics_df,
                expected_mapping,
                expected_detail,
                args.mapping_threshold,
                args.detail_threshold,
                args.systemone_concurrency,
                args.systemone_batch_size,
            )
            runs.append(systemone_run)

        agreement = {}
        if llm_mapping_df is not None and systemone_df is not None:
            agreement = mapping_agreement(llm_mapping_df, systemone_df)

        print_summary(question_part, runs, agreement)
        all_results["question_parts"][question_part] = {
            "n_responses": len(responses_df),
            "runs": [run.as_dict() for run in runs],
            "mapping_agreement": agreement,
        }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = (
        results_dir
        / f"systemone_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
