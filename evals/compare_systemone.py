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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))

from datasets import (  # noqa: E402
    DatasetConfig,
    load_detail_ground_truth,
    load_local_mapping_data,
)
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

# USD per 1M tokens, overridable via env. jev pricing from docs.typesafe.ai
# (jev-1.12, September 2026): input $0.042/1M, output free. LLM default
# assumes GPT-4.1 — override when benchmarking a different model.
JEV_INPUT_PRICE_PER_M = float(os.getenv("JEV_INPUT_PRICE_PER_M", 0.042))
JEV_OUTPUT_PRICE_PER_M = float(os.getenv("JEV_OUTPUT_PRICE_PER_M", 0.0))
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
            **asdict(self),
            "seconds": round(self.seconds, 2),
            "cost_usd": round(self.cost_usd, 6),
        }


def build_llm(llm_model: str, llm_api: str | None = None) -> OpenAILLM:
    """Build the eval LLM client, handling gpt-5 family quirks.

    gpt-5* models are served only through the Responses API and reject
    explicit temperature settings; other models default to Chat Completions
    with temperature 0. Pass llm_api ("chat"/"responses") to override.
    """
    is_gpt5_family = llm_model.startswith("gpt-5")
    use_responses_api = llm_api == "responses" if llm_api else is_gpt5_family
    request_kwargs = {} if is_gpt5_family else {"temperature": 0}
    return OpenAILLM(
        model=llm_model,
        request_kwargs=request_kwargs,
        use_responses_api=use_responses_api,
        base_url=os.getenv("LLM_GATEWAY_URL"),
        api_key=os.getenv("CONSULT_EVAL_LITELLM_API_KEY"),
    )


def llm_prices() -> tuple[float, float]:
    return (
        float(os.getenv("LLM_INPUT_PRICE_PER_M", DEFAULT_LLM_INPUT_PRICE_PER_M)),
        float(os.getenv("LLM_OUTPUT_PRICE_PER_M", DEFAULT_LLM_OUTPUT_PRICE_PER_M)),
    )


def cost_usd(input_tokens: int, output_tokens: int, prices: tuple[float, float]) -> float:
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _numeric_metrics(metrics: dict) -> dict:
    return {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }


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
    return _numeric_metrics(
        calculate_mapping_metrics(df, column_one="expected", column_two="labels")
    )


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
    return _numeric_metrics(
        calculate_mapping_metrics(
            merged, column_one="labels_llm", column_two="labels_systemone"
        )
    )


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
        # roc_curve gives cumulative TP/FP rates per candidate threshold, from
        # which accuracy at each threshold follows in one pass.
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            expected_yes, probabilities
        )
        n_yes = int(expected_yes.sum())
        n_no = len(expected_yes) - n_yes
        accuracies = (
            true_positive_rate * n_yes + (1 - false_positive_rate) * n_no
        ) / len(expected_yes)
        best = accuracies.argmax()
        metrics["best_threshold"] = float(thresholds[best])
        metrics["best_thr_accuracy"] = float(accuracies[best])
    return metrics


async def _measure(usage, coroutine):
    """Await a stage, returning (result, seconds, input/output token deltas)."""
    input_before, output_before = usage.input_tokens, usage.output_tokens
    start = time.perf_counter()
    result = await coroutine
    return (
        result,
        time.perf_counter() - start,
        usage.input_tokens - input_before,
        usage.output_tokens - output_before,
    )


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

    (mapping_df, unprocessable), seconds, input_tokens, output_tokens = await _measure(
        llm.usage,
        theme_mapping(
            responses_df=responses_df[["response_id", "response"]],
            llm=llm,
            question=question,
            refined_themes_df=topics_df[["topic_id", "topic"]],
            concurrency=concurrency,
        ),
    )
    if not unprocessable.empty:
        print(f"  Warning: {len(unprocessable)} responses unprocessable (LLM mapping)")
    runs = [
        StageRun(
            backend="llm",
            stage="mapping",
            seconds=seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(input_tokens, output_tokens, prices),
            metrics={
                f"map_{key}": value
                for key, value in mapping_accuracy_metrics(
                    mapping_df, expected_mapping
                ).items()
            },
        )
    ]

    (detail_df, _), seconds, input_tokens, output_tokens = await _measure(
        llm.usage,
        detail_detection(
            responses_df=responses_df[["response_id", "response"]],
            llm=llm,
            question=question,
            concurrency=concurrency,
        ),
    )
    runs.append(
        StageRun(
            backend="llm",
            stage="detail_detection",
            seconds=seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(input_tokens, output_tokens, prices),
            metrics={
                f"detail_{key}": value
                for key, value in detail_accuracy_metrics(
                    detail_df, expected_detail
                ).items()
            },
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
    (
        (classified_df, unprocessable),
        seconds,
        input_tokens,
        output_tokens,
    ) = await _measure(
        client.usage,
        classify_responses_systemone(
            responses_df=responses_df[["response_id", "response"]],
            client=client,
            question=question,
            refined_themes_df=topics_df[["topic_id", "topic"]],
            threshold=threshold,
            detail_threshold=detail_threshold,
            concurrency=concurrency,
            batch_size=batch_size,
        ),
    )
    if not unprocessable.empty:
        print(f"  Warning: {len(unprocessable)} responses unprocessable (SystemOne)")

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
        cost_usd=cost_usd(
            input_tokens, output_tokens, (JEV_INPUT_PRICE_PER_M, JEV_OUTPUT_PRICE_PER_M)
        ),
        metrics=metrics,
    )
    return run, classified_df


# Rows of the side-by-side table: (label, key, direction, format).
# Direction "lower" = lower is better (compared as % change); "higher" =
# higher is better (compared as absolute difference).
COMPARISON_ROWS = [
    ("Time (s)", "seconds", "lower", "{:.1f}"),
    ("Input tokens", "input_tokens", "lower", "{:,.0f}"),
    ("Output tokens", "output_tokens", "lower", "{:,.0f}"),
    ("Cost (USD)", "cost_usd", "lower", "${:.4f}"),
    ("Mapping F1", "map_f1_score", "higher", "{:.3f}"),
    ("Mapping accuracy", "map_accuracy_score", "higher", "{:.3f}"),
    ("Mapping overlap", "map_overlap_rate", "higher", "{:.3f}"),
    ("Evidence accuracy", "detail_accuracy", "higher", "{:.3f}"),
    ("Evidence kappa", "detail_cohen_kappa", "higher", "{:.3f}"),
    ("Evidence AUC", "detail_auc", "higher", "{:.3f}"),
]

# Headline metrics drawn as bar charts underneath the table.
CHART_KEYS = {
    "seconds",
    "cost_usd",
    "map_f1_score",
    "map_accuracy_score",
    "detail_accuracy",
}
CHART_ROWS = [row for row in COMPARISON_ROWS if row[1] in CHART_KEYS]

BAR_WIDTH = 28
BACKEND_COLOURS = {"llm": "cyan", "systemone": "magenta"}


def _aggregate(runs: list[StageRun]) -> dict:
    """Collapse one backend's stage runs into a single comparable value set.

    Stage metrics are already namespaced (map_/detail_) by their producers,
    so aggregation is a plain sum-and-merge.
    """
    values = {
        "seconds": sum(run.seconds for run in runs),
        "input_tokens": sum(run.input_tokens for run in runs),
        "output_tokens": sum(run.output_tokens for run in runs),
        "cost_usd": sum(run.cost_usd for run in runs),
    }
    for run in runs:
        values.update(run.metrics)
    return values


def _delta_cell(llm_value: float, s1_value: float, direction: str) -> str:
    """Render the SystemOne-vs-LLM difference, green when SystemOne wins."""
    if direction == "lower":
        if llm_value == 0:
            return "—"
        change = (s1_value - llm_value) / llm_value * 100
        colour = "green" if change < 0 else "red" if change > 0 else "dim"
        return f"[{colour}]{change:+.0f}%[/]"
    difference = s1_value - llm_value
    if abs(difference) < 0.0005:
        return "[dim]±0.000[/]"
    colour = "green" if difference > 0 else "red"
    return f"[{colour}]{difference:+.3f}[/]"


def _bar(value: float, scale: float, colour: str) -> str:
    if scale <= 0:
        return "░" * BAR_WIDTH
    filled = max(1, round(BAR_WIDTH * value / scale)) if value > 0 else 0
    return f"[{colour}]{'█' * filled}[/][dim]{'░' * (BAR_WIDTH - filled)}[/]"


def print_summary(question_part: str, runs: list[StageRun], agreement: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    backends = {
        "llm": [run for run in runs if run.backend == "llm"],
        "systemone": [run for run in runs if run.backend == "systemone"],
    }
    aggregates = {
        name: _aggregate(stage_runs)
        for name, stage_runs in backends.items()
        if stage_runs
    }

    table = Table(title=f"LLM vs SystemOne — {question_part}")
    table.add_column("Metric")
    for name in aggregates:
        table.add_column(name.upper() if name == "llm" else "SystemOne", justify="right")
    if len(aggregates) == 2:
        table.add_column("SystemOne Δ", justify="right")

    for label, key, direction, fmt in COMPARISON_ROWS:
        cells = [
            fmt.format(values[key]) if key in values else "—"
            for values in aggregates.values()
        ]
        if all(cell == "—" for cell in cells):
            continue
        row = [label, *cells]
        if len(aggregates) == 2:
            llm_values, s1_values = aggregates["llm"], aggregates["systemone"]
            row.append(
                _delta_cell(llm_values[key], s1_values[key], direction)
                if key in llm_values and key in s1_values
                else "—"
            )
        table.add_row(*row)
    console.print(table)

    if len(aggregates) == 2:
        for label, key, direction, fmt in CHART_ROWS:
            if any(key not in values for values in aggregates.values()):
                continue
            hint = "lower is better" if direction == "lower" else "higher is better"
            console.print(f"[bold]{label}[/] [dim]({hint})[/]")
            scale = max(values[key] for values in aggregates.values())
            for name, values in aggregates.items():
                bar = _bar(values[key], scale, BACKEND_COLOURS[name])
                console.print(
                    f"  {name:<10} {bar} {fmt.format(values[key])}"
                )
        console.print()

    if agreement:
        agreement_summary = ", ".join(
            f"{key}={value:.3f}" for key, value in agreement.items()
        )
        console.print(f"LLM vs SystemOne mapping agreement: {agreement_summary}")


async def compare_question_part(
    item: dict,
    config: DatasetConfig,
    llm: OpenAILLM | None,
    systemone_client: SystemOne | None,
    *,
    limit: int | None = None,
    llm_concurrency: int = 10,
    mapping_threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
    detail_threshold: float = DEFAULT_DETAIL_THRESHOLD,
    systemone_concurrency: int = DEFAULT_CONCURRENCY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Run the stage-level comparison for one question part and print it.

    Either backend may be None to skip it. Returns the results as a
    JSON-serialisable dict.
    """
    question_part = item["metadata"]["question_part"]
    print(f"\n=== {question_part} ===")
    responses_df = pd.DataFrame(item["input"]["responses"])
    if limit:
        responses_df = responses_df.head(limit)
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
            llm_concurrency,
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
            mapping_threshold,
            detail_threshold,
            systemone_concurrency,
            batch_size,
        )
        runs.append(systemone_run)

    agreement = {}
    if llm_mapping_df is not None and systemone_df is not None:
        agreement = mapping_agreement(llm_mapping_df, systemone_df)

    print_summary(question_part, runs, agreement)
    return {
        "n_responses": len(responses_df),
        "runs": [run.as_dict() for run in runs],
        "mapping_agreement": agreement,
    }


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
        llm = build_llm(args.llm_model, args.llm_api)
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
        part_results = await compare_question_part(
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
        all_results["question_parts"][item["metadata"]["question_part"]] = part_results

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
