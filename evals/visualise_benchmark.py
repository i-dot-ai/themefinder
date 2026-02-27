#!/usr/bin/env python
"""Visualise benchmark results from local files and Langfuse.

A generalised benchmark visualisation system that:
- Auto-detects available metrics from column names
- Adapts to any dataset size (2, 3, 32+ questions)
- Combines local CSV data with Langfuse-derived information
- Generates adaptive HTML reports with interactive charts

Usage:
    uv run python visualise_benchmark.py --benchmark 20260202_134824
    uv run python visualise_benchmark.py --benchmark 20260202_134824 --output report.html
    uv run python visualise_benchmark.py --benchmark 20260202_134824 --no-langfuse
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import dotenv
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Directory where benchmark results are stored
BENCHMARK_RESULTS_DIR = Path(__file__).parent / "benchmark_results"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DatasetMetadata:
    """Metadata about the input dataset."""

    name: str  # e.g., "gambling_XS" → "Gambling Xs"
    response_count: int  # Total responses (sum across all question parts)
    question_count: int  # Number of question_part_N directories
    responses_per_question: dict[int, int] = field(
        default_factory=dict
    )  # qp_num → count


@dataclass
class BenchmarkData:
    """All data for a benchmark run."""

    benchmark_id: str
    config: dict = field(default_factory=dict)
    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    langfuse_traces: list[dict] = field(default_factory=list)
    detected_metrics: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    questions: dict[int, str] = field(
        default_factory=dict
    )  # question_part_N -> question text
    dataset_metadata: DatasetMetadata | None = None

    @property
    def dataset(self) -> str:
        """Get the dataset name from config."""
        return self.config.get("dataset", "unknown")

    @property
    def models(self) -> list[str]:
        """Get unique model names from results."""
        if self.results_df.empty:
            return []
        return sorted(self.results_df["model_tag"].unique().tolist())

    @property
    def eval_types(self) -> list[str]:
        """Get unique eval types from results."""
        if self.results_df.empty:
            return []
        return sorted(self.results_df["eval"].unique().tolist())

    @property
    def total_cost(self) -> float:
        """Get total cost from local CSV data."""
        if self.results_df.empty or "cost_usd" not in self.results_df.columns:
            return 0.0
        return self.results_df["cost_usd"].sum()

    @property
    def total_traces(self) -> int:
        """Get total number of evaluation runs."""
        return len(self.results_df)

    @property
    def avg_latency(self) -> float:
        """Get average latency from local CSV data."""
        if self.results_df.empty or "duration_seconds" not in self.results_df.columns:
            return 0.0
        return self.results_df["duration_seconds"].mean()


# =============================================================================
# Data Loading
# =============================================================================


def load_local_results(benchmark_id: str) -> pd.DataFrame:
    """Load benchmark results from local CSV file."""
    results_dir = BENCHMARK_RESULTS_DIR / benchmark_id
    raw_results_path = results_dir / "raw_results.csv"

    if not raw_results_path.exists():
        console.print(f"[yellow]No local results found at {raw_results_path}[/yellow]")
        return pd.DataFrame()

    df = pd.read_csv(raw_results_path)
    console.print(f"[green]Loaded {len(df)} results from {raw_results_path}[/green]")
    return df


def load_config(benchmark_id: str) -> dict:
    """Load benchmark configuration from JSON file."""
    results_dir = BENCHMARK_RESULTS_DIR / benchmark_id
    config_path = results_dir / "config.json"

    if not config_path.exists():
        console.print(f"[yellow]No config found at {config_path}[/yellow]")
        return {}

    with open(config_path) as f:
        config = json.load(f)

    console.print(
        f"[green]Loaded config: dataset={config.get('dataset', 'unknown')}[/green]"
    )
    return config


def load_questions(dataset_name: str) -> dict[int, str]:
    """Load question text from the dataset directory."""
    # Find dataset directory (in evals/data/)
    datasets_dir = Path(__file__).parent / "data"
    dataset_path = datasets_dir / dataset_name / "inputs"

    if not dataset_path.exists():
        console.print(f"[yellow]Dataset path not found: {dataset_path}[/yellow]")
        return {}

    questions = {}
    for qp_dir in sorted(dataset_path.iterdir()):
        if qp_dir.is_dir() and qp_dir.name.startswith("question_part_"):
            question_file = qp_dir / "question.json"
            if question_file.exists():
                try:
                    with open(question_file) as f:
                        data = json.load(f)
                    qp_num = int(qp_dir.name.replace("question_part_", ""))
                    question_text = data.get("question_text", "")
                    # Append scale statement if present
                    if data.get("scale_statement"):
                        question_text = (
                            f'{question_text}\n\nStatement: "{data["scale_statement"]}"'
                        )
                    questions[qp_num] = question_text
                except (json.JSONDecodeError, ValueError) as e:
                    console.print(
                        f"[yellow]Error loading {question_file}: {e}[/yellow]"
                    )

    if questions:
        console.print(f"[green]Loaded {len(questions)} questions from dataset[/green]")
    return questions


def load_dataset_metadata(dataset_name: str) -> DatasetMetadata | None:
    """Load metadata by inspecting dataset directory structure."""
    datasets_dir = Path(__file__).parent / "data"
    dataset_path = datasets_dir / dataset_name / "inputs"

    if not dataset_path.exists():
        console.print(f"[yellow]Dataset directory not found: {dataset_path}[/yellow]")
        return None

    # Count question_part_N directories
    qp_dirs = [
        d
        for d in dataset_path.iterdir()
        if d.is_dir() and d.name.startswith("question_part_")
    ]
    question_count = len(qp_dirs)

    # Count responses in each question part
    responses_per_question = {}
    for qp_dir in qp_dirs:
        try:
            qp_num = int(qp_dir.name.replace("question_part_", ""))
            responses_file = qp_dir / "responses.jsonl"
            if responses_file.exists():
                with open(responses_file) as f:
                    responses_per_question[qp_num] = sum(1 for _ in f)
        except ValueError:
            continue

    response_count = sum(responses_per_question.values())

    # Format name (underscores → spaces, title case)
    formatted_name = dataset_name.replace("_", " ").title()

    console.print(
        f"[green]Loaded dataset metadata: {response_count} responses, "
        f"{question_count} questions[/green]"
    )

    return DatasetMetadata(
        name=formatted_name,
        response_count=response_count,
        question_count=question_count,
        responses_per_question=responses_per_question,
    )


def fetch_langfuse_traces(benchmark_id: str) -> list[dict]:
    """Fetch all traces for a benchmark from Langfuse with pagination."""
    try:
        from langfuse import Langfuse

        client = Langfuse(timeout=30)  # Increased from default 5s

        results = []
        page = 1
        batch_size = 100

        while True:
            traces = client.api.trace.list(
                tags=[f"benchmark:{benchmark_id}"],
                limit=batch_size,
                page=page,
            )

            if not traces.data:
                break

            for trace in traces.data:
                metadata = trace.metadata or {}
                results.append(
                    {
                        "id": trace.id,
                        "name": trace.name,
                        "session_id": trace.session_id,
                        "model": metadata.get("model", "unknown"),
                        "model_tag": metadata.get("model_tag", "unknown"),
                        "eval_type": metadata.get("eval_type", "unknown"),
                        "run_number": metadata.get("run_number", 0),
                        "cost_usd": trace.total_cost or 0,
                        "latency_s": trace.latency or 0,
                        "tags": trace.tags or [],
                    }
                )

            if len(traces.data) < batch_size:
                break  # Last page

            page += 1

        console.print(f"[green]Fetched {len(results)} traces from Langfuse[/green]")
        return results

    except Exception as e:
        console.print(f"[yellow]Could not fetch Langfuse data: {e}[/yellow]")
        return []


def load_benchmark_data(
    benchmark_id: str, skip_langfuse: bool = False
) -> BenchmarkData:
    """Load all data for a benchmark from local files and Langfuse."""
    # Load local data
    results_df = load_local_results(benchmark_id)
    config = load_config(benchmark_id)

    # Load Langfuse data (optional) - use langfuse_benchmark_id if different from directory
    langfuse_id = config.get("langfuse_benchmark_id", benchmark_id)
    langfuse_traces = [] if skip_langfuse else fetch_langfuse_traces(langfuse_id)

    # Detect available metrics
    detected_metrics = detect_all_metrics(results_df) if not results_df.empty else {}

    # Load questions and metadata from dataset
    dataset_name = config.get("dataset", "")
    questions = load_questions(dataset_name) if dataset_name else {}
    dataset_metadata = load_dataset_metadata(dataset_name) if dataset_name else None

    return BenchmarkData(
        benchmark_id=benchmark_id,
        config=config,
        results_df=results_df,
        langfuse_traces=langfuse_traces,
        detected_metrics=detected_metrics,
        questions=questions,
        dataset_metadata=dataset_metadata,
    )


# =============================================================================
# Metric Auto-Detection
# =============================================================================


def detect_eval_metrics(df: pd.DataFrame, eval_type: str) -> dict[str, list[str]]:
    """Auto-detect available metric columns for an eval type.

    Returns a dict mapping metric group names to lists of column names.
    E.g., {"accuracy": ["question_part_1_accuracy", "question_part_2_accuracy"]}
    """
    if df.empty:
        return {}

    # Filter to rows for this eval type to find relevant columns
    eval_df = df[df["eval"] == eval_type]
    if eval_df.empty:
        return {}

    # Patterns for each eval type's metrics
    # Support both old naming (f1_score) and new naming (f1)
    patterns = {
        "mapping": {
            "f1": r"^question_part_\d+_f1(_score)?$",  # Match f1 or f1_score
            "accuracy_score": r"^question_part_\d+_accuracy_score$",
            "overlap_rate": r"^question_part_\d+_overlap_rate$",
        },
        "generation": {
            "groundedness": r"^question_part_\d+_(groundedness|Precision Average Groundedness)$",
            "coverage": r"^question_part_\d+_(coverage|Recall Average topic Representation)$",
            "specificity": r"^question_part_\d+_specificity$",
            "redundancy": r"^question_part_\d+_redundancy$",
        },
        "condensation": {
            "compression_quality": r"^question_part_\d+_compression_quality$",
            "information_retention": r"^question_part_\d+_information_retention$",
            "redundancy": r"^question_part_\d+_redundancy$",
        },
        "refinement": {
            "information_retention": r"^question_part_\d+_information_retention$",
            "response_references": r"^question_part_\d+_response_references$",
            "distinctiveness": r"^question_part_\d+_distinctiveness$",
            "fluency": r"^question_part_\d+_fluency$",
        },
    }

    eval_patterns = patterns.get(eval_type, {})
    if not eval_patterns:
        return {}

    detected = {}
    for metric_name, pattern in eval_patterns.items():
        matching_cols = [c for c in df.columns if re.match(pattern, c)]
        # Only include if at least one column has non-null data
        if matching_cols:
            non_null = eval_df[matching_cols].notna().any().any()
            if non_null:
                # Sort by question part number
                matching_cols.sort(key=lambda x: int(re.search(r"\d+", x).group()))
                detected[metric_name] = matching_cols

    return detected


def detect_all_metrics(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Detect all available metrics for all eval types in the dataframe."""
    eval_types = df["eval"].unique() if not df.empty else []
    return {eval_type: detect_eval_metrics(df, eval_type) for eval_type in eval_types}


# =============================================================================
# Aggregation
# =============================================================================


def _build_agg_dict(df: pd.DataFrame) -> dict[str, str]:
    """Build aggregation dict based on available columns."""
    agg_dict = {}
    if "cost_usd" in df.columns:
        agg_dict["cost_usd"] = "sum"
    if "duration_seconds" in df.columns:
        agg_dict["duration_seconds"] = "mean"
    if "input_tokens" in df.columns:
        agg_dict["input_tokens"] = "sum"
    if "output_tokens" in df.columns:
        agg_dict["output_tokens"] = "sum"
    if "total_tokens" in df.columns:
        agg_dict["total_tokens"] = "sum"
    return agg_dict


def aggregate_by_model(data: BenchmarkData) -> pd.DataFrame:
    """Aggregate metrics by model."""
    if data.results_df.empty:
        return pd.DataFrame()

    df = data.results_df
    agg_dict = _build_agg_dict(df)

    if agg_dict:
        agg = df.groupby("model_tag").agg(agg_dict).reset_index()
    else:
        agg = df.groupby("model_tag").size().reset_index(name="eval_count")

    agg["eval_count"] = df.groupby("model_tag").size().values
    return agg.sort_values("model_tag")


def aggregate_by_eval(data: BenchmarkData) -> pd.DataFrame:
    """Aggregate metrics by eval type."""
    if data.results_df.empty:
        return pd.DataFrame()

    df = data.results_df
    agg_dict = _build_agg_dict(df)

    if agg_dict:
        agg = df.groupby("eval").agg(agg_dict).reset_index()
    else:
        agg = df.groupby("eval").size().reset_index(name="model_count")

    agg["model_count"] = df.groupby("eval")["model_tag"].nunique().values
    return agg.sort_values("eval")


def aggregate_performance_metrics(data: BenchmarkData, eval_type: str) -> pd.DataFrame:
    """Aggregate performance metrics for a specific eval type by model.

    Returns a dataframe with columns: model_tag, metric_name, mean, std, n
    """
    if data.results_df.empty or eval_type not in data.detected_metrics:
        return pd.DataFrame()

    df = data.results_df[data.results_df["eval"] == eval_type]
    if df.empty:
        return pd.DataFrame()

    metrics = data.detected_metrics[eval_type]
    rows = []

    for model in df["model_tag"].unique():
        model_df = df[df["model_tag"] == model]

        for metric_group, cols in metrics.items():
            for col in cols:
                values = model_df[col].dropna()
                if len(values) > 0:
                    # Extract question part number for display
                    qp_match = re.search(r"question_part_(\d+)", col)
                    qp_num = qp_match.group(1) if qp_match else "?"

                    rows.append(
                        {
                            "model_tag": model,
                            "metric_group": metric_group,
                            "question_part": int(qp_num) if qp_num != "?" else 0,
                            "column": col,
                            "mean": values.mean(),
                            "std": values.std() if len(values) > 1 else 0,
                            "n": len(values),
                        }
                    )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def calculate_consistency_metrics(data: BenchmarkData) -> pd.DataFrame:
    """Calculate score consistency metrics (std, coefficient of variation) per model+eval.

    Groups existing benchmark data by model_tag and eval type, computes
    standard deviation and coefficient of variation across runs.
    Flags models with CV > 0.15 as inconsistent.

    Args:
        data: BenchmarkData with results_df populated.

    Returns:
        DataFrame with columns: model_tag, eval, metric, mean, std, cv, flagged.
    """
    if data.results_df.empty or not data.detected_metrics:
        return pd.DataFrame()

    rows = []

    for eval_type, metrics_dict in data.detected_metrics.items():
        eval_df = data.results_df[data.results_df["eval"] == eval_type]
        if eval_df.empty:
            continue

        for metric_group, cols in metrics_dict.items():
            for model in eval_df["model_tag"].unique():
                model_df = eval_df[eval_df["model_tag"] == model]

                # Collect all values across question parts for this metric group
                all_values = []
                for col in cols:
                    if col in model_df.columns:
                        values = model_df[col].dropna().tolist()
                        all_values.extend(values)

                if len(all_values) < 2:
                    continue

                mean_val = float(np.mean(all_values))
                std_val = float(np.std(all_values, ddof=1))
                cv = std_val / mean_val if mean_val > 0 else 0.0

                rows.append(
                    {
                        "model_tag": model,
                        "eval": eval_type,
                        "metric": metric_group,
                        "mean": round(mean_val, 3),
                        "std": round(std_val, 3),
                        "cv": round(cv, 3),
                        "n": len(all_values),
                        "flagged": cv > 0.15,
                    }
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# Composite Performance Index (TQI)
# =============================================================================

# Stage weights for the ThemeFinder Quality Index
STAGE_WEIGHTS = {
    "generation": 0.40,
    "condensation": 0.10,
    "refinement": 0.10,
    "mapping": 0.40,
}

GENERATION_WEIGHTS = {"groundedness": 0.45, "coverage": 0.45, "specificity": 0.10}
GENERATION_SCALE = {"groundedness": 5.0, "coverage": 5.0, "specificity": 1.0}

CONDENSATION_WEIGHTS = {
    "compression_quality": 0.40,
    "information_retention": 0.40,
    "redundancy_inv": 0.20,  # 1 - redundancy (lower is better)
}
CONDENSATION_SCALE = {
    "compression_quality": 5.0,
    "information_retention": 5.0,
    "redundancy_inv": 1.0,
}

REFINEMENT_WEIGHTS = {
    "information_retention": 0.25,
    "response_references": 0.25,
    "distinctiveness": 0.25,
    "fluency": 0.25,
}
REFINEMENT_SCALE = {k: 5.0 for k in REFINEMENT_WEIGHTS}


def _weighted_stage_score(
    stage_df: pd.DataFrame,
    metrics_dict: dict[str, list[str]],
    weights: dict[str, float],
    scales: dict[str, float],
    invert: set[str] | None = None,
) -> float | None:
    """Calculate a weighted stage score from detected metric columns.

    Args:
        stage_df: DataFrame filtered to one model + one eval type.
        metrics_dict: Detected metric columns {metric_name: [col_names]}.
        weights: Weight per metric (keys must match metrics_dict or use suffixed names).
        scales: Max raw value per metric (for normalisation to 0-1).
        invert: Set of metric names to invert (1 - normalised value).

    Returns:
        Weighted average in [0, 1], or None if no data.
    """
    invert = invert or set()
    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, weight in weights.items():
        # Handle suffixed weight keys (e.g. "redundancy_inv" → "redundancy")
        col_key = metric_name.removesuffix("_inv")
        cols = metrics_dict.get(col_key, [])
        if not cols:
            continue
        values = stage_df[cols].values.flatten()
        values = values[~np.isnan(values.astype(float))]
        if len(values) == 0:
            continue
        scale = scales.get(metric_name, 1.0)
        normalised = float(np.mean(values)) / scale
        if metric_name in invert:
            normalised = 1.0 - normalised
        weighted_sum += weight * normalised
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else None


def calculate_composite_scores(data: BenchmarkData) -> pd.DataFrame:
    """Calculate composite ThemeFinder Quality Index (TQI) per model.

    Combines generation, condensation, refinement, and mapping
    stage scores into a single 0-1 composite using statistically-derived weights.

    Returns DataFrame with columns:
        model_tag, generation_score, condensation_score, refinement_score,
        mapping_score, composite
    """
    if data.results_df.empty or not data.detected_metrics:
        return pd.DataFrame()

    df = data.results_df
    models = sorted(df["model_tag"].unique())
    rows = []

    for model in models:
        model_df = df[df["model_tag"] == model]
        scores: dict[str, float | None] = {}

        # --- Generation stage score ---
        gen_metrics = data.detected_metrics.get("generation", {})
        gen_df = model_df[model_df["eval"] == "generation"]
        scores["generation"] = (
            _weighted_stage_score(
                gen_df,
                gen_metrics,
                GENERATION_WEIGHTS,
                GENERATION_SCALE,
            )
            if not gen_df.empty and gen_metrics
            else None
        )

        # --- Condensation stage score ---
        cond_metrics = data.detected_metrics.get("condensation", {})
        cond_df = model_df[model_df["eval"] == "condensation"]
        scores["condensation"] = (
            _weighted_stage_score(
                cond_df,
                cond_metrics,
                CONDENSATION_WEIGHTS,
                CONDENSATION_SCALE,
                invert={"redundancy_inv"},
            )
            if not cond_df.empty and cond_metrics
            else None
        )

        # --- Refinement stage score ---
        ref_metrics = data.detected_metrics.get("refinement", {})
        ref_df = model_df[model_df["eval"] == "refinement"]
        scores["refinement"] = (
            _weighted_stage_score(
                ref_df,
                ref_metrics,
                REFINEMENT_WEIGHTS,
                REFINEMENT_SCALE,
            )
            if not ref_df.empty and ref_metrics
            else None
        )

        # --- Mapping stage score ---
        map_metrics = data.detected_metrics.get("mapping", {})
        map_df = model_df[model_df["eval"] == "mapping"]
        f1_cols = map_metrics.get("f1", [])
        if map_df.empty or not f1_cols:
            scores["mapping"] = None
        else:
            values = map_df[f1_cols].values.flatten()
            values = values[~np.isnan(values.astype(float))]
            scores["mapping"] = float(np.mean(values)) if len(values) > 0 else None

        # --- Composite ---
        if all(scores[s] is not None for s in STAGE_WEIGHTS):
            composite = sum(
                STAGE_WEIGHTS[s] * scores[s]
                for s in STAGE_WEIGHTS  # type: ignore[operator]
            )
        else:
            composite = None

        rows.append(
            {
                "model_tag": model,
                "generation_score": scores["generation"],
                "condensation_score": scores["condensation"],
                "refinement_score": scores["refinement"],
                "mapping_score": scores["mapping"],
                "composite": composite,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# Terminal Output
# =============================================================================


def print_summary(data: BenchmarkData) -> None:
    """Print a summary of the benchmark data."""
    console.print(f"\n[bold cyan]Benchmark Summary: {data.benchmark_id}[/bold cyan]")
    console.print(f"Dataset: [bold]{data.dataset}[/bold]")
    console.print(f"Models: {', '.join(data.models)}")
    console.print(f"Eval types: {', '.join(data.eval_types)}")
    console.print(f"Total runs: {data.total_traces}")
    console.print(f"Total cost: ${data.total_cost:.4f}")
    console.print(f"Avg latency: {data.avg_latency:.1f}s")
    console.print()


def print_model_table(data: BenchmarkData) -> None:
    """Print cost and latency by model."""
    agg = aggregate_by_model(data)
    if agg.empty:
        return

    table = Table(title="Metrics by Model")
    table.add_column("Model", style="cyan")
    table.add_column("Evals", justify="right")

    # Conditionally add columns
    has_cost = "cost_usd" in agg.columns
    has_latency = "duration_seconds" in agg.columns
    has_tokens = "total_tokens" in agg.columns

    if has_cost:
        table.add_column("Total Cost", justify="right", style="green")
    if has_latency:
        table.add_column("Avg Latency", justify="right")
    if has_tokens:
        table.add_column("Total Tokens", justify="right")

    for _, row in agg.iterrows():
        row_values = [row["model_tag"], str(int(row["eval_count"]))]
        if has_cost:
            row_values.append(f"${row['cost_usd']:.4f}")
        if has_latency:
            row_values.append(f"{row['duration_seconds']:.1f}s")
        if has_tokens:
            row_values.append(f"{int(row['total_tokens']):,}")
        table.add_row(*row_values)

    console.print(table)
    console.print()


def print_eval_table(data: BenchmarkData) -> None:
    """Print cost and latency by eval type."""
    agg = aggregate_by_eval(data)
    if agg.empty:
        return

    table = Table(title="Metrics by Eval Type")
    table.add_column("Eval Type", style="cyan")
    table.add_column("Models", justify="right")

    # Conditionally add columns
    has_cost = "cost_usd" in agg.columns
    has_latency = "duration_seconds" in agg.columns
    has_tokens = "total_tokens" in agg.columns

    if has_cost:
        table.add_column("Total Cost", justify="right", style="green")
    if has_latency:
        table.add_column("Avg Latency", justify="right")
    if has_tokens:
        table.add_column("Total Tokens", justify="right")

    for _, row in agg.iterrows():
        row_values = [row["eval"], str(int(row["model_count"]))]
        if has_cost:
            row_values.append(f"${row['cost_usd']:.4f}")
        if has_latency:
            row_values.append(f"{row['duration_seconds']:.1f}s")
        if has_tokens:
            row_values.append(f"{int(row['total_tokens']):,}")
        table.add_row(*row_values)

    console.print(table)
    console.print()


def print_performance_table(
    data: BenchmarkData, eval_type: str, metric_group: str
) -> None:
    """Print a performance metrics table for a specific eval type and metric."""
    perf = aggregate_performance_metrics(data, eval_type)
    if perf.empty:
        return

    # Filter to the metric group
    perf = perf[perf["metric_group"] == metric_group]
    if perf.empty:
        return

    # Pivot: models as rows, question parts as columns
    question_parts = sorted(perf["question_part"].unique())
    models = sorted(perf["model_tag"].unique())

    title = f"{eval_type.title()} {metric_group.replace('_', ' ').title()}"
    table = Table(title=title)
    table.add_column("Model", style="cyan")

    for qp in question_parts:
        table.add_column(f"Q{qp}", justify="right")

    for model in models:
        model_perf = perf[perf["model_tag"] == model]
        values = []
        for qp in question_parts:
            qp_row = model_perf[model_perf["question_part"] == qp]
            if not qp_row.empty:
                mean = qp_row["mean"].values[0]
                values.append(f"{mean:.3f}")
            else:
                values.append("-")
        table.add_row(model, *values)

    console.print(table)
    console.print()


def print_all_performance_tables(data: BenchmarkData) -> None:
    """Print all available performance tables."""
    console.print("[bold cyan]Performance Metrics[/bold cyan]\n")

    for eval_type, metrics in data.detected_metrics.items():
        for metric_group in metrics:
            print_performance_table(data, eval_type, metric_group)


def print_composite_table(data: BenchmarkData) -> None:
    """Print composite TQI scores by model."""
    composite_df = calculate_composite_scores(data)
    if composite_df.empty:
        return

    table = Table(title="ThemeFinder Quality Index (TQI)")
    table.add_column("Model", style="cyan")
    table.add_column("Gen", justify="right")
    table.add_column("Cond", justify="right")
    table.add_column("Ref", justify="right")
    table.add_column("Map", justify="right")
    table.add_column("TQI", justify="right", style="bold")

    # Sort by composite descending
    composite_df = composite_df.sort_values(
        "composite", ascending=False, na_position="last"
    )

    for _, row in composite_df.iterrows():
        gen = (
            f"{row['generation_score']:.3f}"
            if pd.notna(row["generation_score"])
            else "-"
        )
        cond = (
            f"{row['condensation_score']:.3f}"
            if pd.notna(row["condensation_score"])
            else "-"
        )
        ref = (
            f"{row['refinement_score']:.3f}"
            if pd.notna(row["refinement_score"])
            else "-"
        )
        mapp = f"{row['mapping_score']:.3f}" if pd.notna(row["mapping_score"]) else "-"
        tqi = f"{row['composite']:.3f}" if pd.notna(row["composite"]) else "-"
        table.add_row(row["model_tag"], gen, cond, ref, mapp, tqi)

    console.print(table)
    console.print()


# =============================================================================
# HTML Report Generation
# =============================================================================


def _get_langfuse_url(benchmark_id: str) -> str | None:
    """Build Langfuse URL for this benchmark's sessions.

    Uses LANGFUSE_BASE_URL and LANGFUSE_PROJECT_ID from environment.
    Returns None if not configured.
    """
    base_url = os.getenv("LANGFUSE_BASE_URL")
    project_id = os.getenv("LANGFUSE_PROJECT_ID")

    if not base_url or not project_id:
        return None

    # Sessions have IDs like: benchmark_20260202_134824_claude-haiku-4.5_mapping_run1
    # Use search parameter to filter sessions containing the benchmark ID
    search_term = quote(f"benchmark_{benchmark_id}")
    return f"{base_url}/project/{project_id}/sessions?search={search_term}"


def _langfuse_button_html(benchmark_id: str) -> str:
    """Generate Langfuse button HTML if configured."""
    url = _get_langfuse_url(benchmark_id)
    if not url:
        return ""
    return (
        f'<a href="{url}" target="_blank" class="langfuse-btn">View in Langfuse ↗</a>'
    )


def generate_html_report(data: BenchmarkData, output_path: str) -> None:
    """Generate an HTML report with interactive charts."""
    # Aggregate data
    by_model = aggregate_by_model(data)
    by_eval = aggregate_by_eval(data)

    # Prepare chart data with safe column access
    models = by_model["model_tag"].tolist() if not by_model.empty else []
    model_costs = (
        by_model["cost_usd"].tolist()
        if not by_model.empty and "cost_usd" in by_model.columns
        else []
    )
    model_latencies = (
        by_model["duration_seconds"].tolist()
        if not by_model.empty and "duration_seconds" in by_model.columns
        else []
    )

    eval_types = by_eval["eval"].tolist() if not by_eval.empty else []
    eval_costs = (
        by_eval["cost_usd"].tolist()
        if not by_eval.empty and "cost_usd" in by_eval.columns
        else []
    )

    # Build HTML with new compact layout
    html_parts = [
        _html_header(data),
        _html_summary_cards(data),
        _html_model_summary_table(data),
        _html_cost_latency_charts(
            models, model_costs, model_latencies, eval_types, eval_costs
        ),
        _html_performance_section(data),
        _html_composite_section(data),
        _html_consistency_section(data),
        _html_footer(),
    ]

    # Write file
    html_content = "\n".join(html_parts)
    with open(output_path, "w") as f:
        f.write(html_content)

    console.print(f"\n[green]HTML report saved to: {output_path}[/green]")


def _html_header(data: BenchmarkData) -> str:
    """Generate HTML header with styles."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>ThemeFinder AutoEval: {data.dataset} - {data.benchmark_id}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0; padding: 16px;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 13px;
        }}
        code, .mono {{ font-family: 'Roboto Mono', monospace; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 12px 0; border-bottom: 1px solid #334155; margin-bottom: 16px;
        }}
        .header h1 {{ margin: 0; font-size: 1.3em; color: #f1f5f9; font-weight: 600; }}
        .header h1 .brand {{ color: #94a3b8; font-weight: 400; }}
        .header-meta {{ color: #64748b; font-size: 0.8em; font-family: 'Roboto Mono', monospace; }}
        .header-meta span {{ margin-left: 16px; }}

        /* Compact stat pills */
        .stats-row {{
            display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
        }}
        .stat-pill {{
            background: #1e293b; padding: 8px 16px; border-radius: 20px;
            display: flex; align-items: center; gap: 8px;
        }}
        .stat-pill .value {{ color: #38bdf8; font-weight: 600; font-family: 'Roboto Mono', monospace; }}
        .stat-pill .label {{ color: #94a3b8; }}

        /* Grid layout */
        .grid {{ display: grid; gap: 16px; }}
        .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
        .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
        @media (max-width: 1200px) {{ .grid-3 {{ grid-template-columns: repeat(2, 1fr); }} }}
        @media (max-width: 800px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}

        /* Cards */
        .card {{
            background: #1e293b; border-radius: 8px; padding: 12px;
            border: 1px solid #334155;
        }}
        .card-title {{
            font-size: 0.75em; color: #94a3b8; text-transform: uppercase;
            letter-spacing: 0.5px; margin-bottom: 8px; font-weight: 500;
        }}

        /* Tables */
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 500; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.3px; }}
        td {{ color: #e2e8f0; }}
        tr:hover td {{ background: #334155; }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: 'Roboto Mono', monospace; }}
        .good {{ color: #4ade80; }}
        .medium {{ color: #fbbf24; }}
        .poor {{ color: #f87171; }}

        /* Tabs */
        .tabs {{ display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }}
        .tab {{
            padding: 6px 14px; background: #334155; border: none; border-radius: 6px;
            color: #94a3b8; cursor: pointer; font-size: 0.85em; transition: all 0.15s;
            font-family: 'Inter', sans-serif; font-weight: 500;
        }}
        .tab:hover {{ background: #475569; color: #e2e8f0; }}
        .tab.active {{ background: #3b82f6; color: white; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Charts */
        .chart {{ height: 220px; }}
        .chart-row {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-top: 16px; }}
        @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}

        /* Section headers */
        .section-header {{
            font-size: 0.9em; color: #f1f5f9; margin: 20px 0 12px 0;
            padding-bottom: 8px; border-bottom: 1px solid #334155;
            font-weight: 500;
        }}

        /* Expandable question */
        .question-expander {{
            margin-bottom: 12px;
        }}
        .question-toggle {{
            background: #334155; border: none; padding: 8px 12px; border-radius: 6px;
            color: #94a3b8; cursor: pointer; font-size: 0.8em; display: flex;
            align-items: center; gap: 6px; font-family: 'Inter', sans-serif;
        }}
        .question-toggle:hover {{ background: #475569; color: #e2e8f0; }}
        .question-toggle .arrow {{ transition: transform 0.2s; }}
        .question-toggle.open .arrow {{ transform: rotate(90deg); }}
        .question-text {{
            display: none; background: #0f172a; border: 1px solid #334155;
            border-radius: 6px; padding: 12px; margin-top: 8px;
            font-size: 0.85em; color: #cbd5e1; line-height: 1.5;
        }}
        .question-text.open {{ display: block; }}

        /* Header actions */
        .header-actions {{
            display: flex; align-items: center; gap: 16px;
        }}
        .langfuse-btn {{
            background: #7c3aed; color: white; padding: 6px 12px;
            border-radius: 6px; text-decoration: none; font-size: 0.8em;
            font-weight: 500; transition: background 0.15s;
        }}
        .langfuse-btn:hover {{ background: #6d28d9; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><span class="brand">ThemeFinder AutoEval:</span> {data.dataset}</h1>
            <div class="header-actions">
                {_langfuse_button_html(data.benchmark_id)}
                <div class="header-meta">
                    <span>{data.benchmark_id}</span>
                    <span>{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
                </div>
            </div>
        </div>
"""


def _html_summary_cards(data: BenchmarkData) -> str:
    """Generate compact stat pills."""
    lf_cost = (
        sum(t["cost_usd"] for t in data.langfuse_traces)
        if data.langfuse_traces
        else data.total_cost
    )
    lf_latency = (
        sum(t["latency_s"] for t in data.langfuse_traces) / len(data.langfuse_traces)
        if data.langfuse_traces
        else data.avg_latency
    )

    # Dataset metadata pills (if available)
    dataset_pills = ""
    if data.dataset_metadata:
        dm = data.dataset_metadata
        dataset_pills = f"""
            <div class="stat-pill"><span class="value">{dm.response_count:,}</span><span class="label">Responses</span></div>
            <div class="stat-pill"><span class="value">{dm.question_count}</span><span class="label">Questions</span></div>
"""

    return f"""
        <div class="stats-row">
            {dataset_pills}
            <div class="stat-pill"><span class="value">{len(data.models)}</span><span class="label">Models</span></div>
            <div class="stat-pill"><span class="value">{len(data.eval_types)}</span><span class="label">Evals</span></div>
            <div class="stat-pill"><span class="value">{data.total_traces}</span><span class="label">Runs</span></div>
            <div class="stat-pill"><span class="value">${lf_cost:.2f}</span><span class="label">Cost</span></div>
            <div class="stat-pill"><span class="value">{lf_latency:.0f}s</span><span class="label">Avg Latency</span></div>
        </div>
"""


def _html_model_summary_table(data: BenchmarkData) -> str:
    """Generate the main model summary table with performance by eval, cost, latency."""
    if data.results_df.empty:
        return ""

    df = data.results_df
    models = sorted(df["model_tag"].unique())
    eval_types = sorted(df["eval"].unique())

    # Get primary metric for each eval type
    primary_metrics = {
        "generation": "groundedness",
        "condensation": "compression_quality",
        "refinement": "information_retention",
        "mapping": "f1",
    }

    # Calculate composite scores
    composite_df = calculate_composite_scores(data)
    has_composite = not composite_df.empty and composite_df["composite"].notna().any()

    # Build header
    header_cells = ["<th>Model</th>"]
    if has_composite:
        header_cells.append("<th class='num'>TQI</th>")
    for et in eval_types:
        if et in primary_metrics:
            header_cells.append(f"<th class='num'>{et.title()}</th>")
    header_cells.append("<th class='num'>Cost</th>")
    header_cells.append("<th class='num'>Latency</th>")
    header_cells.append("<th class='num'>Tokens</th>")

    # Build rows
    rows = []
    for model in models:
        model_df = df[df["model_tag"] == model]
        cells = [f"<td><strong>{model}</strong></td>"]

        # TQI composite score
        if has_composite:
            model_composite = composite_df[composite_df["model_tag"] == model]
            if not model_composite.empty and pd.notna(
                model_composite["composite"].values[0]
            ):
                tqi = model_composite["composite"].values[0]
                css_class = "good" if tqi >= 0.8 else "medium" if tqi >= 0.6 else "poor"
                cells.append(
                    f"<td class='num {css_class}'><strong>{tqi:.3f}</strong></td>"
                )
            else:
                cells.append("<td class='num'>-</td>")

        # Performance metrics for each eval
        for et in eval_types:
            if et not in primary_metrics:
                continue
            metric_name = primary_metrics[et]
            et_df = model_df[model_df["eval"] == et]

            if et_df.empty:
                cells.append("<td class='num'>-</td>")
                continue

            # Find metric columns
            metrics_found = data.detected_metrics.get(et, {}).get(metric_name, [])
            if metrics_found:
                # Average across all question parts
                values = []
                for col in metrics_found:
                    if col in et_df.columns:
                        val = et_df[col].mean()
                        if pd.notna(val):
                            values.append(val)
                if values:
                    avg = sum(values) / len(values)
                    css_class = (
                        "good" if avg >= 0.8 else "medium" if avg >= 0.6 else "poor"
                    )
                    cells.append(f"<td class='num {css_class}'>{avg:.2f}</td>")
                else:
                    cells.append("<td class='num'>-</td>")
            else:
                cells.append("<td class='num'>-</td>")

        # Cost
        if "cost_usd" in model_df.columns:
            cost = model_df["cost_usd"].sum()
            cells.append(f"<td class='num'>${cost:.3f}</td>")
        else:
            cells.append("<td class='num'>-</td>")

        # Latency
        if "duration_seconds" in model_df.columns:
            latency = model_df["duration_seconds"].mean()
            cells.append(f"<td class='num'>{latency:.0f}s</td>")
        else:
            cells.append("<td class='num'>-</td>")

        # Tokens
        if "total_tokens" in model_df.columns:
            tokens = model_df["total_tokens"].sum()
            cells.append(f"<td class='num'>{tokens:,}</td>")
        else:
            cells.append("<td class='num'>-</td>")

        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
        <div class="card">
            <div class="card-title">Model Performance Summary</div>
            <table>
                <thead><tr>{"".join(header_cells)}</tr></thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        </div>
"""


def _html_cost_latency_charts(
    models: list[str],
    model_costs: list[float],
    model_latencies: list[float],
    eval_types: list[str],
    eval_costs: list[float],
) -> str:
    """Generate compact cost and latency charts in a grid."""
    has_cost = model_costs and any(c > 0 for c in model_costs)
    has_latency = model_latencies and any(latency > 0 for latency in model_latencies)

    if not has_cost and not has_latency:
        return ""

    sections = [
        '<div class="section-header">Cost & Latency</div><div class="chart-row">'
    ]

    if has_cost:
        sections.append("""
        <div class="card">
            <div class="card-title">Cost by Model</div>
            <div class="chart" id="cost-model"></div>
        </div>
""")

    if has_latency:
        sections.append("""
        <div class="card">
            <div class="card-title">Latency by Model</div>
            <div class="chart" id="latency-model"></div>
        </div>
""")

    sections.append("</div>")

    # JavaScript with proper layout objects
    sections.append("<script>")
    sections.append("""
        var chartLayout = {
            margin: {t: 20, b: 60, l: 50, r: 20},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {color: '#94a3b8', size: 11, family: 'Inter'},
            xaxis: {gridcolor: '#334155', tickfont: {size: 10}},
            yaxis: {gridcolor: '#334155', tickfont: {size: 10}}
        };
        var chartConfig = {responsive: true, displayModeBar: false};
""")

    if has_cost:
        sections.append(f"""
        Plotly.newPlot('cost-model', [{{
            x: {json.dumps(models)},
            y: {json.dumps(model_costs)},
            type: 'bar',
            marker: {{color: '#f472b6'}},
            text: {json.dumps([f"${c:.3f}" for c in model_costs])},
            textposition: 'outside',
            textfont: {{size: 10, family: 'Roboto Mono'}}
        }}], Object.assign({{}}, chartLayout, {{yaxis: Object.assign({{}}, chartLayout.yaxis, {{title: 'USD'}})}}), chartConfig);
""")

    if has_latency:
        sections.append(f"""
        Plotly.newPlot('latency-model', [{{
            x: {json.dumps(models)},
            y: {json.dumps(model_latencies)},
            type: 'bar',
            marker: {{color: '#38bdf8'}},
            text: {json.dumps([f"{latency:.0f}s" for latency in model_latencies])},
            textposition: 'outside',
            textfont: {{size: 10, family: 'Roboto Mono'}}
        }}], Object.assign({{}}, chartLayout, {{yaxis: Object.assign({{}}, chartLayout.yaxis, {{title: 'Seconds'}})}}), chartConfig);
""")

    sections.append("</script>")

    return "\n".join(sections)


def _html_performance_section(data: BenchmarkData) -> str:
    """Generate tabbed performance section with question-level detail."""
    if not data.detected_metrics:
        return ""

    # Collect all question parts across all evals
    all_questions = set()
    for eval_type, metrics in data.detected_metrics.items():
        for metric_group, cols in metrics.items():
            for col in cols:
                match = re.search(r"question_part_(\d+)", col)
                if match:
                    all_questions.add(int(match.group(1)))

    question_parts = sorted(all_questions)
    if not question_parts:
        return ""

    # Build tabs
    tabs_html = '<div class="tabs">'
    tabs_html += (
        '<button class="tab active" onclick="showTab(\'overview\')">Overview</button>'
    )
    for qp in question_parts:
        tabs_html += f'<button class="tab" onclick="showTab(\'q{qp}\')">Q{qp}</button>'
    tabs_html += "</div>"

    # Overview tab content - shows all questions in one chart per metric
    overview_content = _html_overview_tab(data, question_parts)

    # Individual question tab content
    question_tabs = ""
    for qp in question_parts:
        question_tabs += _html_question_tab(data, qp)

    # Tab switching script
    tab_script = """
    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');
            // Trigger Plotly resize for any charts in the tab
            window.dispatchEvent(new Event('resize'));
        }

        function toggleQuestion(qp) {
            var textEl = document.getElementById('question-text-' + qp);
            var toggleEl = event.currentTarget;
            textEl.classList.toggle('open');
            toggleEl.classList.toggle('open');
        }
    </script>
    """

    return f"""
        <div class="section-header">Performance by Question</div>
        {tabs_html}
        {overview_content}
        {question_tabs}
        {tab_script}
"""


def _html_overview_tab(data: BenchmarkData, question_parts: list[int]) -> str:
    """Generate overview tab showing all questions."""
    colours = [
        "#f472b6",
        "#38bdf8",
        "#4ade80",
        "#fbbf24",
        "#a78bfa",
        "#22d3d8",
        "#fb7185",
        "#a3e635",
    ]
    models = data.models

    charts = []
    chart_data = []

    for eval_type, metrics in data.detected_metrics.items():
        for metric_group, cols in metrics.items():
            chart_id = f"overview-{eval_type}-{metric_group}"
            title = f"{eval_type.title()} {metric_group.replace('_', ' ').title()}"

            # Build traces for each model
            traces = []
            for i, model in enumerate(models):
                y_values = []
                for qp in question_parts:
                    # Try alternative naming
                    matching_col = None
                    for c in cols:
                        if f"question_part_{qp}" in c:
                            matching_col = c
                            break
                    if matching_col:
                        model_df = data.results_df[
                            (data.results_df["model_tag"] == model)
                            & (data.results_df["eval"] == eval_type)
                        ]
                        if not model_df.empty and matching_col in model_df.columns:
                            val = model_df[matching_col].mean()
                            y_values.append(val if pd.notna(val) else 0)
                        else:
                            y_values.append(0)
                    else:
                        y_values.append(0)

                traces.append(
                    {
                        "name": model,
                        "x": [f"Q{qp}" for qp in question_parts],
                        "y": y_values,
                        "type": "bar",
                        "marker": {"color": colours[i % len(colours)]},
                    }
                )

            charts.append(f"""
                <div class="card">
                    <div class="card-title">{title}</div>
                    <div class="chart" id="{chart_id}"></div>
                </div>
""")
            chart_data.append((chart_id, traces))

    # Build JavaScript
    script_parts = [
        """
        var overviewLayout = {
            barmode: 'group',
            margin: {t: 10, b: 50, l: 40, r: 10},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {color: '#94a3b8', size: 10, family: 'Inter'},
            xaxis: {gridcolor: '#334155'},
            yaxis: {gridcolor: '#334155', range: [0, 1.05]},
            legend: {orientation: 'h', y: -0.25, font: {size: 9}},
            showlegend: true
        };
        var overviewConfig = {responsive: true, displayModeBar: false};
"""
    ]

    for chart_id, traces in chart_data:
        script_parts.append(f"""
        Plotly.newPlot('{chart_id}', {json.dumps(traces)}, overviewLayout, overviewConfig);
""")

    return f"""
        <div id="tab-overview" class="tab-content active">
            <div class="grid grid-2">{"".join(charts)}</div>
        </div>
        <script>{"".join(script_parts)}</script>
"""


def _html_question_tab(data: BenchmarkData, qp: int) -> str:
    """Generate a single question tab with detailed metrics."""
    models = data.models

    # Get question text if available
    question_text = data.questions.get(qp, "")
    question_html = ""
    if question_text:
        # Escape HTML and convert newlines
        escaped_text = (
            question_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        question_html = f"""
            <div class="question-expander">
                <button class="question-toggle" onclick="toggleQuestion({qp})">
                    <span class="arrow">▶</span> View Question
                </button>
                <div class="question-text" id="question-text-{qp}">{escaped_text}</div>
            </div>
"""

    # Build table for this question
    header_cells = ["<th>Model</th>"]
    metric_cols = []

    for eval_type, metrics in data.detected_metrics.items():
        for metric_group, cols in metrics.items():
            # Find column for this question part
            for col in cols:
                if f"question_part_{qp}" in col:
                    header_cells.append(
                        f"<th class='num'>{eval_type[:4].title()} {metric_group[:3].title()}</th>"
                    )
                    metric_cols.append((eval_type, col))
                    break

    rows = []
    for model in models:
        cells = [f"<td><strong>{model}</strong></td>"]
        for eval_type, col in metric_cols:
            model_df = data.results_df[
                (data.results_df["model_tag"] == model)
                & (data.results_df["eval"] == eval_type)
            ]
            if not model_df.empty and col in model_df.columns:
                val = model_df[col].mean()
                if pd.notna(val):
                    css_class = (
                        "good" if val >= 0.8 else "medium" if val >= 0.6 else "poor"
                    )
                    cells.append(f"<td class='num {css_class}'>{val:.3f}</td>")
                else:
                    cells.append("<td class='num'>-</td>")
            else:
                cells.append("<td class='num'>-</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
        <div id="tab-q{qp}" class="tab-content">
            {question_html}
            <div class="card">
                <div class="card-title">Question {qp} - All Metrics</div>
                <table>
                    <thead><tr>{"".join(header_cells)}</tr></thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
        </div>
"""


def _html_consistency_section(data: BenchmarkData) -> str:
    """Generate consistency metrics section for the HTML report."""
    consistency_df = calculate_consistency_metrics(data)
    if consistency_df.empty:
        return ""

    # Build table rows
    rows = []
    for _, row in consistency_df.iterrows():
        flag_icon = "&#9888;" if row["flagged"] else ""
        cv_class = "poor" if row["flagged"] else "good"
        rows.append(
            f"<tr>"
            f"<td><strong>{row['model_tag']}</strong></td>"
            f"<td>{row['eval']}</td>"
            f"<td>{row['metric']}</td>"
            f"<td class='num'>{row['mean']:.3f}</td>"
            f"<td class='num'>{row['std']:.3f}</td>"
            f"<td class='num {cv_class}'>{row['cv']:.3f} {flag_icon}</td>"
            f"<td class='num'>{row['n']}</td>"
            f"</tr>"
        )

    flagged_count = consistency_df["flagged"].sum()
    flag_summary = ""
    if flagged_count > 0:
        flag_summary = f"<p style='color: #fbbf24; font-size: 0.85em; margin-top: 8px;'>&#9888; {int(flagged_count)} model/metric combinations with CV &gt; 0.15 (inconsistent scoring)</p>"

    return f"""
        <div class="section-header">Consistency Analysis</div>
        <div class="card">
            <div class="card-title">Score Consistency Across Runs (CV &gt; 0.15 flagged)</div>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Eval</th>
                        <th>Metric</th>
                        <th class='num'>Mean</th>
                        <th class='num'>Std</th>
                        <th class='num'>CV</th>
                        <th class='num'>N</th>
                    </tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
            {flag_summary}
        </div>
"""


def _html_composite_section(data: BenchmarkData) -> str:
    """Generate composite TQI section with stacked bar chart and heatmap."""
    composite_df = calculate_composite_scores(data)
    if composite_df.empty or composite_df["composite"].isna().all():
        return ""

    # Filter to models with complete composite scores, sort by TQI descending
    valid_df = composite_df.dropna(subset=["composite"]).sort_values(
        "composite",
        ascending=True,  # ascending for horizontal bar (bottom = best)
    )
    if valid_df.empty:
        return ""

    models = valid_df["model_tag"].tolist()
    gen_contributions = (
        valid_df["generation_score"] * STAGE_WEIGHTS["generation"]
    ).tolist()
    cond_contributions = (
        valid_df["condensation_score"] * STAGE_WEIGHTS["condensation"]
    ).tolist()
    ref_contributions = (
        valid_df["refinement_score"] * STAGE_WEIGHTS["refinement"]
    ).tolist()
    map_contributions = (valid_df["mapping_score"] * STAGE_WEIGHTS["mapping"]).tolist()
    composites = valid_df["composite"].tolist()

    # --- Stacked bar chart data ---
    stacked_traces = json.dumps(
        [
            {
                "name": f"Generation ({int(STAGE_WEIGHTS['generation'] * 100)}%)",
                "y": models,
                "x": gen_contributions,
                "type": "bar",
                "orientation": "h",
                "marker": {"color": "#f472b6"},
                "hovertemplate": "%{y}: %{x:.3f}<extra>Generation</extra>",
            },
            {
                "name": f"Condensation ({int(STAGE_WEIGHTS['condensation'] * 100)}%)",
                "y": models,
                "x": cond_contributions,
                "type": "bar",
                "orientation": "h",
                "marker": {"color": "#fbbf24"},
                "hovertemplate": "%{y}: %{x:.3f}<extra>Condensation</extra>",
            },
            {
                "name": f"Refinement ({int(STAGE_WEIGHTS['refinement'] * 100)}%)",
                "y": models,
                "x": ref_contributions,
                "type": "bar",
                "orientation": "h",
                "marker": {"color": "#a78bfa"},
                "hovertemplate": "%{y}: %{x:.3f}<extra>Refinement</extra>",
            },
            {
                "name": f"Mapping ({int(STAGE_WEIGHTS['mapping'] * 100)}%)",
                "y": models,
                "x": map_contributions,
                "type": "bar",
                "orientation": "h",
                "marker": {"color": "#4ade80"},
                "hovertemplate": "%{y}: %{x:.3f}<extra>Mapping</extra>",
            },
        ]
    )

    # --- Heatmap data ---
    # Columns: individual normalised metrics + stage scores + composite
    heatmap_models = list(reversed(models))  # top = best
    heatmap_cols = ["Grnd", "Cov", "Spec", "Gen", "Cond", "Ref", "Map", "TQI"]
    heatmap_z = []
    heatmap_text = []

    for model in heatmap_models:
        row_data = valid_df[valid_df["model_tag"] == model].iloc[0]

        # Get raw normalised metrics for this model
        model_df = data.results_df[data.results_df["model_tag"] == model]
        gen_df = model_df[model_df["eval"] == "generation"]
        gen_metrics = data.detected_metrics.get("generation", {})

        raw_values = {}
        for metric_name in ["groundedness", "coverage", "specificity"]:
            cols = gen_metrics.get(metric_name, [])
            if cols and not gen_df.empty:
                vals = gen_df[cols].values.flatten()
                vals = vals[~np.isnan(vals.astype(float))]
                if len(vals) > 0:
                    scale = GENERATION_SCALE.get(metric_name, 1.0)
                    raw_values[metric_name] = float(np.mean(vals)) / scale

        z_row = [
            raw_values.get("groundedness", float("nan")),
            raw_values.get("coverage", float("nan")),
            raw_values.get("specificity", float("nan")),
            row_data["generation_score"],
            row_data["condensation_score"],
            row_data["refinement_score"],
            row_data["mapping_score"],
            row_data["composite"],
        ]
        text_row = [f"{v:.3f}" if pd.notna(v) else "-" for v in z_row]
        heatmap_z.append(z_row)
        heatmap_text.append(text_row)

    heatmap_trace = json.dumps(
        [
            {
                "z": heatmap_z,
                "x": heatmap_cols,
                "y": heatmap_models,
                "type": "heatmap",
                "colorscale": [
                    [0.0, "#0f172a"],
                    [0.4, "#1e3a5f"],
                    [0.6, "#2563eb"],
                    [0.8, "#38bdf8"],
                    [1.0, "#4ade80"],
                ],
                "zmin": 0,
                "zmax": 1,
                "text": heatmap_text,
                "texttemplate": "%{text}",
                "textfont": {"size": 11, "family": "Roboto Mono", "color": "#e2e8f0"},
                "hovertemplate": "%{y} | %{x}: %{z:.3f}<extra></extra>",
                "showscale": False,
            }
        ]
    )

    # Composite annotation labels at end of each bar
    annotations = json.dumps(
        [
            {
                "x": composites[i],
                "y": models[i],
                "text": f"  {composites[i]:.3f}",
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 11, "color": "#e2e8f0", "family": "Roboto Mono"},
            }
            for i in range(len(models))
        ]
    )

    bar_height = max(200, len(models) * 50)
    heatmap_height = max(180, len(heatmap_models) * 40 + 60)

    return f"""
        <div class="section-header">ThemeFinder Quality Index (TQI)</div>
        <div class="grid grid-2">
            <div class="card">
                <div class="card-title">Composite Score Breakdown</div>
                <div id="tqi-stacked-bar" style="height: {bar_height}px;"></div>
            </div>
            <div class="card">
                <div class="card-title">Normalised Metric Heatmap</div>
                <div id="tqi-heatmap" style="height: {heatmap_height}px;"></div>
            </div>
        </div>
        <script>
            Plotly.newPlot('tqi-stacked-bar', {stacked_traces}, {{
                barmode: 'stack',
                margin: {{t: 10, b: 30, l: 140, r: 60}},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{color: '#94a3b8', size: 11, family: 'Inter'}},
                xaxis: {{gridcolor: '#334155', range: [0, 1.05], title: 'Score'}},
                yaxis: {{gridcolor: '#334155'}},
                legend: {{orientation: 'h', y: -0.15, font: {{size: 10}}}},
                annotations: {annotations}
            }}, {{responsive: true, displayModeBar: false}});

            Plotly.newPlot('tqi-heatmap', {heatmap_trace}, {{
                margin: {{t: 10, b: 40, l: 140, r: 20}},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{color: '#94a3b8', size: 11, family: 'Inter'}},
                xaxis: {{side: 'top'}},
                yaxis: {{autorange: true}}
            }}, {{responsive: true, displayModeBar: false}});
        </script>
"""


def _html_footer() -> str:
    """Generate HTML footer."""
    return """
    </div>
</body>
</html>
"""


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Visualise benchmark results")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark ID to visualise (e.g., 20260202_134824)",
    )
    parser.add_argument(
        "--output",
        help="Output HTML report path (default: benchmark_report_{id}.html)",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Skip fetching data from Langfuse",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw data as JSON instead of tables/report",
    )
    args = parser.parse_args()

    console.print(f"\n[bold cyan]Loading benchmark: {args.benchmark}[/bold cyan]\n")

    # Load all data
    data = load_benchmark_data(args.benchmark, skip_langfuse=args.no_langfuse)

    if data.results_df.empty:
        console.print(f"[red]No data found for benchmark: {args.benchmark}[/red]")
        return

    # JSON output mode
    if args.json:
        output = {
            "benchmark_id": data.benchmark_id,
            "config": data.config,
            "detected_metrics": data.detected_metrics,
            "models": data.models,
            "eval_types": data.eval_types,
            "total_cost": data.total_cost,
            "avg_latency": data.avg_latency,
            "trace_count": data.total_traces,
            "langfuse_traces": data.langfuse_traces,
        }
        print(json.dumps(output, indent=2, default=str))
        return

    # Print terminal summary
    print_summary(data)
    print_model_table(data)
    print_eval_table(data)
    print_all_performance_tables(data)
    print_composite_table(data)

    # Generate HTML report
    output_path = args.output or f"benchmark_report_{args.benchmark}.html"
    generate_html_report(data, output_path)


if __name__ == "__main__":
    main()
