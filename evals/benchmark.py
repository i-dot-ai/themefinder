#!/usr/bin/env python
"""Multi-model benchmark runner for ThemeFinder evaluations.

Runs evaluations across Azure OpenAI and GCP Vertex AI models (Gemini + Claude)
with configurable hyperparameters, tracks results in Langfuse, and generates
summary tables.

Usage:
    # All models (default)
    uv run python benchmark.py --dataset housing_S --runs 3

    # Azure OpenAI models only
    uv run python benchmark.py --provider azure --dataset housing_S

    # Vertex AI models only (Gemini + Claude)
    uv run python benchmark.py --provider vertex --dataset bbc_mission_public_purposes

    # Specific models across providers
    uv run python benchmark.py --models gpt-4o gemini-2.5-flash claude-haiku-4.5

Requires:
    - LLM_GATEWAY_URL and CONSULT_EVAL_LITELLM_API_KEY env vars
    - Vertex: Not yet implemented (TODO)
"""

import argparse
import asyncio
import json
import logging
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import dotenv
import nest_asyncio
import pandas as pd
from rich.console import Console
from rich.table import Table
from themefinder.llm import OpenAILLM

# Allow nested asyncio.run() calls (needed by eval modules)
nest_asyncio.apply()

# Monkey-patch openai with langfuse-openai for automatic LLM call tracing.
# Must happen before any OpenAILLM instances are created.
try:
    from langfuse.openai import openai as _langfuse_openai

    import openai

    openai.OpenAI = _langfuse_openai.OpenAI
    openai.AsyncOpenAI = _langfuse_openai.AsyncOpenAI
except ImportError:
    pass  # langfuse not installed, tracing disabled

# Fix DNS resolution issues with gRPC 1.58.0+ (must be set before gRPC imports)
# Switches from buggy C-ares resolver to native resolver
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    AZURE_OPENAI = "azure_openai"
    VERTEX_GEMINI = "vertex_gemini"
    VERTEX_CLAUDE = "vertex_claude"


import langfuse_utils  # noqa: E402
from eval_condensation import evaluate_condensation  # noqa: E402
from eval_generation import evaluate_generation  # noqa: E402
from eval_mapping import evaluate_mapping  # noqa: E402
from eval_refinement import evaluate_refinement  # noqa: E402
from eval_sentiment import evaluate_sentiment  # noqa: E402

console = Console()

EVAL_FUNCS = {
    "sentiment": evaluate_sentiment,
    "mapping": evaluate_mapping,
    "generation": evaluate_generation,
    "condensation": evaluate_condensation,
    "refinement": evaluate_refinement,
}


class ValidationErrorCounter(logging.Handler):
    """Logging handler that counts validation warnings/errors from themefinder."""

    # Only count warnings from these logger namespaces
    TRACKED_LOGGERS = ("themefinder",)

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """Reset counters for a new eval run."""
        self.validation_errors = 0
        self.total_warnings = 0
        self.error_details: list[str] = []

    def emit(self, record: logging.LogRecord):
        """Count validation-related log messages from tracked loggers."""
        # Only track warnings from themefinder namespace
        if not record.name.startswith(self.TRACKED_LOGGERS):
            return

        if record.levelno >= logging.WARNING:
            self.total_warnings += 1
            msg = record.getMessage()
            # Check for Pydantic validation errors
            if "validation error" in msg.lower():
                self.validation_errors += 1
                self.error_details.append(msg[:200])


# Benchmark context for structured logging — holds (model, eval_type, run_number)
_benchmark_context: ContextVar[tuple[str, str, int] | None] = ContextVar(
    "_benchmark_context", default=None
)


class _SingleLineFormatter(logging.Formatter):
    """Collapses multiline messages onto a single line for grepability."""

    def format(self, record: logging.LogRecord) -> str:
        result = super().format(record)
        return result.replace("\n", "\\n")


class BenchmarkLogFilter(logging.Filter):
    """Injects benchmark context into log records and filters by namespace."""

    ACCEPTED_PREFIXES = ("themefinder", "theme_finder", "synthetic")

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        is_relevant = (
            name.startswith(self.ACCEPTED_PREFIXES)
            or "/themefinder/" in name  # models.py uses __file__ as logger name
        )
        if not is_relevant:
            return False

        ctx = _benchmark_context.get(None)
        if ctx is not None:
            record.model_name, record.eval_type, record.run_number = ctx
        else:
            record.model_name = "-"
            record.eval_type = "-"
            record.run_number = "-"

        return True


@dataclass
class ModelConfig:
    """Configuration for a single model variant.

    Supports Azure OpenAI, GCP Vertex AI (Gemini), and Vertex AI (Claude) models.
    """

    name: str  # Display name for results
    deployment: str  # Model ID (Azure deployment name or Vertex model ID)
    provider: LLMProvider = LLMProvider.AZURE_OPENAI

    # Azure-specific
    reasoning_effort: str | None = None  # For gpt-5 models: low, medium, high

    # Vertex AI-specific
    project_id: str | None = None  # Falls back to GOOGLE_CLOUD_PROJECT env var
    location: str = "global"  # Vertex AI region
    thinking_budget: int | None = None  # For Gemini/Claude thinking tokens (e.g., 2048)

    # Common
    temperature: float = 0.0
    timeout: int = 600  # 10 min default

    def create_llm(self) -> OpenAILLM:
        """Factory method to create appropriate LLM instance."""
        match self.provider:
            case LLMProvider.AZURE_OPENAI:
                return self._create_azure_llm()
            case LLMProvider.VERTEX_GEMINI | LLMProvider.VERTEX_CLAUDE:
                raise NotImplementedError(
                    f"TODO: implement OpenAILLM adapter for {self.provider.value}"
                )
            case _:
                raise ValueError(f"Unknown provider: {self.provider}")

    def _create_azure_llm(self) -> OpenAILLM:
        """Create OpenAILLM instance via LLM gateway."""
        request_kwargs: dict[str, Any] = {}
        if self.reasoning_effort:
            pass  # Reasoning models don't support custom temperature
        else:
            request_kwargs["temperature"] = self.temperature
        return OpenAILLM(
            model=self.deployment,
            request_kwargs=request_kwargs,
            base_url=os.getenv("LLM_GATEWAY_URL"),
            api_key=os.getenv("CONSULT_EVAL_LITELLM_API_KEY"),
            timeout=self.timeout,
        )

    @property
    def tag(self) -> str:
        """Tag for Langfuse identification."""
        if self.reasoning_effort:
            return f"{self.name}_{self.reasoning_effort}"
        if self.thinking_budget:
            return f"{self.name}_thinking{self.thinking_budget}"
        return self.name


# All available model configurations by provider
# Azure OpenAI models - use dated versions that support structured outputs
AZURE_MODELS = [
    ModelConfig(name="gpt-4o", deployment="gpt-4o-2024-08-06-sweden"),
    ModelConfig(name="gpt-4.1", deployment="gpt-4.1-sweden-2025-03"),
    ModelConfig(name="gpt-4.1-mini", deployment="gpt-4.1-mini"),
    ModelConfig(name="gpt-5-nano", deployment="gpt-5-nano", reasoning_effort="low"),
    ModelConfig(name="gpt-5-mini", deployment="gpt-5-mini", reasoning_effort="low"),
]

# Vertex AI models (Gemini and Claude)
# Requires: uv pip install -e ".[vertex]" and GOOGLE_CLOUD_PROJECT env var
VERTEX_MODELS = [
    # Gemini models - standard (no thinking)
    ModelConfig(
        name="gemini-2.5-flash",
        deployment="gemini-2.5-flash",
        provider=LLMProvider.VERTEX_GEMINI,
    ),
    ModelConfig(
        name="gemini-2.5-flash-lite",
        deployment="gemini-2.5-flash-lite",
        provider=LLMProvider.VERTEX_GEMINI,
    ),
    ModelConfig(
        name="gemini-3-flash-preview",
        deployment="gemini-3-flash-preview",
        provider=LLMProvider.VERTEX_GEMINI,
    ),
    # Gemini thinking models removed: thinking_budget conflicts with
    # structured output in google-genai SDK, causing empty {} responses.
    # See: https://github.com/googleapis/python-genai/issues/782
    # ModelConfig(
    #     name="gemini-2.5-pro",
    #     deployment="gemini-2.5-pro",
    #     provider=LLMProvider.VERTEX_GEMINI,
    # ),
    # Claude models via Vertex AI Model Garden - standard (no thinking)
    ModelConfig(
        name="claude-haiku-4.5",
        deployment="claude-haiku-4-5@20251001",
        provider=LLMProvider.VERTEX_CLAUDE,
    ),
    # Claude thinking model disabled: thinking_budget conflicts with
    # tool_choice in ThemeFinder pipeline (structured output requires forced tool use).
    # See: "Thinking may not be enabled when tool_choice forces tool use."
    # ModelConfig(
    #     name="claude-haiku-4.5-thinking",
    #     deployment="claude-haiku-4-5@20251001",
    #     provider=LLMProvider.VERTEX_CLAUDE,
    #     thinking_budget=2048,
    # ),
    # ModelConfig(
    #     name="claude-sonnet-4.5",
    #     deployment="claude-sonnet-4-5@20250929",
    #     provider=LLMProvider.VERTEX_CLAUDE,
    # ),
]

# Combined model registry by provider
MODEL_REGISTRY: dict[str, list[ModelConfig]] = {
    "azure": AZURE_MODELS,
    "vertex": VERTEX_MODELS,
    "all": AZURE_MODELS + VERTEX_MODELS,
}

# Default models (Azure) for backwards compatibility
DEFAULT_MODELS = AZURE_MODELS


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    dataset: str
    models: list[ModelConfig] = field(default_factory=lambda: DEFAULT_MODELS)
    runs_per_model: int = 5
    evals: list[str] = field(
        default_factory=lambda: [
            "sentiment",
            "mapping",
            "generation",
            "condensation",
            "refinement",
        ]
    )
    judge_model: str | None = (
        None  # Azure deployment name for judge LLM (e.g. "gpt-4o-2024-08-06")
    )


@dataclass
class RunResult:
    """Result from a single evaluation run."""

    model: str
    model_tag: str
    run_number: int
    eval_type: str
    dataset: str
    session_id: str
    scores: dict[str, float]
    timestamp: datetime
    duration_seconds: float = 0.0  # Wall-clock duration of eval
    outputs: dict[str, Any] = field(
        default_factory=dict
    )  # Non-numeric pipeline outputs

    # Token metrics (from Langfuse)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost and latency (from Langfuse)
    cost_usd: float = 0.0
    latency_seconds: float = 0.0


class BenchmarkRunner:
    """Orchestrates benchmark runs across models and evaluations."""

    def __init__(self, config: BenchmarkConfig, output_dir: str = "benchmark_results"):
        self.config = config
        self.results: list[RunResult] = []
        self.benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir
        self._output_path: Path | None = None
        self._results_lock = asyncio.Lock()

    def _create_log_handler(self) -> logging.FileHandler:
        """Create file handler for structured benchmark logging."""
        log_path = self._get_output_path() / "benchmark.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            _SingleLineFormatter(
                "%(asctime)s | %(eval_type)s | %(model_name)s | run:%(run_number)s"
                " | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.addFilter(BenchmarkLogFilter())
        return handler

    async def run(self) -> list[RunResult]:
        """Execute full benchmark suite with deployments in parallel, variants sequential."""
        # Set up structured file logging for this benchmark run
        root_logger = logging.getLogger()
        original_level = root_logger.level
        log_handler = self._create_log_handler()
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.DEBUG)

        try:
            return await self._run_benchmark()
        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()
            root_logger.setLevel(original_level)

    async def _run_benchmark(self) -> list[RunResult]:
        """Inner benchmark execution (wrapped by run() for logging setup)."""
        total_runs = (
            len(self.config.models)
            * self.config.runs_per_model
            * len(self.config.evals)
        )
        runs_per_model = self.config.runs_per_model * len(self.config.evals)

        # Group models by deployment to avoid rate limit contention
        deployment_groups: dict[str, list[ModelConfig]] = {}
        for model in self.config.models:
            if model.deployment not in deployment_groups:
                deployment_groups[model.deployment] = []
            deployment_groups[model.deployment].append(model)

        console.print(
            f"\n[bold cyan]Starting Benchmark: {self.benchmark_id}[/bold cyan]"
        )
        console.print(f"  Dataset: {self.config.dataset}")
        console.print(
            f"  Models: {len(self.config.models)} across {len(deployment_groups)} deployments"
        )
        console.print(
            "  Strategy: [bold]parallel by deployment, sequential within[/bold]"
        )
        console.print(f"  Runs per model: {self.config.runs_per_model}")
        console.print(f"  Evals: {', '.join(self.config.evals)}")
        console.print(f"  Total runs: {total_runs} ({runs_per_model} per model)\n")

        # Run deployment groups in parallel, models within each group sequentially
        deployment_tasks = [
            self._run_deployment_group(deployment, models)
            for deployment, models in deployment_groups.items()
        ]
        results = await asyncio.gather(*deployment_tasks, return_exceptions=True)

        # Report any deployment-level failures
        for deployment, result in zip(deployment_groups.keys(), results):
            if isinstance(result, Exception):
                console.print(f"[bold red][{deployment}] FAILED: {result}[/bold red]")

        return self.results

    async def _run_deployment_group(
        self, deployment: str, models: list[ModelConfig]
    ) -> list[RunResult]:
        """Run all models for a deployment sequentially to avoid rate limit contention."""
        console.print(
            f"[dim][{deployment}] Starting {len(models)} model variant(s)[/dim]"
        )
        group_results = []

        for model_config in models:
            model_results = await self._run_model(model_config)
            group_results.extend(model_results)

        console.print(f"[bold green][{deployment}] All variants complete![/bold green]")
        return group_results

    async def _run_model(self, model_config: ModelConfig) -> list[RunResult]:
        """Run all evals for a single model (sequential within model)."""
        # Each model gets its own validation error counter
        error_counter = ValidationErrorCounter()
        logging.getLogger().addHandler(error_counter)

        model_results = []
        runs_per_model = self.config.runs_per_model * len(self.config.evals)
        current = 0

        try:
            for run_num in range(1, self.config.runs_per_model + 1):
                for eval_type in self.config.evals:
                    current += 1
                    console.print(
                        f"[bold][{model_config.tag}][/bold] "
                        f"[{current}/{runs_per_model}] Run {run_num} | {eval_type}"
                    )

                    # Retry loop for transient network errors
                    max_eval_retries = 3
                    retry_delay = 2.0

                    for attempt in range(max_eval_retries):
                        try:
                            result = await self._run_single_eval(
                                model_config=model_config,
                                run_number=run_num,
                                eval_type=eval_type,
                                error_counter=error_counter,
                            )
                            model_results.append(result)

                            # Thread-safe results collection and saving
                            async with self._results_lock:
                                self.results.append(result)
                                self._print_scores(result.scores)
                                self._save_incremental()
                                self._save_outputs(result)
                            break  # Success, exit retry loop

                        except Exception as e:
                            error_str = str(e)

                            # Check if error is retryable (transient network issues)
                            is_retryable = any(
                                [
                                    "ENOTFOUND" in error_str,
                                    "ECONNREFUSED" in error_str,
                                    "ECONNRESET" in error_str,
                                    "DNS" in error_str,
                                    "temporarily unavailable" in error_str.lower(),
                                    "503" in error_str,
                                ]
                            )

                            if is_retryable and attempt < max_eval_retries - 1:
                                delay = retry_delay * (2**attempt)
                                console.print(
                                    f"  [yellow]Transient error (attempt {attempt + 1}/{max_eval_retries}), "
                                    f"retrying in {delay:.1f}s...[/yellow]"
                                )
                                await asyncio.sleep(delay)
                                continue

                            # Non-retryable or exhausted retries
                            console.print(
                                f"  [bold][{model_config.tag}][/bold] [red]ERROR: {e}[/red]"
                            )
                            break

        finally:
            logging.getLogger().removeHandler(error_counter)

        console.print(f"[bold green][{model_config.tag}] Complete![/bold green]")
        return model_results

    async def _run_single_eval(
        self,
        model_config: ModelConfig,
        run_number: int,
        eval_type: str,
        error_counter: ValidationErrorCounter,
    ) -> RunResult:
        """Run a single evaluation."""
        # Set benchmark context for structured file logging
        token = _benchmark_context.set((model_config.name, eval_type, run_number))
        try:
            return await asyncio.wait_for(
                self._execute_eval(model_config, run_number, eval_type, error_counter),
                timeout=1200,
            )
        finally:
            _benchmark_context.reset(token)

    async def _execute_eval(
        self,
        model_config: ModelConfig,
        run_number: int,
        eval_type: str,
        error_counter: ValidationErrorCounter,
    ) -> RunResult:
        """Execute a single evaluation (called within benchmark context)."""
        # Reset error counter for this run
        error_counter.reset()

        eval_func = EVAL_FUNCS.get(eval_type)
        if not eval_func:
            raise ValueError(f"Unknown eval type: {eval_type}")

        # Create session ID with benchmark context
        session_id = (
            f"benchmark_{self.benchmark_id}_"
            f"{model_config.tag}_{eval_type}_run{run_number}"
        )

        # Set up Langfuse context with benchmark metadata
        langfuse_ctx = langfuse_utils.get_langfuse_context(
            session_id=session_id,
            eval_type=eval_type,
            metadata={
                "benchmark_id": self.benchmark_id,
                "model": model_config.name,
                "model_tag": model_config.tag,
                "deployment": model_config.deployment,
                "reasoning_effort": model_config.reasoning_effort,
                "run_number": run_number,
                "dataset": self.config.dataset,
            },
            tags=[
                self.config.dataset,
                f"model:{model_config.tag}",
                f"run:{run_number}",
                f"benchmark:{self.benchmark_id}",
            ],
        )

        llm = model_config.create_llm()

        # Create dedicated judge LLM if configured (separates judge from task model)
        judge_llm = None
        if self.config.judge_model:
            judge_llm = OpenAILLM(
                model=self.config.judge_model,
                request_kwargs={"temperature": 0},
                base_url=os.getenv("LLM_GATEWAY_URL"),
                api_key=os.getenv("CONSULT_EVAL_LITELLM_API_KEY"),
                timeout=600,
            )

        # Run the evaluation with timing, wrapped in trace_context for v3 API
        # This ensures session_id/tags/metadata are properly attached to traces
        start_time = time.perf_counter()
        # Only pass judge_llm to evals that use LLM-as-judge
        evals_with_judge = {"generation", "condensation", "refinement"}

        with langfuse_utils.trace_context(langfuse_ctx, name=f"{eval_type}_eval"):
            kwargs = {
                "dataset": self.config.dataset,
                "llm": llm,
                "langfuse_ctx": langfuse_ctx,
            }
            if judge_llm and eval_type in evals_with_judge:
                kwargs["judge_llm"] = judge_llm
            scores = await eval_func(**kwargs)
        end_time = time.perf_counter()
        duration_seconds = end_time - start_time

        langfuse_utils.flush(langfuse_ctx)

        # Extract token/cost metrics from Langfuse
        metrics = langfuse_utils.extract_session_metrics(
            client=langfuse_ctx.client,
            session_id=session_id,
            benchmark_tag=f"benchmark:{self.benchmark_id}",
        )

        # Split returned dict into numeric scores and non-numeric outputs
        all_scores = {}
        all_outputs = {}
        for key, value in (scores or {}).items():
            if isinstance(value, (int, float)):
                all_scores[key] = value
            else:
                all_outputs[key] = value

        # Add validation error metrics to scores
        all_scores["validation_errors"] = error_counter.validation_errors
        all_scores["total_warnings"] = error_counter.total_warnings

        return RunResult(
            model=model_config.name,
            model_tag=model_config.tag,
            run_number=run_number,
            eval_type=eval_type,
            dataset=self.config.dataset,
            session_id=session_id,
            scores=all_scores,
            outputs=all_outputs,
            timestamp=datetime.now(),
            duration_seconds=duration_seconds,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            cost_usd=metrics.cost_usd,
            latency_seconds=metrics.latency_seconds,
        )

    def _print_scores(self, scores: dict[str, float]) -> None:
        """Print scores inline."""
        if not scores:
            console.print("  [dim]No scores returned[/dim]")
            return

        # Separate validation metrics from other scores
        validation_errors = scores.get("validation_errors", 0)
        other_scores = {
            k: v
            for k, v in scores.items()
            if k not in ("validation_errors", "total_warnings")
            and isinstance(v, (int, float))
        }

        score_strs = [f"{k}: {v:.3f}" for k, v in other_scores.items()]

        if score_strs:
            console.print(f"  [green]{', '.join(score_strs)}[/green]", end="")

        if validation_errors > 0:
            console.print(
                f"  [yellow]({validation_errors} validation retries)[/yellow]"
            )
        else:
            console.print()  # newline

    def _get_output_path(self) -> Path:
        """Get or create output path for this benchmark."""
        if self._output_path is None:
            self._output_path = Path(self.output_dir) / self.benchmark_id
            self._output_path.mkdir(parents=True, exist_ok=True)
        return self._output_path

    def _save_incremental(self) -> None:
        """Save current results incrementally."""
        output_path = self._get_output_path()

        # Save raw results
        raw_df = self.generate_summary_table()
        raw_df.to_csv(output_path / "raw_results.csv", index=False)

        # Save config (first time only, but cheap to overwrite)
        config_dict = {
            "benchmark_id": self.benchmark_id,
            "dataset": self.config.dataset,
            "runs_per_model": self.config.runs_per_model,
            "evals": self.config.evals,
            "judge_model": self.config.judge_model,
            "models": [
                {
                    "name": m.name,
                    "deployment": m.deployment,
                    "reasoning_effort": m.reasoning_effort,
                }
                for m in self.config.models
            ],
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def _save_outputs(self, result: RunResult) -> None:
        """Save non-numeric pipeline outputs as JSON for a single run."""
        if not result.outputs:
            return

        outputs_dir = self._get_output_path() / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        filename = f"{result.model_tag}_{result.eval_type}_run{result.run_number}.json"
        with open(outputs_dir / filename, "w") as f:
            json.dump(result.outputs, f, indent=2, default=str)

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary DataFrame from results."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            row = {
                "model": result.model,
                "model_tag": result.model_tag,
                "run": result.run_number,
                "eval": result.eval_type,
                "session_id": result.session_id,
                "duration_seconds": result.duration_seconds,
                # Token metrics
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                # Cost and latency
                "cost_usd": result.cost_usd,
                "latency_seconds": result.latency_seconds,
            }
            # Flatten scores into columns
            for score_name, score_value in result.scores.items():
                if isinstance(score_value, (int, float)):
                    row[score_name] = score_value
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_aggregated_table(self) -> pd.DataFrame:
        """Generate aggregated results (mean ± std) per model/eval."""
        df = self.generate_summary_table()
        if df.empty:
            return df

        # Get score columns (numeric columns that aren't metadata)
        metadata_cols = {"model", "model_tag", "run", "eval", "session_id"}
        score_cols = [c for c in df.columns if c not in metadata_cols]

        if not score_cols:
            return pd.DataFrame()

        # Aggregate by model_tag and eval
        agg_rows = []
        for (model_tag, eval_type), group in df.groupby(["model_tag", "eval"]):
            row = {"model_tag": model_tag, "eval": eval_type, "n_runs": len(group)}
            for col in score_cols:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        row[f"{col}_mean"] = values.mean()
                        row[f"{col}_std"] = values.std()
            agg_rows.append(row)

        return pd.DataFrame(agg_rows)

    def print_results_table(self) -> None:
        """Print results as Rich table."""
        agg_df = self.generate_aggregated_table()
        if agg_df.empty:
            console.print("[yellow]No results to display[/yellow]")
            return

        console.print("\n[bold cyan]Benchmark Results Summary[/bold cyan]\n")

        # Create table for each eval type
        for eval_type in self.config.evals:
            eval_df = agg_df[agg_df["eval"] == eval_type]
            if eval_df.empty:
                continue

            table = Table(title=f"{eval_type.title()} Evaluation")
            table.add_column("Model", style="cyan")
            table.add_column("Runs", justify="right")

            # Find score columns for this eval
            score_cols = [c for c in eval_df.columns if c.endswith("_mean")]
            for col in score_cols:
                metric_name = col.replace("_mean", "")
                table.add_column(metric_name, justify="right")

            for _, row in eval_df.iterrows():
                values = [str(row["model_tag"]), str(int(row["n_runs"]))]
                for col in score_cols:
                    mean = row.get(col, float("nan"))
                    std_col = col.replace("_mean", "_std")
                    std = row.get(std_col, float("nan"))
                    if pd.notna(mean):
                        values.append(f"{mean:.3f} ± {std:.3f}")
                    else:
                        values.append("-")
                table.add_row(*values)

            console.print(table)
            console.print()

    def save_results(self) -> None:
        """Save final aggregated results."""
        output_path = self._get_output_path()

        # Final save of raw results
        raw_df = self.generate_summary_table()
        raw_df.to_csv(output_path / "raw_results.csv", index=False)

        # Save aggregated results (only meaningful at the end)
        agg_df = self.generate_aggregated_table()
        agg_df.to_csv(output_path / "aggregated_results.csv", index=False)

        console.print(f"[green]Results saved to {output_path}[/green]")


def query_langfuse_costs(benchmark_id: str) -> pd.DataFrame:
    """Query Langfuse for cost data from benchmark runs.

    Args:
        benchmark_id: The benchmark ID to query.

    Returns:
        DataFrame with cost data per model/eval.
    """
    try:
        from langfuse import Langfuse

        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        base_url = os.getenv("LANGFUSE_BASE_URL")

        if not all([secret_key, public_key, base_url]):
            return pd.DataFrame()

        client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url,
        )

        # Fetch traces with benchmark tag
        traces = client.api.trace.list(
            tags=[f"benchmark:{benchmark_id}"],
            limit=1000,
        )

        if not traces.data:
            console.print("[yellow]No traces found for benchmark[/yellow]")
            return pd.DataFrame()

        rows = []
        for trace in traces.data:
            metadata = trace.metadata or {}
            rows.append(
                {
                    "model_tag": metadata.get("model_tag", "unknown"),
                    "eval": metadata.get("eval_type", "unknown"),
                    "run": metadata.get("run_number", 0),
                    "input_tokens": trace.usage.input if trace.usage else 0,
                    "output_tokens": trace.usage.output if trace.usage else 0,
                    "total_tokens": trace.usage.total if trace.usage else 0,
                    "cost_usd": trace.usage.total_cost if trace.usage else 0,
                }
            )

        return pd.DataFrame(rows)

    except Exception as e:
        console.print(f"[red]Failed to query Langfuse costs: {e}[/red]")
        return pd.DataFrame()


def print_cost_summary(cost_df: pd.DataFrame) -> None:
    """Print cost summary table."""
    if cost_df.empty:
        return

    console.print("\n[bold cyan]Cost Summary[/bold cyan]\n")

    # Aggregate by model
    agg = (
        cost_df.groupby("model_tag")
        .agg(
            {
                "input_tokens": "sum",
                "output_tokens": "sum",
                "total_tokens": "sum",
                "cost_usd": "sum",
            }
        )
        .reset_index()
    )

    table = Table(title="Total Costs by Model")
    table.add_column("Model", style="cyan")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Cost (USD)", justify="right", style="green")

    for _, row in agg.iterrows():
        table.add_row(
            row["model_tag"],
            f"{row['input_tokens']:,}",
            f"{row['output_tokens']:,}",
            f"{row['total_tokens']:,}",
            f"${row['cost_usd']:.4f}",
        )

    # Add total row
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{agg['input_tokens'].sum():,}[/bold]",
        f"[bold]{agg['output_tokens'].sum():,}[/bold]",
        f"[bold]{agg['total_tokens'].sum():,}[/bold]",
        f"[bold green]${agg['cost_usd'].sum():.4f}[/bold green]",
    )

    console.print(table)


async def main():
    """Main entry point."""
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run ThemeFinder benchmark across Azure and Vertex AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models (default)
  uv run python benchmark.py --dataset housing_S --runs 3

  # Run Azure models only
  uv run python benchmark.py --provider azure --dataset housing_S

  # Run Vertex AI models only (Gemini + Claude)
  uv run python benchmark.py --provider vertex --dataset bbc_mission_public_purposes

  # Run specific models by name
  uv run python benchmark.py --models gpt-4o gemini-2.5-flash --runs 2

  # Run only sentiment and mapping evals
  uv run python benchmark.py --evals sentiment mapping
        """,
    )
    parser.add_argument(
        "--dataset",
        default="housing_S",
        help="Dataset identifier (e.g., housing_S, gambling_XS, bbc_mission_public_purposes)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per model (default: 5)",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=["sentiment", "mapping", "generation", "condensation", "refinement"],
        help="Evaluations to run (default: all)",
    )
    parser.add_argument(
        "--provider",
        choices=["azure", "vertex", "all"],
        default="all",
        help="Provider to use: azure, vertex, or all (default)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names/tags to run (overrides --provider)",
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Vertex AI location (default: global)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Azure deployment name for judge LLM (e.g. gpt-4o-2024-08-06). If set, judge evaluations use this model instead of the task model.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick CI smoke test: gambling_XS dataset, 1 run, azure provider, gpt-4.1-mini only.",
    )
    args = parser.parse_args()

    # Apply --quick preset (overrides other args)
    if args.quick:
        args.dataset = "gambling_XS"
        args.runs = 1
        args.provider = "azure"
        args.models = ["gpt-4.1-mini"]

    # Get models based on provider or specific model names
    if args.models:
        # Filter from all available models by name/tag
        all_models = MODEL_REGISTRY["all"]
        models = [
            m for m in all_models if m.name in args.models or m.tag in args.models
        ]
        if not models:
            available = sorted(set(m.name for m in all_models))
            console.print(f"[red]No matching models. Available: {available}[/red]")
            return
    else:
        # Use provider-based selection
        models = MODEL_REGISTRY.get(args.provider, MODEL_REGISTRY["all"])

    # Apply location override for Vertex models
    if args.location != "global":
        models = [
            ModelConfig(
                name=m.name,
                deployment=m.deployment,
                provider=m.provider,
                reasoning_effort=m.reasoning_effort,
                location=args.location,
                temperature=m.temperature,
                timeout=m.timeout,
            )
            if m.provider in (LLMProvider.VERTEX_GEMINI, LLMProvider.VERTEX_CLAUDE)
            else m
            for m in models
        ]

    config = BenchmarkConfig(
        dataset=args.dataset,
        models=models,
        runs_per_model=args.runs,
        evals=args.evals,
        judge_model=args.judge_model,
    )

    runner = BenchmarkRunner(config)

    # Print configuration summary
    providers_used = sorted(set(m.provider.value for m in models))
    console.print("\n[bold cyan]ThemeFinder Benchmark[/bold cyan]")
    console.print(f"  Provider(s): {', '.join(providers_used)}")
    console.print(f"  Models: {[m.name for m in models]}")
    if any(m.provider != LLMProvider.AZURE_OPENAI for m in models):
        console.print(f"  Vertex location: {args.location}")
    console.print()

    try:
        await runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted[/yellow]")

    # Print results
    runner.print_results_table()

    # Query and print costs
    cost_df = query_langfuse_costs(runner.benchmark_id)
    print_cost_summary(cost_df)

    # Save results
    runner.save_results()

    # Generate HTML report
    try:
        from visualise_benchmark import generate_html_report, load_benchmark_data

        data = load_benchmark_data(runner.benchmark_id, skip_langfuse=True)
        report_path = str(runner._get_output_path() / "report.html")
        generate_html_report(data, report_path)
        console.print(f"[green]HTML report: {report_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not generate HTML report: {e}[/yellow]")

    # Machine-parseable output for CI
    print(f"BENCHMARK_ID={runner.benchmark_id}")


if __name__ == "__main__":
    asyncio.run(main())
