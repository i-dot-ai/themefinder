"""Diagnostics for the SystemOne classification stage.

Two views the comparison scripts print alongside their metrics:

- request structure: how the responses and themes translate into SystemOne
  requests (chunks, questions per request, payload size), plus a full sample
  request payload for inspection;
- probability report: the distribution of theme and evidence probabilities
  across the data — histograms, per-theme assignment rates, and the share of
  answers in the uncertain band around the threshold (candidates for
  escalation or threshold tuning).
"""

import json
import math

import msgspec
import pandas as pd

from themefinder.systemone import (
    _build_state,
    _response_questions,
    _theme_texts,
)

# Probabilities this close to a coin flip are candidates for escalation.
UNCERTAIN_BAND = (0.3, 0.7)
HISTOGRAM_BINS = 10
HISTOGRAM_BAR_WIDTH = 30


def build_sample_request(
    responses_df: pd.DataFrame,
    question: str,
    themes_df: pd.DataFrame,
    batch_size: int,
    model: str = "jev-latest",
) -> dict:
    """Build the exact JSON payload of the first chunk's SystemOne request."""
    chunk = responses_df.head(batch_size)
    theme_texts = _theme_texts(themes_df)
    questions: dict = {}
    for response_id in chunk["response_id"]:
        questions.update(_response_questions(response_id, theme_texts))
    return {
        "model": model,
        "state": _build_state(
            question, theme_texts, chunk.to_dict(orient="records")
        ),
        "questions": {
            key: msgspec.to_builtins(value) for key, value in questions.items()
        },
    }


def save_sample_request(item: dict, batch_size: int, output_path, limit=None) -> None:
    """Write one dataset item's first-chunk request payload to a JSON file."""
    responses_df = pd.DataFrame(item["input"]["responses"])
    if limit:
        responses_df = responses_df.head(limit)
    payload = build_sample_request(
        responses_df,
        item["input"]["question"],
        pd.DataFrame(item["input"]["topics"]),
        batch_size,
    )
    output_path.write_text(json.dumps(payload, indent=2))


def request_structure(
    responses_df: pd.DataFrame,
    question: str,
    themes_df: pd.DataFrame,
    batch_size: int,
) -> dict:
    """Summarise how the inputs translate into SystemOne requests."""
    sample = build_sample_request(responses_df, question, themes_df, batch_size)
    questions_per_response = len(themes_df) + 2  # themes + gives_reason + evidence
    return {
        "responses": len(responses_df),
        "themes": len(themes_df),
        "batch_size": batch_size,
        "requests": math.ceil(len(responses_df) / batch_size),
        "questions_per_response": questions_per_response,
        "questions_per_request": min(batch_size, len(responses_df))
        * questions_per_response,
        "sample_request_chars": len(json.dumps(sample)),
    }


def print_request_structure(info: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel

    Console().print(
        Panel(
            f"{info['responses']} responses × {info['themes']} themes → "
            f"{info['requests']} requests of ≤{info['batch_size']} responses, "
            f"{info['questions_per_request']} questions each "
            f"({info['questions_per_response']}/response: per-theme nouls + "
            f"gives_reason + evidence_rich), "
            f"~{info['sample_request_chars'] / 1000:.0f}kB payload/request",
            title="SystemOne request structure",
            expand=False,
        )
    )


def _distribution_stats(values: pd.Series, threshold: float) -> dict:
    low, high = UNCERTAIN_BAND
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.1)),
        "p90": float(values.quantile(0.9)),
        "share_above_threshold": float((values >= threshold).mean()),
        "share_uncertain": float(((values >= low) & (values <= high)).mean()),
        "histogram": [
            int(count)
            for count in pd.cut(
                values,
                bins=[i / HISTOGRAM_BINS for i in range(HISTOGRAM_BINS + 1)],
                include_lowest=True,
            )
            .value_counts(sort=False)
            .tolist()
        ],
    }


def probability_report(
    classified_df: pd.DataFrame, threshold: float, detail_threshold: float
) -> dict:
    """Compute probability distributions from a classification output."""
    theme_probabilities = pd.Series(
        [
            probability
            for probabilities in classified_df["theme_probabilities"]
            for probability in probabilities.values()
        ]
    )
    per_theme: dict[str, list[float]] = {}
    for probabilities in classified_df["theme_probabilities"]:
        for topic_id, probability in probabilities.items():
            per_theme.setdefault(topic_id, []).append(probability)

    return {
        "threshold": threshold,
        "detail_threshold": detail_threshold,
        "theme_probabilities": _distribution_stats(theme_probabilities, threshold),
        "per_theme": {
            topic_id: {
                "mean": float(pd.Series(values).mean()),
                "assignment_rate": float(
                    (pd.Series(values) >= threshold).mean()
                ),
            }
            for topic_id, values in sorted(per_theme.items())
        },
        "evidence_probabilities": _distribution_stats(
            classified_df["evidence_probability"], detail_threshold
        ),
    }


def _print_histogram(console, stats: dict, title: str, threshold: float) -> None:
    console.print(
        f"[bold]{title}[/] [dim](n={stats['count']}, mean={stats['mean']:.3f}, "
        f"median={stats['median']:.3f}, p10={stats['p10']:.3f}, "
        f"p90={stats['p90']:.3f})[/]"
    )
    peak = max(stats["histogram"]) or 1
    for i, count in enumerate(stats["histogram"]):
        low, high = i / HISTOGRAM_BINS, (i + 1) / HISTOGRAM_BINS
        bar = "█" * round(HISTOGRAM_BAR_WIDTH * count / peak)
        marker = " ←threshold" if low <= threshold < high else ""
        console.print(
            f"  {low:.1f}–{high:.1f} [cyan]{bar}[/] {count}[dim]{marker}[/]"
        )
    console.print(
        f"  ≥threshold: {stats['share_above_threshold']:.1%}   "
        f"uncertain ({UNCERTAIN_BAND[0]}–{UNCERTAIN_BAND[1]}): "
        f"{stats['share_uncertain']:.1%}\n"
    )


def print_probability_report(report: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    _print_histogram(
        console,
        report["theme_probabilities"],
        "Theme probabilities (all response × theme pairs)",
        report["threshold"],
    )
    _print_histogram(
        console,
        report["evidence_probabilities"],
        "Evidence-rich probabilities",
        report["detail_threshold"],
    )

    table = Table(title="Per-theme probabilities")
    table.add_column("Theme")
    table.add_column("Mean prob", justify="right")
    table.add_column("Assignment rate", justify="right")
    for topic_id, stats in report["per_theme"].items():
        table.add_row(
            topic_id, f"{stats['mean']:.3f}", f"{stats['assignment_rate']:.1%}"
        )
    console.print(table)
