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
        "state": _build_state(question, theme_texts, chunk.to_dict(orient="records")),
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
    from rich.tree import Tree

    root = Tree(
        f"[bold]SystemOne request[/] — {info['requests']} requests of "
        f"≤{info['batch_size']} responses, {info['questions_per_request']} "
        f"questions each, ~{info['sample_request_chars'] / 1000:.0f}kB payload"
    )
    state = root.add("[cyan]state[/] [dim](shared context, sent once per request)[/]")
    state.add("question — the consultation question")
    state.add(
        f"topics — {{topic_id: text}} × {info['themes']} "
        "[dim](each defined once, referenced by every question)[/]"
    )
    state.add(
        "topic_match_definition / gives_reason_definition / "
        "evidence_rich_definition [dim](judgement rubrics, stated once)[/]"
    )
    state.add(f"responses — [{{response_id, response}}] × ≤{info['batch_size']}")
    questions = root.add(
        f"[magenta]questions[/] "
        f"[dim]({info['questions_per_response']} per response — JSON pointers "
        "into the state)[/]"
    )
    questions.add(
        'r<id>_theme_<topic> — noul {"question", "response_id", "topic_id"} '
        f"× {info['themes']} themes"
    )
    questions.add("r<id>_gives_reason — noul [dim](drives the fallback labels)[/]")
    questions.add("r<id>_evidence_rich — noul [dim](detail detection)[/]")
    Console().print(root)


def print_pipeline_flow(n_themes: int | None, batch_size: int) -> None:
    """Show the five pipeline stages and which ones SystemOne replaces."""
    from rich.console import Console
    from rich.table import Table

    themes = str(n_themes) if n_themes else "N"
    table = Table(
        title="Pipeline stages: what the hybrid replaces",
        caption=(
            "Both pipelines produce the same outputs: themes, mapping (labels "
            "per response) and detailed_responses (evidence_rich per response)."
        ),
    )
    table.add_column("Stage")
    table.add_column("LLM pipeline (find_themes)")
    table.add_column("Hybrid pipeline (find_themes_hybrid)")

    for stage, task in [
        ("1. Theme generation", "draft themes from responses"),
        ("2. Theme condensation", "merge duplicate themes"),
        ("3. Theme refinement", f"finalise {themes} themes"),
    ]:
        table.add_row(
            f"{stage}\n[dim]{task}[/]",
            "LLM (generative)",
            "[dim]same — unchanged[/]",
        )
    table.add_row(
        "4. Theme mapping\n[dim]label each response with themes[/]",
        "LLM: prompts of 20 responses,\nfree-text JSON output",
        f"[magenta]replaced by jev SystemOne[/] ┐\none request per {batch_size} "
        "responses;",
    )
    table.add_row(
        "5. Detail detection\n[dim]flag evidence-rich responses[/]",
        "LLM: a second full pass\nover every response",
        f"[magenta]replaced by jev SystemOne[/] ┘\nmerged into the SAME request "
        f"—\n{themes} theme nouls + gives_reason +\nevidence_rich per response, "
        "all\nanswered in parallel with\ncalibrated probabilities",
    )
    Console().print(table)


def result_caveats(part_results: dict, detail_threshold: float) -> list[str]:
    """Validate one question part's results and explain anything misleading."""
    caveats = [
        "Mapping 'accuracy' is exact-set match — a response scores 0 unless its "
        "full label set matches the ground truth. F1 gives per-theme partial "
        "credit and is the fairer headline."
    ]
    runs = part_results["runs"]
    n_responses = part_results["n_responses"]
    metrics_by_backend = {}
    for run in runs:
        metrics_by_backend.setdefault(run["backend"], {}).update(run["metrics"])

    token_totals = {
        backend: sum(run["input_tokens"] for run in runs if run["backend"] == backend)
        for backend in metrics_by_backend
    }
    if "llm" in token_totals and "systemone" in token_totals and token_totals["llm"]:
        ratio = token_totals["systemone"] / token_totals["llm"]
        if ratio > 3:
            caveats.append(
                f"SystemOne uses ~{ratio:.0f}× more input tokens than the LLM — "
                "that is expected (one question per response × theme) and not a "
                "cost problem: jev input is $0.042/1M with free output, so "
                "compare the cost column, not the token columns."
            )

    for backend, metrics in metrics_by_backend.items():
        kappa = metrics.get("detail_cohen_kappa")
        accuracy = metrics.get("detail_accuracy")
        if kappa is not None and accuracy is not None and abs(kappa) < 0.05:
            caveats.append(
                f"{backend}: evidence kappa ≈ 0 means its predictions are "
                f"effectively single-class at this threshold, so the accuracy "
                f"of {accuracy:.2f} only mirrors class prevalence. Judge it by "
                "AUC and the best_threshold diagnostic instead."
            )
        auc = metrics.get("detail_auc")
        if auc is not None and auc < 0.65:
            caveats.append(
                f"{backend}: evidence AUC of {auc:.2f} is close to chance. When "
                "both backends also score poorly here, treat this question "
                "part's detail ground truth as unreliable rather than "
                "concluding either backend failed."
            )
        best_threshold = metrics.get("detail_best_threshold")
        if best_threshold is not None and abs(best_threshold - detail_threshold) > 0.1:
            caveats.append(
                f"{backend}: the swept best evidence threshold "
                f"({best_threshold:.2f}) is far from the configured "
                f"{detail_threshold:.2f} — consider --detail-threshold "
                f"{best_threshold:.2f}, but note it was tuned on this same "
                "ground truth, so validate it on the other question part."
            )
        ci_low = metrics.get("map_f1_ci_low")
        ci_high = metrics.get("map_f1_ci_high")
        if ci_low is not None and ci_high is not None and ci_high - ci_low > 0.1:
            caveats.append(
                f"{backend}: mapping F1 95% CI is ±{(ci_high - ci_low) / 2:.2f} "
                f"at n={n_responses} — F1 differences smaller than this are "
                "within noise."
            )

    agreement = part_results.get("mapping_agreement") or {}
    backend_f1s = [
        metrics.get("map_f1_score")
        for metrics in metrics_by_backend.values()
        if metrics.get("map_f1_score") is not None
    ]
    if (
        agreement.get("f1_score") is not None
        and len(backend_f1s) == 2
        and agreement["f1_score"] < min(backend_f1s)
    ):
        caveats.append(
            "The backends agree with the ground truth more than with each "
            "other — multi-label theme assignment has genuine ambiguity, so "
            "low cross-backend agreement is not by itself an error signal."
        )

    return caveats


def print_caveats(caveats: list[str]) -> None:
    from rich.console import Console
    from rich.panel import Panel

    if not caveats:
        return
    Console().print(
        Panel(
            "\n".join(f"• {caveat}" for caveat in caveats),
            title="Notes & validation",
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
                "assignment_rate": float((pd.Series(values) >= threshold).mean()),
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
        console.print(f"  {low:.1f}–{high:.1f} [cyan]{bar}[/] {count}[dim]{marker}[/]")
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
