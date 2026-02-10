"""Langfuse-compatible evaluators for ThemeFinder evaluations.

Provides evaluator functions that can be used with Langfuse's run_experiment() API.
Each evaluator returns a Langfuse Evaluation object with name, value, and optional comment.
"""

import json
import logging
import random
from functools import lru_cache
from typing import Any

from langchain_core.output_parsers.json import parse_json_markdown

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger("themefinder.evals.evaluators")

# Minimum score (0-5) to consider a topic well-grounded or captured
GROUNDEDNESS_THRESHOLD = 3

# Score mapping from ternary decisions to numeric values (0-5 scale)
DECISION_SCORES = {
    "STRONG": 5,
    "PARTIAL": 3,
    "NO": 0,
}


def _shuffle_themes(themes: list[dict] | dict) -> list[dict] | dict:
    """Shuffle theme order to reduce positional bias in LLM-as-judge.

    Args:
        themes: Themes as a list of dicts or a dict keyed by label.

    Returns:
        Shuffled copy in the same format as input.
    """
    if isinstance(themes, list):
        shuffled = list(themes)
        random.shuffle(shuffled)
        return shuffled

    if isinstance(themes, dict):
        keys = list(themes.keys())
        random.shuffle(keys)
        return {k: themes[k] for k in keys}

    return themes


def _parse_evaluation_response(response_content: str) -> dict[str, Any]:
    """Parse the binary judgment response from the generation eval prompt.

    Extracts per-theme evaluations and maps decisions to numeric scores.

    Args:
        response_content: Raw JSON response from the judge LLM.

    Returns:
        Dict with scores, average, threshold counts, and per-theme details.
    """
    parsed = parse_json_markdown(response_content)
    evaluations = parsed.get("evaluations", parsed)

    scores = []
    details = []

    for theme_label, evaluation in evaluations.items():
        if isinstance(evaluation, dict):
            decision = evaluation.get("decision", "NO").upper()
            score = DECISION_SCORES.get(decision, 0)
            scores.append(score)
            details.append({
                "theme": theme_label,
                "matched_to": evaluation.get("matched_to", "none"),
                "decision": decision,
                "reasoning": evaluation.get("reasoning", ""),
                "score": score,
            })
        elif isinstance(evaluation, (int, float)):
            # Backwards compatibility with old 0-5 format
            scores.append(evaluation)
            details.append({
                "theme": theme_label,
                "decision": "LEGACY",
                "score": evaluation,
            })

    return {
        "scores": scores,
        "average": float(np.mean(scores)) if scores else 0.0,
        "n_below_threshold": sum(s < GROUNDEDNESS_THRESHOLD for s in scores),
        "n_total": len(scores),
        "details": details,
    }


def _build_comment(result: dict[str, Any], metric_name: str) -> str:
    """Build an enriched comment string from evaluation details.

    Args:
        result: Parsed evaluation result from _parse_evaluation_response.
        metric_name: Either "groundedness" or "coverage".

    Returns:
        Comment string with threshold summary and per-theme decisions.
    """
    threshold_label = "themes below threshold" if metric_name == "groundedness" else "themes not captured"
    summary = f"{result['n_below_threshold']}/{result['n_total']} {threshold_label}"

    details = result.get("details", [])
    if not details or details[0].get("decision") == "LEGACY":
        return summary

    decision_parts = []
    for detail in details:
        decision = detail.get("decision", "?")
        theme = detail.get("theme", "?")
        matched = detail.get("matched_to", "none")
        reasoning = detail.get("reasoning", "")
        if decision == "NO":
            decision_parts.append(f"  {decision}: {theme} (no match) — {reasoning}")
        else:
            decision_parts.append(f"  {decision}: {theme} → {matched} — {reasoning}")

    return f"{summary}\n" + "\n".join(decision_parts)


def _calculate_groundedness_scores(
    generated_themes: list[dict] | dict,
    expected_themes: dict,
    llm: Any,
) -> dict[str, Any]:
    """Calculate groundedness scores using LLM-as-judge.

    Args:
        generated_themes: Generated themes (list of dicts or dict)
        expected_themes: Expected theme framework
        llm: LangChain LLM instance

    Returns:
        Dict with scores list, average, count below threshold, and details
    """
    from utils import read_and_render

    shuffled_generated = _shuffle_themes(generated_themes)
    shuffled_expected = _shuffle_themes(expected_themes)

    response = llm.invoke(
        read_and_render(
            "generation_eval.txt",
            {"topic_list_1": shuffled_generated, "topic_list_2": shuffled_expected},
        )
    )

    return _parse_evaluation_response(response.content)


def _calculate_coverage_scores(
    generated_themes: list[dict] | dict,
    expected_themes: dict,
    llm: Any,
) -> dict[str, Any]:
    """Calculate coverage scores (recall direction) using LLM-as-judge.

    Args:
        generated_themes: Generated themes (list of dicts or dict)
        expected_themes: Expected theme framework
        llm: LangChain LLM instance

    Returns:
        Dict with scores list, average, count below threshold, and details
    """
    from utils import read_and_render

    shuffled_generated = _shuffle_themes(generated_themes)
    shuffled_expected = _shuffle_themes(expected_themes)

    # Reverse direction: expected -> generated
    response = llm.invoke(
        read_and_render(
            "generation_eval.txt",
            {"topic_list_1": shuffled_expected, "topic_list_2": shuffled_generated},
        )
    )

    return _parse_evaluation_response(response.content)


def create_groundedness_evaluator(llm: Any):
    """Factory for theme groundedness evaluator.

    Args:
        llm: LangChain LLM instance for scoring

    Returns:
        Evaluator function compatible with run_experiment()
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        logger.warning("Langfuse not available, using dict fallback")
        Evaluation = dict  # Fallback for local testing

    def groundedness_evaluator(*, output: dict, expected_output: dict, **kwargs) -> Any:
        """Evaluate how well generated themes are grounded in expected themes.

        Args:
            output: Task output containing "themes" key
            expected_output: Expected output containing "themes" key

        Returns:
            Langfuse Evaluation with groundedness score (0-5 scale)
        """
        try:
            result = _calculate_groundedness_scores(
                output.get("themes", []),
                expected_output.get("themes", {}),
                llm,
            )

            comment = _build_comment(result, "groundedness")

            if Evaluation == dict:
                return {
                    "name": "groundedness",
                    "value": round(result["average"], 2),
                    "comment": comment,
                }

            return Evaluation(
                name="groundedness",
                value=round(result["average"], 2),
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Groundedness evaluation failed: {e}")
            if Evaluation == dict:
                return {"name": "groundedness", "value": 0.0, "comment": f"Error: {e}"}
            return Evaluation(name="groundedness", value=0.0, comment=f"Error: {e}")

    return groundedness_evaluator


def create_coverage_evaluator(llm: Any):
    """Factory for theme coverage evaluator (recall direction).

    Args:
        llm: LangChain LLM instance for scoring

    Returns:
        Evaluator function compatible with run_experiment()
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        logger.warning("Langfuse not available, using dict fallback")
        Evaluation = dict

    def coverage_evaluator(*, output: dict, expected_output: dict, **kwargs) -> Any:
        """Evaluate how well expected themes are covered by generated themes.

        Args:
            output: Task output containing "themes" key
            expected_output: Expected output containing "themes" key

        Returns:
            Langfuse Evaluation with coverage score (0-5 scale)
        """
        try:
            result = _calculate_coverage_scores(
                output.get("themes", []),
                expected_output.get("themes", {}),
                llm,
            )

            comment = _build_comment(result, "coverage")

            if Evaluation == dict:
                return {
                    "name": "coverage",
                    "value": round(result["average"], 2),
                    "comment": comment,
                }

            return Evaluation(
                name="coverage",
                value=round(result["average"], 2),
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Coverage evaluation failed: {e}")
            if Evaluation == dict:
                return {"name": "coverage", "value": 0.0, "comment": f"Error: {e}"}
            return Evaluation(name="coverage", value=0.0, comment=f"Error: {e}")

    return coverage_evaluator


def sentiment_accuracy_evaluator(
    *, output: dict, expected_output: dict, **kwargs
) -> Any:
    """Aggregate accuracy evaluator for sentiment analysis.

    Args:
        output: Task output containing "positions" dict mapping response_id to position
        expected_output: Expected output containing "positions" dict

    Returns:
        Langfuse Evaluation with accuracy score (0-1 scale)
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        Evaluation = dict

    try:
        output_positions = output.get("positions", {})
        expected_positions = expected_output.get("positions", {})

        if not expected_positions:
            if Evaluation == dict:
                return {
                    "name": "accuracy",
                    "value": 0.0,
                    "comment": "No expected positions",
                }
            return Evaluation(
                name="accuracy", value=0.0, comment="No expected positions"
            )

        correct = sum(
            1
            for rid, pos in output_positions.items()
            if expected_positions.get(rid) == pos
        )
        total = len(expected_positions)
        accuracy = correct / total if total > 0 else 0.0

        if Evaluation == dict:
            return {
                "name": "accuracy",
                "value": round(accuracy, 3),
                "comment": f"{correct}/{total} correct predictions",
            }

        return Evaluation(
            name="accuracy",
            value=round(accuracy, 3),
            comment=f"{correct}/{total} correct predictions",
        )
    except Exception as e:
        logger.error(f"Sentiment accuracy evaluation failed: {e}")
        if Evaluation == dict:
            return {"name": "accuracy", "value": 0.0, "comment": f"Error: {e}"}
        return Evaluation(name="accuracy", value=0.0, comment=f"Error: {e}")


def mapping_f1_evaluator(*, output: dict, expected_output: dict, **kwargs) -> Any:
    """Multi-label F1 evaluator for theme mapping.

    Args:
        output: Task output containing "labels" dict mapping response_id to label list
        expected_output: Expected output containing "mappings" dict

    Returns:
        Langfuse Evaluation with F1 score (0-1 scale)
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        Evaluation = dict

    try:
        output_labels = output.get("labels", {})
        expected_mappings = expected_output.get("mappings", {})

        if not expected_mappings:
            if Evaluation == dict:
                return {
                    "name": "f1_score",
                    "value": 0.0,
                    "comment": "No expected mappings",
                }
            return Evaluation(
                name="f1_score", value=0.0, comment="No expected mappings"
            )

        # Convert to lists for MultiLabelBinarizer
        response_ids = list(expected_mappings.keys())
        y_true = [expected_mappings.get(rid, []) for rid in response_ids]
        y_pred = [output_labels.get(rid, []) for rid in response_ids]

        # Fit binarizer on all possible labels
        mlb = MultiLabelBinarizer()
        all_labels = set()
        for labels in y_true + y_pred:
            all_labels.update(labels)
        mlb.fit([list(all_labels)])

        # Transform and calculate F1
        y_true_bin = mlb.transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        f1 = metrics.f1_score(y_true_bin, y_pred_bin, average="samples")

        if Evaluation == dict:
            return {
                "name": "f1_score",
                "value": round(f1, 3),
                "comment": f"Evaluated on {len(response_ids)} responses",
            }

        return Evaluation(
            name="f1_score",
            value=round(f1, 3),
            comment=f"Evaluated on {len(response_ids)} responses",
        )
    except Exception as e:
        logger.error(f"Mapping F1 evaluation failed: {e}")
        if Evaluation == dict:
            return {"name": "f1_score", "value": 0.0, "comment": f"Error: {e}"}
        return Evaluation(name="f1_score", value=0.0, comment=f"Error: {e}")


def _calculate_title_specificity(
    themes: list[dict] | dict,
    llm: Any,
) -> dict[str, Any]:
    """Calculate title specificity using LLM-as-judge.

    Args:
        themes: Generated themes (list of dicts or dict).
        llm: LangChain LLM instance.

    Returns:
        Dict with ratio of specific titles and per-theme details.
    """
    from utils import read_and_render

    # Extract just the titles for evaluation
    if isinstance(themes, list):
        titles = [t.get("topic_label", t.get("topic", "")) for t in themes]
    elif isinstance(themes, dict):
        titles = list(themes.keys())
    else:
        titles = []

    if not titles:
        return {"ratio": 0.0, "n_specific": 0, "n_total": 0, "details": []}

    response = llm.invoke(
        read_and_render(
            "title_specificity_eval.txt",
            {"theme_titles": titles},
        )
    )

    parsed = parse_json_markdown(response.content)
    evaluations = parsed.get("evaluations", parsed)

    n_specific = 0
    details = []

    for title, evaluation in evaluations.items():
        decision = evaluation.get("decision", "VAGUE").upper()
        is_specific = decision == "SPECIFIC"
        if is_specific:
            n_specific += 1
        details.append({
            "title": title,
            "decision": decision,
            "reasoning": evaluation.get("reasoning", ""),
        })

    n_total = len(evaluations)
    ratio = n_specific / n_total if n_total > 0 else 0.0

    return {
        "ratio": ratio,
        "n_specific": n_specific,
        "n_total": n_total,
        "details": details,
    }


def create_title_specificity_evaluator(llm: Any):
    """Factory for theme title specificity evaluator.

    Args:
        llm: LangChain LLM instance for evaluation.

    Returns:
        Evaluator function compatible with run_experiment().
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        logger.warning("Langfuse not available, using dict fallback")
        Evaluation = dict

    def specificity_evaluator(*, output: dict, expected_output: dict, **kwargs) -> Any:
        """Evaluate how specific the generated theme titles are.

        Returns:
            Langfuse Evaluation with specificity ratio (0-1 scale).
        """
        try:
            result = _calculate_title_specificity(
                output.get("themes", []),
                llm,
            )

            comment = f"{result['n_specific']}/{result['n_total']} titles specific"
            vague_titles = [d["title"] for d in result.get("details", []) if d["decision"] == "VAGUE"]
            if vague_titles:
                comment += f"\nVague: {', '.join(vague_titles)}"

            if Evaluation == dict:
                return {
                    "name": "specificity",
                    "value": round(result["ratio"], 2),
                    "comment": comment,
                }

            return Evaluation(
                name="specificity",
                value=round(result["ratio"], 2),
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Specificity evaluation failed: {e}")
            if Evaluation == dict:
                return {"name": "specificity", "value": 0.0, "comment": f"Error: {e}"}
            return Evaluation(name="specificity", value=0.0, comment=f"Error: {e}")

    return specificity_evaluator


def _calculate_condensation_scores(
    condensed_themes: list[dict] | dict,
    original_themes: list[dict] | dict,
    llm: Any,
) -> dict[str, Any]:
    """Calculate condensation quality scores using LLM-as-judge.

    Args:
        condensed_themes: Condensed themes (list of dicts or dict).
        original_themes: Original themes before condensation.
        llm: LangChain LLM instance.

    Returns:
        Dict with compression_quality, information_retention, and reasoning.
    """
    from utils import read_and_render

    response = llm.invoke(
        read_and_render(
            "condensation_eval.txt",
            {
                "original_topics": original_themes,
                "condensed_topics": condensed_themes,
            },
        )
    )

    parsed = parse_json_markdown(response.content)

    return {
        "compression_quality": parsed.get("compression_quality", 0),
        "compression_quality_reasoning": parsed.get("compression_quality_reasoning", ""),
        "information_retention": parsed.get("information_retention", 0),
        "information_retention_reasoning": parsed.get("information_retention_reasoning", ""),
    }


def create_condensation_quality_evaluator(llm: Any):
    """Factory for condensation quality evaluator (compression + information retention).

    Args:
        llm: LangChain LLM instance for scoring.

    Returns:
        Evaluator function that returns a list of Evaluation objects.
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        logger.warning("Langfuse not available, using dict fallback")
        Evaluation = dict

    def condensation_evaluator(*, output: dict, expected_output: dict, **kwargs) -> list:
        """Evaluate condensation quality on compression and information retention.

        Args:
            output: Task output containing "themes" key (condensed themes).
            expected_output: Expected output containing "themes" key (original themes).

        Returns:
            List of Evaluation objects (one per metric).
        """
        try:
            result = _calculate_condensation_scores(
                output.get("themes", []),
                expected_output.get("themes", []),
                llm,
            )

            evaluations = []
            for metric in ("compression_quality", "information_retention"):
                score = result.get(metric, 0)
                reasoning = result.get(f"{metric}_reasoning", "")

                if Evaluation == dict:
                    evaluations.append({
                        "name": metric,
                        "value": round(float(score), 2),
                        "comment": reasoning,
                    })
                else:
                    evaluations.append(Evaluation(
                        name=metric,
                        value=round(float(score), 2),
                        comment=reasoning,
                    ))

            return evaluations

        except Exception as e:
            logger.error(f"Condensation quality evaluation failed: {e}")
            fallback = []
            for metric in ("compression_quality", "information_retention"):
                if Evaluation == dict:
                    fallback.append({"name": metric, "value": 0.0, "comment": f"Error: {e}"})
                else:
                    fallback.append(Evaluation(name=metric, value=0.0, comment=f"Error: {e}"))
            return fallback

    return condensation_evaluator


REFINEMENT_METRICS = (
    "information_retention",
    "response_references",
    "distinctiveness",
    "fluency",
)


def _calculate_refinement_scores(
    refined_themes: list[dict] | dict,
    original_themes: list[dict] | dict,
    llm: Any,
) -> dict[str, Any]:
    """Calculate refinement quality scores using LLM-as-judge.

    Args:
        refined_themes: Refined themes (list of dicts or dict).
        original_themes: Original themes before refinement.
        llm: LangChain LLM instance.

    Returns:
        Dict with four metric scores and reasoning.
    """
    from utils import read_and_render

    response = llm.invoke(
        read_and_render(
            "refinement_eval.txt",
            {
                "original_topics": original_themes,
                "new_topics": refined_themes,
            },
        )
    )

    parsed = parse_json_markdown(response.content)

    return {
        metric: parsed.get(metric, 0)
        for metric in REFINEMENT_METRICS
    } | {
        f"{metric}_reasoning": parsed.get(f"{metric}_reasoning", "")
        for metric in REFINEMENT_METRICS
    }


def create_refinement_quality_evaluator(llm: Any):
    """Factory for refinement quality evaluator (4 dimensions).

    Args:
        llm: LangChain LLM instance for scoring.

    Returns:
        Evaluator function that returns a list of Evaluation objects.
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        logger.warning("Langfuse not available, using dict fallback")
        Evaluation = dict

    def refinement_evaluator(*, output: dict, expected_output: dict, **kwargs) -> list:
        """Evaluate refinement quality on four dimensions.

        Args:
            output: Task output containing "themes" key (refined themes).
            expected_output: Expected output containing "themes" key (original themes).

        Returns:
            List of Evaluation objects (one per metric).
        """
        try:
            result = _calculate_refinement_scores(
                output.get("themes", []),
                expected_output.get("themes", []),
                llm,
            )

            evaluations = []
            for metric in REFINEMENT_METRICS:
                score = result.get(metric, 0)
                reasoning = result.get(f"{metric}_reasoning", "")

                if Evaluation == dict:
                    evaluations.append({
                        "name": metric,
                        "value": round(float(score), 2),
                        "comment": reasoning,
                    })
                else:
                    evaluations.append(Evaluation(
                        name=metric,
                        value=round(float(score), 2),
                        comment=reasoning,
                    ))

            return evaluations

        except Exception as e:
            logger.error(f"Refinement quality evaluation failed: {e}")
            fallback = []
            for metric in REFINEMENT_METRICS:
                if Evaluation == dict:
                    fallback.append({"name": metric, "value": 0.0, "comment": f"Error: {e}"})
                else:
                    fallback.append(Evaluation(name=metric, value=0.0, comment=f"Error: {e}"))
            return fallback

    return refinement_evaluator


@lru_cache(maxsize=1)
def _get_sentence_model():
    """Load and cache the sentence-transformers model (loaded once per process)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def calculate_redundancy_score(themes: list[dict] | dict, threshold: float = 0.85) -> dict[str, Any]:
    """Calculate semantic redundancy between theme titles using sentence embeddings.

    Uses sentence-transformers (all-MiniLM-L6-v2) to compute pairwise cosine
    similarity and flags pairs above the threshold.

    Args:
        themes: Generated themes (list of dicts or dict).
        threshold: Cosine similarity threshold above which pairs are flagged.

    Returns:
        Dict with redundancy ratio, flagged pairs, and details.
    """
    try:
        _get_sentence_model()
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping redundancy check")
        return {"ratio": 0.0, "n_redundant_pairs": 0, "n_total_pairs": 0, "flagged_pairs": []}

    # Extract titles
    if isinstance(themes, list):
        titles = [t.get("topic_label", t.get("topic", "")) for t in themes]
    elif isinstance(themes, dict):
        titles = list(themes.keys())
    else:
        titles = []

    if len(titles) < 2:
        return {"ratio": 0.0, "n_redundant_pairs": 0, "n_total_pairs": 0, "flagged_pairs": []}

    model = _get_sentence_model()
    embeddings = model.encode(titles, convert_to_tensor=True)

    # Compute pairwise cosine similarity
    from sentence_transformers.util import cos_sim
    similarity_matrix = cos_sim(embeddings, embeddings)

    flagged_pairs = []
    n_total_pairs = 0

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            n_total_pairs += 1
            sim = float(similarity_matrix[i][j])
            if sim >= threshold:
                flagged_pairs.append({
                    "theme_a": titles[i],
                    "theme_b": titles[j],
                    "similarity": round(sim, 3),
                })

    ratio = len(flagged_pairs) / n_total_pairs if n_total_pairs > 0 else 0.0

    return {
        "ratio": ratio,
        "n_redundant_pairs": len(flagged_pairs),
        "n_total_pairs": n_total_pairs,
        "flagged_pairs": flagged_pairs,
    }


def create_redundancy_evaluator():
    """Factory for semantic redundancy evaluator (no LLM needed).

    Returns:
        Evaluator function compatible with run_experiment().
    """
    try:
        from langfuse import Evaluation
    except ImportError:
        logger.warning("Langfuse not available, using dict fallback")
        Evaluation = dict

    def redundancy_evaluator(*, output: dict, expected_output: dict, **kwargs) -> Any:
        """Evaluate semantic redundancy among generated themes.

        Returns:
            Langfuse Evaluation with redundancy ratio (0-1 scale, lower is better).
        """
        try:
            result = calculate_redundancy_score(output.get("themes", []))

            comment = f"{result['n_redundant_pairs']}/{result['n_total_pairs']} pairs above threshold"
            if result["flagged_pairs"]:
                pair_strs = [f"  {p['theme_a']} ↔ {p['theme_b']} ({p['similarity']})" for p in result["flagged_pairs"]]
                comment += "\n" + "\n".join(pair_strs)

            if Evaluation == dict:
                return {
                    "name": "redundancy",
                    "value": round(result["ratio"], 2),
                    "comment": comment,
                }

            return Evaluation(
                name="redundancy",
                value=round(result["ratio"], 2),
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Redundancy evaluation failed: {e}")
            if Evaluation == dict:
                return {"name": "redundancy", "value": 0.0, "comment": f"Error: {e}"}
            return Evaluation(name="redundancy", value=0.0, comment=f"Error: {e}")

    return redundancy_evaluator
