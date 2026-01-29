"""Langfuse-compatible evaluators for ThemeFinder evaluations.

Provides evaluator functions that can be used with Langfuse's run_experiment() API.
Each evaluator returns a Langfuse Evaluation object with name, value, and optional comment.
"""

import json
import logging
from typing import Any

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger("themefinder.evals.evaluators")

# Minimum score (0-5) to consider a topic well-grounded or captured
GROUNDEDNESS_THRESHOLD = 3


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
        Dict with scores list, average, and count below threshold
    """
    from utils import read_and_render

    response = llm.invoke(
        read_and_render(
            "generation_eval.txt",
            {"topic_list_1": generated_themes, "topic_list_2": expected_themes},
        )
    )
    scores = list(json.loads(response.content).values())

    return {
        "scores": scores,
        "average": float(np.mean(scores)) if scores else 0.0,
        "n_below_threshold": sum(s < GROUNDEDNESS_THRESHOLD for s in scores),
        "n_total": len(scores),
    }


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
        Dict with scores list, average, and count below threshold
    """
    from utils import read_and_render

    # Reverse direction: expected -> generated
    response = llm.invoke(
        read_and_render(
            "generation_eval.txt",
            {"topic_list_1": expected_themes, "topic_list_2": generated_themes},
        )
    )
    scores = list(json.loads(response.content).values())

    return {
        "scores": scores,
        "average": float(np.mean(scores)) if scores else 0.0,
        "n_below_threshold": sum(s < GROUNDEDNESS_THRESHOLD for s in scores),
        "n_total": len(scores),
    }


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
            scores = _calculate_groundedness_scores(
                output.get("themes", []),
                expected_output.get("themes", {}),
                llm,
            )

            if Evaluation == dict:
                return {
                    "name": "groundedness",
                    "value": round(scores["average"], 2),
                    "comment": f"{scores['n_below_threshold']}/{scores['n_total']} themes below threshold",
                }

            return Evaluation(
                name="groundedness",
                value=round(scores["average"], 2),
                comment=f"{scores['n_below_threshold']}/{scores['n_total']} themes below threshold",
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
            scores = _calculate_coverage_scores(
                output.get("themes", []),
                expected_output.get("themes", {}),
                llm,
            )

            if Evaluation == dict:
                return {
                    "name": "coverage",
                    "value": round(scores["average"], 2),
                    "comment": f"{scores['n_below_threshold']}/{scores['n_total']} themes not captured",
                }

            return Evaluation(
                name="coverage",
                value=round(scores["average"], 2),
                comment=f"{scores['n_below_threshold']}/{scores['n_total']} themes not captured",
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
