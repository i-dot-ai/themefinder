"""Sentiment analysis evaluation with Langfuse dataset and experiment support.

Evaluates the sentiment analysis task that classifies responses as AGREE/DISAGREE.
"""

import argparse
import asyncio
import os
from datetime import datetime

import nest_asyncio

# Allow nested asyncio.run() calls for Langfuse experiment runner
nest_asyncio.apply()

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI
from themefinder import sentiment_analysis

import langfuse_utils
from datasets import DatasetConfig, load_local_data
from evaluators import sentiment_accuracy_evaluator
from metrics import calculate_sentiment_metrics


async def evaluate_sentiment(
    dataset: str = "gambling_XS",
) -> dict:
    """Run sentiment evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")

    Returns:
        Dict containing evaluation scores
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="sentiment")
    session_id = (
        f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    langfuse_ctx = langfuse_utils.get_langfuse_context(
        session_id=session_id,
        eval_type="sentiment",
        metadata={"dataset": dataset},
        tags=[dataset],
    )

    callbacks = [langfuse_ctx.handler] if langfuse_ctx.handler else []

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
        callbacks=callbacks,
    )

    # Branch: Langfuse dataset vs local fallback
    if langfuse_ctx.is_enabled:
        result = await _run_with_langfuse(langfuse_ctx, config, llm)
    else:
        result = await _run_local_fallback(config, llm)

    langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(ctx, config: DatasetConfig, llm) -> dict:
    """Run experiment using Langfuse dataset.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance

    Returns:
        Experiment result dict
    """
    try:
        dataset = ctx.client.get_dataset(config.name)
    except Exception as e:
        print(
            f"Dataset {config.name} not found in Langfuse, falling back to local: {e}"
        )
        return await _run_local_fallback(config, llm)

    def task(*, item, **kwargs) -> dict:
        """Task function for experiment runner."""
        responses_df = pd.DataFrame(item.input["responses"])
        question = item.input["question"]

        # Run sentiment analysis
        result_df = asyncio.run(
            sentiment_analysis(
                responses_df=responses_df[["response_id", "response"]],
                llm=llm,
                question=question,
            )
        )

        # Handle tuple return
        if isinstance(result_df, tuple):
            result_df = result_df[0]

        # Build positions map
        positions = dict(
            zip(
                result_df["response_id"].astype(str),
                result_df["position"].map(
                    {"DISAGREEMENT": "DISAGREE", "AGREEMENT": "AGREE"}
                ),
            )
        )

        return {"positions": positions}

    with langfuse_utils.trace_context(ctx):
        result = dataset.run_experiment(
            name=ctx.session_id,
            task=task,
            evaluators=[sentiment_accuracy_evaluator],
            max_concurrency=1,
        )

    print(f"Sentiment Eval Results (Langfuse experiment): {ctx.session_id}")
    return {"experiment_id": ctx.session_id, "result": result}


async def _run_local_fallback(config: DatasetConfig, llm) -> dict:
    """Run evaluation without Langfuse (local development).

    Args:
        config: DatasetConfig
        llm: LangChain LLM instance

    Returns:
        Dict containing evaluation scores
    """
    data_items = load_local_data(config)
    all_scores = {}

    for item in data_items:
        question_part = item.get("metadata", {}).get("question_part", "unknown")
        responses_df = pd.DataFrame(item["input"]["responses"])
        question = item["input"]["question"]
        expected_positions = item["expected_output"]["positions"]

        result, _ = await sentiment_analysis(
            responses_df=responses_df[["response_id", "response"]],
            llm=llm,
            question=question,
        )

        # Build comparison DataFrame
        result = result.rename(columns={"position": "ai_position"})
        result["ai_position"] = result["ai_position"].map(
            {"DISAGREEMENT": "DISAGREE", "AGREEMENT": "AGREE"}
        )
        result["supervisor_position"] = (
            result["response_id"].astype(str).map(expected_positions)
        )

        eval_scores = calculate_sentiment_metrics(result)
        print(f"Sentiment Eval ({question_part}): accuracy={eval_scores['accuracy']}")

        # Collect scores with question prefix
        for key, value in eval_scores.items():
            all_scores[f"{question_part}_{key}"] = value

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment analysis evaluation")
    parser.add_argument(
        "--dataset",
        default="gambling_XS",
        help="Dataset identifier (e.g., gambling_XS)",
    )
    args = parser.parse_args()

    asyncio.run(evaluate_sentiment(dataset=args.dataset))
