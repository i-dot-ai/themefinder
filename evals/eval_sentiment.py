"""Sentiment analysis evaluation with Langfuse dataset and experiment support.

Evaluates the sentiment analysis task that classifies responses as AGREE/DISAGREE.
"""

import argparse
import asyncio
import os
from datetime import datetime

import nest_asyncio

# Allow nested asyncio.run() calls for async evaluation context
nest_asyncio.apply()

import dotenv
import langfuse_utils
import pandas as pd
from datasets import DatasetConfig, load_local_data
from evaluators import sentiment_accuracy_evaluator
from langchain_openai import AzureChatOpenAI
from metrics import calculate_sentiment_metrics

from themefinder import sentiment_analysis


async def evaluate_sentiment(
    dataset: str = "gambling_XS",
    llm: AzureChatOpenAI | None = None,
    langfuse_ctx: langfuse_utils.LangfuseContext | None = None,
) -> dict:
    """Run sentiment evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")
        llm: Optional pre-configured LLM instance (for benchmark runs)
        langfuse_ctx: Optional pre-configured Langfuse context (for benchmark runs)

    Returns:
        Dict containing evaluation scores
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="sentiment")

    # Use provided context or create new one
    owns_context = langfuse_ctx is None
    if langfuse_ctx is None:
        session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        langfuse_ctx = langfuse_utils.get_langfuse_context(
            session_id=session_id,
            eval_type="sentiment",
            metadata={"dataset": dataset},
            tags=[dataset],
        )

    # Use provided LLM or create new one
    if llm is None:
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

    # Only flush if we created the context
    if owns_context:
        langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(ctx, config: DatasetConfig, llm) -> dict:
    """Run evaluation with manual dataset iteration for proper trace control.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance

    Returns:
        Dict containing evaluation scores
    """
    try:
        dataset = ctx.client.get_dataset(config.name)
    except Exception as e:
        print(
            f"Dataset {config.name} not found in Langfuse, falling back to local: {e}"
        )
        return await _run_local_fallback(config, llm)

    all_scores = {}
    items = list(dataset.items)

    for item in items:
        # Create trace for this item with full metadata
        with langfuse_utils.dataset_item_trace(ctx, item, ctx.session_id) as (
            trace,
            trace_id,
        ):
            # Extract input
            responses_df = pd.DataFrame(item.input["responses"])
            question = item.input["question"]

            # Run sentiment analysis
            result_df, _ = await sentiment_analysis(
                responses_df=responses_df[["response_id", "response"]],
                llm=llm,
                question=question,
            )

            # Build positions map
            positions = dict(
                zip(
                    result_df["response_id"].astype(str),
                    result_df["position"].map(
                        {"DISAGREEMENT": "DISAGREE", "AGREEMENT": "AGREE", "UNCLEAR": "UNCLEAR"}
                    ),
                )
            )
            output = {"positions": positions}

            # Update trace with output
            if trace:
                trace.update(output=output)

            # Run evaluator and attach score
            eval_result = sentiment_accuracy_evaluator(
                output=output,
                expected_output=item.expected_output,
            )

            if trace_id and ctx.client:
                ctx.client.create_score(
                    trace_id=trace_id,
                    name=eval_result.name,
                    value=eval_result.value,
                    data_type="NUMERIC",
                )

            # Collect for return
            item_key = item.metadata.get("question_part", item.id)
            all_scores[f"{item_key}_accuracy"] = eval_result.value

            # Include pipeline output for disk persistence
            all_scores[f"{item_key}_output"] = output

    print(f"Sentiment Eval Results: {ctx.session_id}")
    return all_scores


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
