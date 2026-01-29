"""Theme mapping evaluation with Langfuse dataset and experiment support.

Evaluates the theme mapping task that assigns themes to responses.
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
from themefinder import theme_mapping

import langfuse_utils
from datasets import DatasetConfig, load_local_data
from evaluators import mapping_f1_evaluator
from metrics import calculate_mapping_metrics


async def evaluate_mapping(
    dataset: str = "gambling_XS",
    question_num: int | None = None,
) -> dict:
    """Run mapping evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")
        question_num: Optional specific question number (1-3) to evaluate

    Returns:
        Dict containing evaluation scores
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="mapping")
    session_id = (
        f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    langfuse_ctx = langfuse_utils.get_langfuse_context(
        session_id=session_id,
        eval_type="mapping",
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
        result = await _run_with_langfuse(langfuse_ctx, config, llm, question_num)
    else:
        result = await _run_local_fallback(config, llm, question_num)

    langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(
    ctx, config: DatasetConfig, llm, question_num: int | None
) -> dict:
    """Run experiment using Langfuse dataset.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance
        question_num: Optional specific question to evaluate

    Returns:
        Experiment result dict
    """
    try:
        dataset = ctx.client.get_dataset(config.name)
    except Exception as e:
        print(
            f"Dataset {config.name} not found in Langfuse, falling back to local: {e}"
        )
        return await _run_local_fallback(config, llm, question_num)

    def task(*, item, **kwargs) -> dict:
        """Task function for experiment runner."""
        responses_df = pd.DataFrame(item.input["responses"])
        question = item.input["question"]
        topics_df = pd.DataFrame(item.input["topics"])
        topics_df = topics_df.rename(columns={"topic_id": "topic_id", "topic": "topic"})

        # Run theme mapping
        result_df = asyncio.run(
            theme_mapping(
                responses_df=responses_df[["response_id", "response"]],
                llm=llm,
                question=question,
                refined_themes_df=topics_df,
            )
        )

        # Handle tuple return
        if isinstance(result_df, tuple):
            result_df = result_df[0]

        # Build labels map
        labels = dict(
            zip(
                result_df["response_id"].astype(str),
                result_df["labels"].tolist(),
            )
        )

        return {"labels": labels}

    with langfuse_utils.trace_context(ctx):
        result = dataset.run_experiment(
            name=ctx.session_id,
            task=task,
            evaluators=[mapping_f1_evaluator],
            max_concurrency=1,
        )

    print(f"Mapping Eval Results (Langfuse experiment): {ctx.session_id}")
    return {"experiment_id": ctx.session_id, "result": result}


async def _run_local_fallback(
    config: DatasetConfig, llm, question_num: int | None
) -> dict:
    """Run evaluation without Langfuse (local development).

    Args:
        config: DatasetConfig
        llm: LangChain LLM instance
        question_num: Optional specific question to evaluate

    Returns:
        Dict containing evaluation scores
    """
    data_items = load_local_data(config)

    # Filter to specific question if requested
    if question_num is not None:
        data_items = [
            item
            for item in data_items
            if f"part_{question_num}"
            in item.get("metadata", {}).get("question_part", "")
        ]

    all_scores = {}

    for item in data_items:
        question_part = item.get("metadata", {}).get("question_part", "unknown")
        responses_df = pd.DataFrame(item["input"]["responses"])
        question = item["input"]["question"]
        topics_df = pd.DataFrame(item["input"]["topics"])
        expected_mappings = item["expected_output"]["mappings"]

        result, _ = await theme_mapping(
            responses_df=responses_df[["response_id", "response"]],
            llm=llm,
            question=question,
            refined_themes_df=topics_df[["topic_id", "topic"]],
        )

        # Merge for comparison
        responses_df["topics"] = (
            responses_df["response_id"].astype(str).map(expected_mappings)
        )
        responses_df = responses_df.merge(
            result[["response_id", "labels"]], "inner", on="response_id"
        )

        mapping_metrics = calculate_mapping_metrics(
            df=responses_df, column_one="topics", column_two="labels"
        )
        print(f"Theme Mapping ({question_part}): \n {mapping_metrics}")

        # Collect scores with question prefix
        for key, value in mapping_metrics.items():
            if isinstance(value, (int, float)):
                all_scores[f"{question_part}_{key}"] = value

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run theme mapping evaluation")
    parser.add_argument(
        "--dataset",
        default="gambling_XS",
        help="Dataset identifier (e.g., gambling_XS)",
    )
    parser.add_argument(
        "--question", type=int, default=None, help="Specific question number (1-3)"
    )
    args = parser.parse_args()

    asyncio.run(evaluate_mapping(dataset=args.dataset, question_num=args.question))
