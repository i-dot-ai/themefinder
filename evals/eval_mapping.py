"""Theme mapping evaluation with Langfuse dataset and experiment support.

Evaluates the theme mapping task that assigns themes to responses.
"""

import argparse
import asyncio
import os
from datetime import datetime

import dotenv
import langfuse_utils
import pandas as pd
from datasets import DatasetConfig, load_local_data
from evaluators import mapping_f1_evaluator
from langchain_openai import AzureChatOpenAI
from metrics import calculate_mapping_metrics

from themefinder import theme_mapping


async def evaluate_mapping(
    dataset: str = "gambling_XS",
    question_num: int | None = None,
    llm: AzureChatOpenAI | None = None,
    langfuse_ctx: langfuse_utils.LangfuseContext | None = None,
) -> dict:
    """Run mapping evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")
        question_num: Optional specific question number (1-3) to evaluate
        llm: Optional pre-configured LLM instance (for benchmark runs)
        langfuse_ctx: Optional pre-configured Langfuse context (for benchmark runs)

    Returns:
        Dict containing evaluation scores
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="mapping")

    # Use provided context or create new one
    owns_context = langfuse_ctx is None
    if langfuse_ctx is None:
        session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        langfuse_ctx = langfuse_utils.get_langfuse_context(
            session_id=session_id,
            eval_type="mapping",
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
        result = await _run_with_langfuse(langfuse_ctx, config, llm, question_num)
    else:
        result = await _run_local_fallback(config, llm, question_num)

    # Only flush if we created the context
    if owns_context:
        langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(
    ctx, config: DatasetConfig, llm, question_num: int | None
) -> dict:
    """Run evaluation with manual dataset iteration for proper trace control.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance
        question_num: Optional specific question to evaluate

    Returns:
        Dict containing evaluation scores
    """
    try:
        dataset = ctx.client.get_dataset(config.name)
    except Exception as e:
        print(
            f"Dataset {config.name} not found in Langfuse, falling back to local: {e}"
        )
        return await _run_local_fallback(config, llm, question_num)

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
            topics_df = pd.DataFrame(item.input["topics"])
            topics_df = topics_df.rename(
                columns={"topic_id": "topic_id", "topic": "topic"}
            )

            # Run theme mapping
            result_df, unprocessable_df = await theme_mapping(
                responses_df=responses_df[["response_id", "response"]],
                llm=llm,
                question=question,
                refined_themes_df=topics_df,
            )
            if not unprocessable_df.empty:
                print(f"  Warning: {len(unprocessable_df)} responses could not be processed")

            # Build labels map
            labels = dict(
                zip(
                    result_df["response_id"].astype(str),
                    result_df["labels"].tolist(),
                )
            )
            output = {"labels": labels}

            # Update trace with output
            if trace:
                trace.update(output=output)

            # Run evaluator and attach score
            eval_result = mapping_f1_evaluator(
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
            all_scores[f"{item_key}_f1"] = eval_result.value

            # Include pipeline output for disk persistence
            all_scores[f"{item_key}_output"] = output

    print(f"Mapping Eval Results: {ctx.session_id}")
    return all_scores


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

        result, unprocessable_df = await theme_mapping(
            responses_df=responses_df[["response_id", "response"]],
            llm=llm,
            question=question,
            refined_themes_df=topics_df[["topic_id", "topic"]],
        )
        if not unprocessable_df.empty:
            print(f"  Warning: {len(unprocessable_df)} responses could not be processed")

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
    import nest_asyncio

    nest_asyncio.apply()

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
