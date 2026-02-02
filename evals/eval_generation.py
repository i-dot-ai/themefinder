"""Theme generation evaluation with Langfuse dataset and experiment support.

Evaluates the full theme generation pipeline (generation -> condensation -> refinement)
against a ground truth theme framework.
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
from evaluators import create_coverage_evaluator, create_groundedness_evaluator
from langchain_openai import AzureChatOpenAI
from metrics import calculate_generation_metrics

from themefinder import theme_condensation, theme_generation, theme_refinement


async def evaluate_generation(
    dataset: str = "gambling_XS",
    llm: AzureChatOpenAI | None = None,
    langfuse_ctx: langfuse_utils.LangfuseContext | None = None,
) -> dict:
    """Run generation evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")
        llm: Optional pre-configured LLM instance (for benchmark runs)
        langfuse_ctx: Optional pre-configured Langfuse context (for benchmark runs)

    Returns:
        Dict containing evaluation scores
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="generation")

    # Use provided context or create new one
    owns_context = langfuse_ctx is None
    if langfuse_ctx is None:
        session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        langfuse_ctx = langfuse_utils.get_langfuse_context(
            session_id=session_id,
            eval_type="generation",
            metadata={"dataset": dataset},
            tags=[dataset],
        )

    callbacks = [langfuse_ctx.handler] if langfuse_ctx.handler else []

    # Use provided LLM or create new one
    if llm is None:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("DEPLOYMENT_NAME"),
            temperature=0,
            callbacks=callbacks,
        )

    # Branch: Langfuse dataset vs local fallback
    if langfuse_ctx.is_enabled:
        result = await _run_with_langfuse(langfuse_ctx, config, llm, callbacks)
    else:
        result = await _run_local_fallback(config, llm, callbacks)

    # Only flush if we created the context
    if owns_context:
        langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(ctx, config: DatasetConfig, llm, callbacks: list) -> dict:
    """Run evaluation with manual dataset iteration for proper trace control.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance
        callbacks: LangChain callbacks list

    Returns:
        Dict containing evaluation scores
    """
    try:
        dataset = ctx.client.get_dataset(config.name)
    except Exception as e:
        print(
            f"Dataset {config.name} not found in Langfuse, falling back to local: {e}"
        )
        return await _run_local_fallback(config, llm, callbacks)

    # Create evaluator functions
    groundedness_evaluator = create_groundedness_evaluator(llm)
    coverage_evaluator = create_coverage_evaluator(llm)

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

            # Run full pipeline
            themes_df, _ = await theme_generation(
                responses_df=responses_df,
                llm=llm,
                question=question,
            )
            condensed_df, _ = await theme_condensation(
                themes_df,
                llm=llm,
                question=question,
            )
            refined_df, _ = await theme_refinement(
                condensed_df,
                llm=llm,
                question=question,
            )

            output = {"themes": refined_df.to_dict(orient="records")}

            # Update trace with output
            if trace:
                trace.update(output=output)

            # Run evaluators and attach scores
            for evaluator in [groundedness_evaluator, coverage_evaluator]:
                eval_result = evaluator(
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
                all_scores[f"{item_key}_{eval_result.name}"] = eval_result.value

    print(f"Theme Generation Eval Results: {ctx.session_id}")
    return all_scores


async def _run_local_fallback(config: DatasetConfig, llm, callbacks: list) -> dict:
    """Run evaluation without Langfuse (local development).

    Args:
        config: DatasetConfig
        llm: LangChain LLM instance
        callbacks: LangChain callbacks list

    Returns:
        Dict containing evaluation scores
    """
    data_items = load_local_data(config)
    all_scores = {}

    for item in data_items:
        question_part = item.get("metadata", {}).get("question_part", "unknown")
        responses_df = pd.DataFrame(item["input"]["responses"])
        question = item["input"]["question"]
        theme_framework = item["expected_output"]["themes"]

        themes_df, _ = await theme_generation(
            responses_df=responses_df,
            llm=llm,
            question=question,
        )
        condensed_df, _ = await theme_condensation(
            themes_df,
            llm=llm,
            question=question,
        )
        refined_df, _ = await theme_refinement(
            condensed_df,
            llm=llm,
            question=question,
        )

        eval_scores = calculate_generation_metrics(
            refined_df, theme_framework, callbacks=callbacks
        )
        print(f"Theme Generation ({question_part}): \n {eval_scores}")

        # Collect scores with question prefix
        for key, value in eval_scores.items():
            if isinstance(value, (int, float)):
                all_scores[f"{question_part}_{key}"] = value

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run theme generation evaluation")
    parser.add_argument(
        "--dataset",
        default="gambling_XS",
        help="Dataset identifier (e.g., gambling_XS)",
    )
    args = parser.parse_args()

    asyncio.run(evaluate_generation(dataset=args.dataset))
