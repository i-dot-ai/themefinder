"""Theme refinement evaluation with Langfuse dataset and experiment support.

Evaluates the theme refinement task that improves theme labels and descriptions.
"""

import argparse
import asyncio
import os
from datetime import datetime

import dotenv
import langfuse_utils
import pandas as pd
from datasets import DatasetConfig, load_local_data
from evaluators import create_refinement_quality_evaluator
from langchain_openai import AzureChatOpenAI

from themefinder import theme_refinement


async def evaluate_refinement(
    dataset: str = "gambling_XS",
    llm: AzureChatOpenAI | None = None,
    langfuse_ctx: langfuse_utils.LangfuseContext | None = None,
    judge_llm: AzureChatOpenAI | None = None,
) -> dict:
    """Run refinement evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")
        llm: Optional pre-configured LLM instance (for benchmark runs)
        langfuse_ctx: Optional pre-configured Langfuse context (for benchmark runs)

    Returns:
        Dict containing evaluation results
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="refinement")

    # Use provided context or create new one
    owns_context = langfuse_ctx is None
    if langfuse_ctx is None:
        session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        langfuse_ctx = langfuse_utils.get_langfuse_context(
            session_id=session_id,
            eval_type="refinement",
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

    # Select judge LLM (falls back to task LLM if not provided)
    eval_llm = judge_llm or llm

    # Branch: Langfuse dataset vs local fallback
    if langfuse_ctx.is_enabled:
        result = await _run_with_langfuse(
            langfuse_ctx, config, llm, eval_llm, callbacks
        )
    else:
        result = await _run_local_fallback(config, llm, eval_llm)

    # Only flush if we created the context
    if owns_context:
        langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(
    ctx, config: DatasetConfig, llm, eval_llm, callbacks: list
) -> dict:
    """Run evaluation with manual dataset iteration for proper trace control.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance (task model)
        eval_llm: LangChain LLM instance (judge model, may be same as llm)
        callbacks: LangChain callbacks list

    Returns:
        Dict containing evaluation results
    """
    try:
        dataset = ctx.client.get_dataset(config.name)
    except Exception as e:
        print(
            f"Dataset {config.name} not found in Langfuse, falling back to local: {e}"
        )
        return await _run_local_fallback(config, llm, eval_llm)

    refinement_evaluator = create_refinement_quality_evaluator(eval_llm)

    all_results = {}
    items = list(dataset.items)

    for item in items:
        # Create trace for this item with full metadata
        with langfuse_utils.dataset_item_trace(ctx, item, ctx.session_id) as (
            trace,
            trace_id,
        ):
            # Extract input
            themes_df = pd.DataFrame(item.input["themes"])
            question = item.input.get("question", "")
            original_records = themes_df.to_dict(orient="records")

            # Run refinement
            refined_df, _ = await theme_refinement(
                themes_df,
                llm=llm,
                question=question,
            )

            refined_records = refined_df.to_dict(orient="records")
            output = {"refined_themes": refined_records}

            # Update trace with output
            if trace:
                trace.update(output=output)

            # LLM-as-judge: 4-dimension quality evaluation
            quality_results = refinement_evaluator(
                output={"themes": refined_records},
                expected_output={"themes": original_records},
            )

            # Record each evaluation as a Langfuse score
            for evaluation in quality_results:
                score_name = (
                    evaluation.get("name", "")
                    if isinstance(evaluation, dict)
                    else evaluation.name
                )
                score_value = (
                    evaluation.get("value", 0.0)
                    if isinstance(evaluation, dict)
                    else evaluation.value
                )
                score_comment = (
                    evaluation.get("comment", "")
                    if isinstance(evaluation, dict)
                    else evaluation.comment
                )

                if trace_id and ctx.client:
                    ctx.client.create_score(
                        trace_id=trace_id,
                        name=score_name,
                        value=score_value,
                        data_type="NUMERIC",
                        comment=score_comment,
                    )

            # Collect for return
            item_key = item.metadata.get("question_part", item.id)
            all_results[f"{item_key}_output"] = output

            # Add quality scores to results (numeric — will go to CSV columns)
            for evaluation in quality_results:
                name = (
                    evaluation.get("name", "")
                    if isinstance(evaluation, dict)
                    else evaluation.name
                )
                value = (
                    evaluation.get("value", 0.0)
                    if isinstance(evaluation, dict)
                    else evaluation.value
                )
                all_results[f"{item_key}_{name}"] = value

    print(f"Refinement Eval Results: {ctx.session_id}")
    return all_results


async def _run_local_fallback(config: DatasetConfig, llm, eval_llm) -> dict:
    """Run evaluation without Langfuse (local development).

    Args:
        config: DatasetConfig
        llm: LangChain LLM instance (task model)
        eval_llm: LangChain LLM instance (judge model)

    Returns:
        Dict containing evaluation scores
    """
    data_items = load_local_data(config)
    refinement_evaluator = create_refinement_quality_evaluator(eval_llm)
    all_results = {}

    for item in data_items:
        question_part = item.get("metadata", {}).get("question_part", "unknown")
        themes_df = pd.DataFrame(item["input"]["themes"])
        question = item["input"].get("question", "")
        original_records = themes_df.to_dict(orient="records")

        with langfuse_utils.trace_context(
            langfuse_utils.LangfuseContext(client=None, handler=None)
        ):
            refined_df, _ = await theme_refinement(
                themes_df,
                llm=llm,
                question=question,
            )

        refined_records = refined_df.to_dict(orient="records")
        all_results[f"{question_part}_output"] = {"refined_themes": refined_records}

        # LLM-as-judge: 4-dimension quality evaluation
        quality_results = refinement_evaluator(
            output={"themes": refined_records},
            expected_output={"themes": original_records},
        )

        for evaluation in quality_results:
            name = (
                evaluation.get("name", "")
                if isinstance(evaluation, dict)
                else evaluation.name
            )
            value = (
                evaluation.get("value", 0.0)
                if isinstance(evaluation, dict)
                else evaluation.value
            )
            all_results[f"{question_part}_{name}"] = value
            print(f"  {question_part}/{name}: {value}")

    print("Refinement Eval Results (local)")
    return all_results


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()

    parser = argparse.ArgumentParser(description="Run theme refinement evaluation")
    parser.add_argument(
        "--dataset",
        default="gambling_XS",
        help="Dataset identifier (e.g., gambling_XS)",
    )
    args = parser.parse_args()

    asyncio.run(evaluate_refinement(dataset=args.dataset))
