"""Theme generation evaluation with Langfuse dataset and experiment support.

Evaluates the full theme generation pipeline (generation -> condensation -> refinement)
against a ground truth theme framework.
"""

import argparse
import asyncio
import contextvars
import logging
import os
from datetime import datetime
from functools import partial

import dotenv
import langfuse_utils
import pandas as pd
from datasets import DatasetConfig, load_local_data
from evaluators import (
    create_coverage_evaluator,
    create_groundedness_evaluator,
    create_redundancy_evaluator,
    create_title_specificity_evaluator,
)
from langchain_openai import AzureChatOpenAI
from metrics import calculate_generation_metrics

from themefinder import theme_condensation, theme_generation, theme_refinement

logger = logging.getLogger("themefinder.evals.generation")


async def evaluate_generation(
    dataset: str = "gambling_XS",
    llm: AzureChatOpenAI | None = None,
    langfuse_ctx: langfuse_utils.LangfuseContext | None = None,
    judge_llm: AzureChatOpenAI | None = None,
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
        result = await _run_with_langfuse(
            langfuse_ctx, config, llm, callbacks, judge_llm=judge_llm
        )
    else:
        result = await _run_local_fallback(config, llm, callbacks)

    # Only flush if we created the context
    if owns_context:
        langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(
    ctx, config: DatasetConfig, llm, callbacks: list, judge_llm=None
) -> dict:
    """Run evaluation with manual dataset iteration for proper trace control.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance
        callbacks: LangChain callbacks list
        judge_llm: Optional dedicated judge LLM (defaults to task llm)

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

    # Use dedicated judge LLM if provided, otherwise fall back to task LLM
    eval_llm = judge_llm or llm

    # Create evaluator functions
    groundedness_evaluator = create_groundedness_evaluator(eval_llm)
    coverage_evaluator = create_coverage_evaluator(eval_llm)
    specificity_evaluator = create_title_specificity_evaluator(eval_llm)
    redundancy_evaluator = create_redundancy_evaluator()

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

            # Parse combined "label: description" topic field into separate fields
            # for evaluators that need the label alone (specificity, redundancy)
            themes = []
            for record in refined_df.to_dict(orient="records"):
                if "topic" in record and ":" in record["topic"]:
                    label, description = record["topic"].split(":", 1)
                    record["topic_label"] = label.strip()
                    record["topic_description"] = description.strip()
                themes.append(record)
            output = {"themes": themes}

            # Update trace with output
            if trace:
                trace.update(output=output)

            # Run LLM-based evaluators concurrently (copy context per task for logging)
            loop = asyncio.get_event_loop()
            llm_evaluators = [
                groundedness_evaluator,
                coverage_evaluator,
                specificity_evaluator,
            ]
            llm_tasks = [
                loop.run_in_executor(
                    None,
                    partial(
                        contextvars.copy_context().run,
                        evaluator,
                        output=output,
                        expected_output=item.expected_output,
                    ),
                )
                for evaluator in llm_evaluators
            ]
            try:
                llm_results = await asyncio.wait_for(
                    asyncio.gather(*llm_tasks, return_exceptions=True),
                    timeout=300,
                )
            except asyncio.TimeoutError:
                logger.error("LLM evaluators timed out after 300s")
                llm_results = [TimeoutError("LLM evaluators timed out")] * len(
                    llm_evaluators
                )

            # Run redundancy evaluator synchronously (embedding-based, no LLM call)
            redundancy_result = redundancy_evaluator(
                output=output, expected_output=item.expected_output
            )

            # Process all results
            all_eval_results = list(llm_results) + [redundancy_result]
            item_key = item.metadata.get("question_part", item.id)
            for eval_result in all_eval_results:
                if isinstance(eval_result, Exception):
                    logger.error(f"Evaluator failed: {eval_result}")
                    continue

                if trace_id and ctx.client:
                    # Extract name, value, comment from either Evaluation object or dict
                    if isinstance(eval_result, dict):
                        eval_name = eval_result.get("name", "unknown")
                        eval_value = eval_result.get("value", 0.0)
                        eval_comment = eval_result.get("comment")
                    else:
                        eval_name = eval_result.name
                        eval_value = eval_result.value
                        eval_comment = getattr(eval_result, "comment", None)

                    score_kwargs = {
                        "trace_id": trace_id,
                        "name": eval_name,
                        "value": eval_value,
                        "data_type": "NUMERIC",
                    }
                    if eval_comment:
                        score_kwargs["comment"] = eval_comment
                    ctx.client.create_score(**score_kwargs)

                # Collect for return
                if isinstance(eval_result, dict):
                    all_scores[f"{item_key}_{eval_result.get('name', 'unknown')}"] = (
                        eval_result.get("value", 0.0)
                    )
                else:
                    all_scores[f"{item_key}_{eval_result.name}"] = eval_result.value

            # Include pipeline output for disk persistence
            all_scores[f"{item_key}_output"] = output

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
    import nest_asyncio

    nest_asyncio.apply()

    parser = argparse.ArgumentParser(description="Run theme generation evaluation")
    parser.add_argument(
        "--dataset",
        default="gambling_XS",
        help="Dataset identifier (e.g., gambling_XS)",
    )
    args = parser.parse_args()

    asyncio.run(evaluate_generation(dataset=args.dataset))
