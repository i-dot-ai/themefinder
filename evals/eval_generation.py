"""Theme generation evaluation with Langfuse dataset and experiment support.

Evaluates the full theme generation pipeline (generation -> condensation -> refinement)
against a ground truth theme framework.
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

import langfuse_utils
from datasets import DatasetConfig, load_local_data
from evaluators import create_coverage_evaluator, create_groundedness_evaluator
from metrics import calculate_generation_metrics
from themefinder import theme_condensation, theme_generation, theme_refinement


async def evaluate_generation(
    dataset: str = "gambling_XS",
) -> dict:
    """Run generation evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")

    Returns:
        Dict containing evaluation scores
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="generation")
    session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    langfuse_ctx = langfuse_utils.get_langfuse_context(
        session_id=session_id,
        eval_type="generation",
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
        result = await _run_with_langfuse(langfuse_ctx, config, llm, callbacks)
    else:
        result = await _run_local_fallback(config, llm, callbacks)

    langfuse_utils.flush(langfuse_ctx)
    return result


async def _run_with_langfuse(ctx, config: DatasetConfig, llm, callbacks: list) -> dict:
    """Run experiment using Langfuse dataset.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance
        callbacks: LangChain callbacks list

    Returns:
        Experiment result dict
    """
    try:
        dataset = ctx.client.get_dataset(config.name)
    except Exception as e:
        print(f"Dataset {config.name} not found in Langfuse, falling back to local: {e}")
        return await _run_local_fallback(config, llm, callbacks)

    def task(*, item, **kwargs) -> dict:
        """Task function for experiment runner."""
        responses_df = pd.DataFrame(item.input["responses"])
        question = item.input["question"]

        # Run full pipeline synchronously (experiment runner is sync)
        themes_df = asyncio.run(
            theme_generation(
                responses_df=responses_df,
                llm=llm,
                question=question,
            )
        )
        condensed_df = asyncio.run(
            theme_condensation(
                themes_df,
                llm=llm,
                question=question,
            )
        )
        refined_df = asyncio.run(
            theme_refinement(
                condensed_df,
                llm=llm,
                question=question,
            )
        )

        # Handle tuple returns (df, metadata)
        if isinstance(themes_df, tuple):
            themes_df = themes_df[0]
        if isinstance(condensed_df, tuple):
            condensed_df = condensed_df[0]
        if isinstance(refined_df, tuple):
            refined_df = refined_df[0]

        return {"themes": refined_df.to_dict(orient="records")}

    with langfuse_utils.trace_context(ctx):
        result = dataset.run_experiment(
            name=ctx.session_id,
            task=task,
            evaluators=[
                create_groundedness_evaluator(llm),
                create_coverage_evaluator(llm),
            ],
            max_concurrency=1,
        )

    print(f"Theme Generation Eval Results (Langfuse experiment): {ctx.session_id}")
    return {"experiment_id": ctx.session_id, "result": result}


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
    parser.add_argument("--dataset", default="gambling_XS", help="Dataset identifier (e.g., gambling_XS)")
    args = parser.parse_args()

    asyncio.run(evaluate_generation(dataset=args.dataset))
