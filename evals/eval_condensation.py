"""Theme condensation evaluation with Langfuse dataset and experiment support.

Evaluates the theme condensation task that reduces a large set of themes
to a smaller, more manageable set.
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
from themefinder import theme_condensation
from utils import read_and_render


async def evaluate_condensation(
    dataset: str = "gambling_XS",
) -> dict:
    """Run condensation evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")

    Returns:
        Dict containing evaluation results
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="condensation")
    session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    langfuse_ctx = langfuse_utils.get_langfuse_context(
        session_id=session_id,
        eval_type="condensation",
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

    Note: Condensation uses qualitative LLM evaluation, so we run it
    but don't use numeric evaluators.

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
        themes_df = pd.DataFrame(item.input["themes"])
        question = item.input["question"]

        # Run condensation
        condensed_df = asyncio.run(
            theme_condensation(
                themes_df,
                llm=llm,
                question=question,
            )
        )

        # Handle tuple return
        if isinstance(condensed_df, tuple):
            condensed_df = condensed_df[0]

        return {"condensed_themes": condensed_df.to_dict(orient="records")}

    with langfuse_utils.trace_context(ctx):
        result = dataset.run_experiment(
            name=ctx.session_id,
            task=task,
            evaluators=[],  # Qualitative evaluation done separately
            max_concurrency=1,
        )

    print(f"Condensation Eval Results (Langfuse experiment): {ctx.session_id}")
    return {"experiment_id": ctx.session_id, "result": result}


async def _run_local_fallback(config: DatasetConfig, llm, callbacks: list) -> dict:
    """Run evaluation without Langfuse (local development).

    Args:
        config: DatasetConfig
        llm: LangChain LLM instance
        callbacks: LangChain callbacks list

    Returns:
        Dict containing evaluation response
    """
    data_items = load_local_data(config)
    all_results = {}

    for item in data_items:
        question_part = item.get("metadata", {}).get("question_part", "unknown")
        themes_df = pd.DataFrame(item["input"]["themes"])
        question = item["input"]["question"]

        with langfuse_utils.trace_context(
            langfuse_utils.LangfuseContext(client=None, handler=None)
        ):
            condensed_df, _ = await theme_condensation(
                themes_df,
                llm=llm,
                question=question,
            )

        # Qualitative evaluation via LLM
        original_themes = themes_df[["topic_label", "topic_description"]].to_dict(orient="records")
        condensed_themes = condensed_df[["topic_label", "topic_description"]].to_dict(orient="records")

        eval_prompt = read_and_render(
            "condensation_eval.txt",
            {"original_topics": original_themes, "condensed_topics": condensed_themes},
        )
        response = llm.invoke(eval_prompt)
        print(f"Theme Condensation ({question_part}): \n {response.content}")

        all_results[f"{question_part}_evaluation"] = response.content

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run theme condensation evaluation")
    parser.add_argument("--dataset", default="gambling_XS", help="Dataset identifier (e.g., gambling_XS)")
    args = parser.parse_args()

    asyncio.run(evaluate_condensation(dataset=args.dataset))
