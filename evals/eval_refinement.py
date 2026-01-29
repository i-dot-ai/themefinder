"""Theme refinement evaluation with Langfuse dataset and experiment support.

Evaluates the theme refinement task that improves theme labels and descriptions.
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
from themefinder import theme_refinement
from utils import read_and_render


async def evaluate_refinement(
    dataset: str = "gambling_XS",
) -> dict:
    """Run refinement evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")

    Returns:
        Dict containing evaluation results
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="refinement")
    session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    langfuse_ctx = langfuse_utils.get_langfuse_context(
        session_id=session_id,
        eval_type="refinement",
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

    Note: Refinement uses qualitative LLM evaluation, so we run it
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
        question = item.input.get("question", "")

        # Run refinement
        refined_df = asyncio.run(
            theme_refinement(
                themes_df,
                llm=llm,
                question=question,
            )
        )

        # Handle tuple return
        if isinstance(refined_df, tuple):
            refined_df = refined_df[0]

        return {"refined_themes": refined_df.to_dict(orient="records")}

    with langfuse_utils.trace_context(ctx):
        result = dataset.run_experiment(
            name=ctx.session_id,
            task=task,
            evaluators=[],  # Qualitative evaluation done separately
            max_concurrency=1,
        )

    print(f"Refinement Eval Results (Langfuse experiment): {ctx.session_id}")
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
        question = item["input"].get("question", "")

        with langfuse_utils.trace_context(
            langfuse_utils.LangfuseContext(client=None, handler=None)
        ):
            refined_df, _ = await theme_refinement(
                themes_df,
                llm=llm,
                question=question,
            )

        # Qualitative evaluation via LLM
        # Original themes have topic_label, topic_description, and combined topic
        original_themes = themes_df[["topic_label", "topic_description"]].to_dict(orient="records")
        # Refined themes have combined topic column - parse it back for comparison
        refined_themes = []
        for _, row in refined_df.iterrows():
            topic_parts = row["topic"].split(": ", 1)
            refined_themes.append({
                "topic_label": topic_parts[0] if len(topic_parts) > 0 else "",
                "topic_description": topic_parts[1] if len(topic_parts) > 1 else "",
            })

        eval_prompt = read_and_render(
            "refinement_eval.txt",
            {"original_topics": original_themes, "new_topics": refined_themes},
        )
        response = llm.invoke(eval_prompt)
        print(f"Theme Refinement ({question_part}): \n {response.content}")

        all_results[f"{question_part}_evaluation"] = response.content

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run theme refinement evaluation")
    parser.add_argument("--dataset", default="gambling_XS", help="Dataset identifier (e.g., gambling_XS)")
    args = parser.parse_args()

    asyncio.run(evaluate_refinement(dataset=args.dataset))
