"""Theme condensation evaluation with Langfuse dataset and experiment support.

Evaluates the theme condensation task that reduces a large set of themes
to a smaller, more manageable set.
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
from evaluators import calculate_redundancy_score
from langchain_openai import AzureChatOpenAI
from utils import read_and_render

from themefinder import theme_condensation


async def evaluate_condensation(
    dataset: str = "gambling_XS",
    llm: AzureChatOpenAI | None = None,
    langfuse_ctx: langfuse_utils.LangfuseContext | None = None,
    judge_llm: AzureChatOpenAI | None = None,
) -> dict:
    """Run condensation evaluation.

    Args:
        dataset: Dataset identifier (e.g., "gambling_S", "healthcare_M")
        llm: Optional pre-configured LLM instance (for benchmark runs)
        langfuse_ctx: Optional pre-configured Langfuse context (for benchmark runs)

    Returns:
        Dict containing evaluation results
    """
    dotenv.load_dotenv()

    config = DatasetConfig(dataset=dataset, stage="condensation")

    # Use provided context or create new one
    owns_context = langfuse_ctx is None
    if langfuse_ctx is None:
        session_id = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        langfuse_ctx = langfuse_utils.get_langfuse_context(
            session_id=session_id,
            eval_type="condensation",
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

    Note: Condensation uses qualitative LLM evaluation, so no numeric evaluators.

    Args:
        ctx: LangfuseContext
        config: DatasetConfig
        llm: LangChain LLM instance
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
        return await _run_local_fallback(config, llm, callbacks)

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
            question = item.input["question"]

            # Run condensation
            condensed_df, _ = await theme_condensation(
                themes_df,
                llm=llm,
                question=question,
            )

            condensed_records = condensed_df.to_dict(orient="records")
            output = {"condensed_themes": condensed_records}

            # Update trace with output
            if trace:
                trace.update(output=output)

            # Calculate redundancy score for condensed themes
            redundancy = calculate_redundancy_score(condensed_records)

            if trace_id and ctx.client:
                comment = f"{redundancy['n_redundant_pairs']}/{redundancy['n_total_pairs']} pairs above threshold"
                if redundancy["flagged_pairs"]:
                    pair_strs = [f"  {p['theme_a']} ↔ {p['theme_b']} ({p['similarity']})" for p in redundancy["flagged_pairs"]]
                    comment += "\n" + "\n".join(pair_strs)
                ctx.client.create_score(
                    trace_id=trace_id,
                    name="redundancy",
                    value=round(redundancy["ratio"], 2),
                    data_type="NUMERIC",
                    comment=comment,
                )

            # Collect for return
            item_key = item.metadata.get("question_part", item.id)
            all_results[f"{item_key}_output"] = output
            all_results[f"{item_key}_redundancy"] = round(redundancy["ratio"], 2)

    print(f"Condensation Eval Results: {ctx.session_id}")
    return all_results


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
        original_themes = themes_df[["topic_label", "topic_description"]].to_dict(
            orient="records"
        )
        condensed_themes = condensed_df[["topic_label", "topic_description"]].to_dict(
            orient="records"
        )

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
    parser.add_argument(
        "--dataset",
        default="gambling_XS",
        help="Dataset identifier (e.g., gambling_XS)",
    )
    args = parser.parse_args()

    asyncio.run(evaluate_condensation(dataset=args.dataset))
