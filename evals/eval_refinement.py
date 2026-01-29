import asyncio
import io
import os
from datetime import datetime

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

import langfuse_utils
from themefinder import theme_refinement
from utils import download_file_from_bucket, read_and_render


def load_condensed_themes() -> pd.DataFrame:
    dotenv.load_dotenv()
    bucket_name = os.getenv("THEMEFINDER_S3_BUCKET_NAME")
    condensed_themes = pd.read_csv(
        io.BytesIO(
            download_file_from_bucket(
                "app_data/evals/theme_refinement/eval_condensed_topics.csv",
                bucket_name=bucket_name,
            )
        )
    )
    return condensed_themes


async def evaluate_refinement():
    dotenv.load_dotenv()

    # Langfuse setup
    session_id = f"eval_refinement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    langfuse_ctx = langfuse_utils.get_langfuse_context(
        session_id=session_id,
        eval_type="refinement",
    )
    callbacks = [langfuse_ctx.handler] if langfuse_ctx.handler else []

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
        callbacks=callbacks,
    )

    condensed_themes = load_condensed_themes()

    # Wrap all LLM calls in trace_context to propagate tags/metadata
    with langfuse_utils.trace_context(langfuse_ctx):
        refined_themes, _ = await theme_refinement(
            condensed_themes,
            llm=llm,
            question="",
        )
        condensed_themes_dict = condensed_themes[
            ["topic_label", "topic_description"]
        ].to_dict(orient="records")
        eval_prompt = read_and_render(
            "refinement_eval.txt",
            {"original_topics": condensed_themes_dict, "new_topics": refined_themes},
        )
        response = llm.invoke(eval_prompt)

    print(f"Theme Refinement Eval Results: \n {response.content}")

    # Flush (no numeric scores for qualitative eval)
    langfuse_utils.flush(langfuse_ctx)


if __name__ == "__main__":
    asyncio.run(evaluate_refinement())
