import asyncio
import io
import os

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

from themefinder import theme_refinement
from utils import download_file_from_bucket, read_and_render


def load_condensed_themes() -> tuple[str, pd.DataFrame]:
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
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
    )
    condensed_themes = load_condensed_themes()
    refined_themes = await theme_refinement(
        condensed_themes,
        llm=llm,
        question="",
    )
    condensed_themes = condensed_themes[["topic_label", "topic_description"]].to_dict(
        orient="records"
    )
    eval_prompt = read_and_render(
        "refinement_eval.txt",
        {"original_topics": condensed_themes, "new_topics": refined_themes},
    )
    response = llm.invoke(eval_prompt)
    print(f"Theme Refinement Eval Results: \n {response.content}")


if __name__ == "__main__":
    asyncio.run(evaluate_refinement())
