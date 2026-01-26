import asyncio
import json
from pathlib import Path

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

from metrics import calculate_generation_metrics
from themefinder import theme_condensation, theme_generation, theme_refinement


def load_responses_and_framework() -> tuple[pd.DataFrame, str, dict]:
    data_dir = Path(__file__).parent / "data/generation"
    sentiments = pd.read_csv(data_dir / "eval_sentiments.csv")
    with (data_dir / "expanded_question.txt").open() as f:
        question = f.read()
    with (data_dir / "framework_themes.json").open() as f:
        theme_framework = json.load(f)
    return sentiments, question, theme_framework


async def evaluate_generation():
    dotenv.load_dotenv()
    import os

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
    )
    sentiments, question, theme_framework = load_responses_and_framework()
    themes_df, _ = await theme_generation(
        responses_df=sentiments,
        llm=llm,
        question=question,
    )
    condensed_themes_df, _ = await theme_condensation(
        themes_df,
        llm=llm,
        question=question,
    )
    refined_themes_df, _ = await theme_refinement(
        condensed_themes_df,
        llm=llm,
        question=question,
    )
    eval_scores = calculate_generation_metrics(refined_themes_df, theme_framework)
    print(f"Theme Generation Eval Results: \n {eval_scores}")


if __name__ == "__main__":
    asyncio.run(evaluate_generation())
