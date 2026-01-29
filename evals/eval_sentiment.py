import asyncio
import io
import os

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

from metrics import calculate_sentiment_metrics
from themefinder import sentiment_analysis
from utils import download_file_from_bucket


def load_responses() -> tuple[str, pd.DataFrame]:
    dotenv.load_dotenv()
    bucket_name = os.getenv("THEMEFINDER_S3_BUCKET_NAME")
    question = download_file_from_bucket(
        "app_data/evals/response_sentiment/expanded_question.txt",
        bucket_name=bucket_name,
    ).decode("utf-8")
    responses = pd.read_parquet(
        io.BytesIO(
            download_file_from_bucket(
                "app_data/evals/response_sentiment/responses.parquet",
                bucket_name=bucket_name,
            )
        )
    )
    return question, responses


async def evaluate_sentiment():
    dotenv.load_dotenv()

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
    )

    question, responses = load_responses()

    result, _ = await sentiment_analysis(
        responses_df=responses[["response_id", "response"]],
        llm=llm,
        question=question,
    )

    responses = responses.rename(columns={"position": "supervisor_position"})
    result = result.rename(columns={"position": "ai_position"})
    merged = responses.merge(
        result[["response_id", "ai_position"]], "inner", on="response_id"
    )
    merged["ai_position"] = merged["ai_position"].map(
        {"DISAGREEMENT": "DISAGREE", "AGREEMENT": "AGREE"}
    )
    accuracy = calculate_sentiment_metrics(merged)
    print(f"AI Agreement Accuracy: {accuracy["accuracy"]}")


if __name__ == "__main__":
    asyncio.run(evaluate_sentiment())
