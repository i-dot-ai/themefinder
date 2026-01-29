import ast
import asyncio
import io
import json
import os
from datetime import datetime

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

import langfuse_utils
from metrics import calculate_mapping_metrics
from themefinder import theme_mapping
from utils import download_file_from_bucket


def load_mapped_responses(
    question_number: int = 1,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    dotenv.load_dotenv()
    bucket_name = os.getenv("THEMEFINDER_S3_BUCKET_NAME")
    question = download_file_from_bucket(
        f"app_data/evals/theme_mapping/question_{question_number}_expanded_question.txt",
        bucket_name=bucket_name,
    ).decode()
    topics = pd.DataFrame(
        json.loads(
            download_file_from_bucket(
                f"app_data/evals/theme_mapping/question_{question_number}_topics.json",
                bucket_name=bucket_name,
            )
        )
    ).T
    topics["topic"] = topics["topic_name"] + ": " + topics["rationale"]
    topics = topics.rename_axis("topic_id").reset_index()
    responses = pd.read_csv(
        io.BytesIO(
            download_file_from_bucket(
                f"app_data/evals/theme_mapping/question_{question_number}_responses.csv",
                bucket_name=bucket_name,
            )
        )
    )
    responses["topics"] = responses["topics"].apply(ast.literal_eval)
    return question, topics[["topic_id", "topic"]], responses


async def evaluate_mapping(question_num: int | None = None):
    dotenv.load_dotenv()

    # Langfuse setup
    session_id = f"eval_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    langfuse_ctx = langfuse_utils.get_langfuse_context(
        session_id=session_id,
        metadata={
            "eval_type": "mapping",
            "model": os.getenv("DEPLOYMENT_NAME", "unknown"),
        },
    )
    callbacks = [langfuse_ctx.handler] if langfuse_ctx.handler else []

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
        callbacks=callbacks,
    )

    questions_to_process = [question_num] if question_num is not None else range(1, 4)
    all_scores = {}

    for i in questions_to_process:
        question, topics, responses = load_mapped_responses(i)
        result, _ = await theme_mapping(
            responses_df=responses[["response_id", "response"]],
            llm=llm,
            question=question,
            refined_themes_df=topics,
        )
        responses = responses.merge(
            result[["response_id", "labels"]], "inner", on="response_id"
        )
        mapping_metrics = calculate_mapping_metrics(
            df=responses, column_one="topics", column_two="labels"
        )
        print(f"Theme Mapping Question {i} Eval Results: \n {mapping_metrics}")

        # Collect scores with question prefix
        for key, value in mapping_metrics.items():
            if isinstance(value, (int, float)):
                all_scores[f"q{i}_{key}"] = value

    # Attach scores and flush
    langfuse_utils.create_scores(langfuse_ctx, all_scores)
    langfuse_utils.flush(langfuse_ctx)


if __name__ == "__main__":
    asyncio.run(evaluate_mapping())
