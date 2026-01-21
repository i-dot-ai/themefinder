import argparse
import asyncio
import datetime
import json
import logging
import os
from pathlib import Path

import boto3
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from themefinder import theme_condensation, theme_generation, theme_refinement
from themefinder.advanced_tasks.theme_clustering_agent import ThemeClusteringAgent
from themefinder.models import HierarchicalClusteringResponse, ThemeNode


class ThemeNodeList(BaseModel):
    """List of theme nodes for serialization to JSON."""

    theme_nodes: list[ThemeNode]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("DATA_S3_BUCKET")
BASE_PREFIX = "app_data/consultations/"


def download_s3_subdir(subdir: str) -> None:
    """
    Recursively downloads the contents of a specified subdirectory from S3
    to a local directory with the same name as the subdir.
    """
    prefix = str(Path(BASE_PREFIX) / subdir).rstrip("/") + "/"
    logger.info("prefix: %s", prefix)
    logger.info("bucket: %s", BUCKET_NAME)

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    inputs_prefix = str(Path(BASE_PREFIX) / subdir / "inputs").rstrip("/") + "/"
    logger.info("S3 inputs prefix: %s", inputs_prefix)
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=inputs_prefix)
    logger.info(
        "Created paginator for bucket: %s with prefix: %s", BUCKET_NAME, inputs_prefix
    )

    for page in pages:
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if key.endswith((".csv", ".json", ".jsonl")):
                relative_path = os.path.relpath(key, prefix)
                local_path = Path(subdir) / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET_NAME, key, str(local_path))
                logger.info("Downloaded %s to %s", key, local_path)


def upload_directory_to_s3(local_path: str) -> None:
    """
    Recursively uploads all files from a local directory to a corresponding subdir path in S3.

    Args:
        local_path (str): Local path to the directory to upload (e.g., outputs/sign_off/2025_02_02).
    """
    s3 = boto3.client("s3")

    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = Path(root) / file
            s3_key = str(Path(BASE_PREFIX) / local_file_path).replace("\\", "/")
            s3.upload_file(str(local_file_path), BUCKET_NAME, s3_key)
            logger.info(
                "Uploaded %s to s3://%s/%s", local_file_path, BUCKET_NAME, s3_key
            )


def load_question(consultation_dir: str, question_dir: str) -> tuple:
    """
    Load question and response data from specified directories.

    Args:
        consultation_dir: Base directory of the consultation
        question_dir: Directory containing question-specific data

    Returns:
        tuple: (question text, dataframe of responses)
    """
    question_path = Path(consultation_dir) / "inputs" / question_dir / "question.json"
    responses_path = (
        Path(consultation_dir) / "inputs" / question_dir / "responses.jsonl"
    )

    with question_path.open() as f:
        question = json.load(f)["question_text"]

    # Read JSONL file line by line
    responses = []
    with responses_path.open() as f:
        for line in f:
            responses.append(json.loads(line))
    responses_df = pd.DataFrame(responses)
    responses = responses_df.rename(
        columns={"themefinder_id": "response_id", "text": "response"}
    )
    return question, responses


async def generate_themes(question: str, responses_df):
    """
    Generate refined themes from question and responses through multiple analysis steps.

    Args:
        question: The survey question text
        responses_df: DataFrame containing survey responses

    Returns:
        pd.DataFrame: DataFrame containing refined themes
    """
    theme_df, _ = await theme_generation(
        responses_df, llm, question=question, partition_key=None
    )

    condensed_theme_df, _ = await theme_condensation(
        theme_df,
        llm,
        question=question,
    )
    refined_themes_df, _ = await theme_refinement(
        condensed_theme_df,
        llm,
        question=question,
    )

    return refined_themes_df


def agentic_theme_clustering(
    refined_themes_df,
    llm,
    max_cluster_iterations=0,
    n_target_themes=10,
):
    """
    Cluster themes using hierarchical clustering.

    Parameters:
    -----------
    refined_themes_df : pd.DataFrame
        DataFrame containing themes with 'topic' column
    llm : object
        Language model object for theme clustering
    max_cluster_iterations : int, default=0
        Maximum number of clustering iterations to attempt
    n_target_themes : int, default=10
        Number of themes to target

    Returns:
    --------
    pd.DataFrame
        All themes from clustering
    """
    refined_themes_df[["topic_label", "topic_description"]] = refined_themes_df[
        "topic"
    ].str.split(":", n=1, expand=True)

    initial_themes = [
        ThemeNode(
            topic_id=row["topic_id"],
            topic_label=row["topic_label"],
            topic_description=row["topic_description"],
            source_topic_count=row["source_topic_count"],
        )
        for _, row in refined_themes_df.iterrows()
    ]
    agent = ThemeClusteringAgent(
        llm.with_structured_output(HierarchicalClusteringResponse),
        initial_themes,
    )
    logger.info(
        f"Clustering themes with max_iterations={max_cluster_iterations}, target_themes={n_target_themes}"
    )
    all_themes_df = None
    for i in range(3):
        try:
            all_themes_df = agent.cluster_themes(
                max_iterations=max_cluster_iterations, target_themes=n_target_themes
            )
            if len(all_themes_df) > 0:
                break
        except:  # noqa: E722
            logger.info(f"Error when clustering, attempt {i}, retrying")

    if all_themes_df is None or len(all_themes_df) == 0:
        logger.warning("Theme clustering failed after 3 attempts")
        return None

    return all_themes_df


async def process_consultation(consultation_dir: str, llm) -> str:
    """
    Process all questions in a consultation directory, generating theme analyses.

    Creates a new directory structure for outputs:
    - consultation_dir/
      - outputs/
        - YYYY-MM-DD_HH-MM-SS/
          - question_dir_files

    Args:
        consultation_dir: Directory containing question subdirectories
        llm: Language model instance for processing
    """
    date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

    skipped_questions = []
    for question_dir in os.listdir(Path(consultation_dir) / "inputs"):
        if "question" in question_dir:
            logger.info("Processing %s...", question_dir)
            try:
                # Create question-specific directory in the datestamped folder
                question_output_dir = (
                    Path(consultation_dir)
                    / "outputs"
                    / "sign_off"
                    / date
                    / question_dir
                )
                if not question_output_dir.exists():
                    question_output_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Created outputs directory at: %s", question_output_dir)

                # Load question and responses
                question, responses_df = load_question(consultation_dir, question_dir)

                # Generate themes
                refined_themes_df = await generate_themes(question, responses_df)

                def refined_themes_to_theme_node(row: dict):
                    topic_label, topic_description = row["topic"].split(":")
                    return ThemeNode(
                        topic_id=row["topic_id"],
                        topic_label=topic_label,
                        topic_description=topic_description,
                        source_topic_count=row["source_topic_count"],
                        parent_id="0",
                        children=[],
                    )

                # Cluster themes if more than 20
                if len(refined_themes_df) > 20:
                    all_themes_df = agentic_theme_clustering(refined_themes_df, llm)
                    if all_themes_df is not None:
                        all_themes_list = [
                            ThemeNode(**row) for _, row in all_themes_df.iterrows()
                        ]
                    else:
                        logger.info("Clustering failed, using unclustered themes")
                        all_themes_list = [
                            refined_themes_to_theme_node(row)
                            for _, row in refined_themes_df.iterrows()
                        ]
                else:
                    logger.info("Fewer than 20 themes, clustering not required")
                    all_themes_list = [
                        refined_themes_to_theme_node(row)
                        for _, row in refined_themes_df.iterrows()
                    ]

                with open(
                    os.path.join(question_output_dir, "clustered_themes.json"), "w"
                ) as f:
                    f.write(
                        ThemeNodeList(theme_nodes=all_themes_list).model_dump_json()
                    )

                logger.info(
                    "Completed processing %s, saved to %s",
                    question_dir,
                    question_output_dir,
                )
            except Exception:
                logger.exception("Error processing %s", question_dir)
                skipped_questions.append(question_dir)
    if skipped_questions:
        logger.warning("Skipped questions: %s", skipped_questions)
    return str(Path(consultation_dir) / "outputs" / "sign_off" / date)


if __name__ == "__main__":
    llm = ChatOpenAI(
        model="gpt-4o-sweden",
        temperature=0,
        openai_api_base=os.environ["LLM_GATEWAY_URL"],
        openai_api_key=os.environ["LITELLM_CONSULT_OPENAI_API_KEY"],
    )

    parser = argparse.ArgumentParser(description="Download a subdirectory from S3.")
    parser.add_argument(
        "--subdir",
        type=str,
        required=True,
        help="Subdirectory within the base S3 path to download.",
    )
    parser.add_argument(
        "--job-type",
        type=str,
        required=False,
        help="Type of job to execute.",
    )
    args = parser.parse_args()

    logger.info("Starting processing for subdirectory: %s", args.subdir)
    download_s3_subdir(args.subdir)
    output_dir = asyncio.run(process_consultation(args.subdir, llm))
    upload_directory_to_s3(output_dir)
    logger.info("Processing completed for subdirectory: %s", args.subdir)
