"""Langfuse dataset management utilities for ThemeFinder evaluations.

Provides dataset configuration, naming conventions, and local fallback loading
for evaluation datasets.

Dataset naming: eval/{dataset}/{stage}
Where dataset is a descriptor like "gambling_100", "healthcare_500", "tuition_1000"
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("themefinder.evals.datasets")

VALID_STAGES = ["generation", "sentiment", "mapping", "condensation", "refinement"]

# Data directory for local eval datasets
DATA_DIR = Path(__file__).parent / "data"


@dataclass
class DatasetConfig:
    """Configuration for a Langfuse dataset.

    Attributes:
        dataset: Dataset identifier (e.g., "gambling_100", "healthcare_500")
        stage: Evaluation stage (generation, sentiment, mapping, condensation, refinement)
    """

    dataset: str
    stage: str

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{self.stage}'. Must be one of: {VALID_STAGES}"
            )

    @property
    def name(self) -> str:
        """Langfuse dataset name using folder notation."""
        return f"eval/{self.dataset}/{self.stage}"

    @property
    def local_path(self) -> Path:
        """Path to local data directory for this dataset."""
        return DATA_DIR / self.dataset


def get_or_create_dataset(client: Any, config: DatasetConfig) -> Any:
    """Get existing dataset or create a new one.

    Args:
        client: Langfuse client instance
        config: Dataset configuration

    Returns:
        Langfuse dataset object
    """
    try:
        return client.get_dataset(config.name)
    except Exception:
        logger.info(f"Creating new dataset: {config.name}")
        return client.create_dataset(
            name=config.name,
            description=f"{config.stage.title()} eval for {config.dataset}",
            metadata={
                "dataset": config.dataset,
                "stage": config.stage,
            },
        )


def _load_responses(
    config: DatasetConfig, question_part: str = "question_part_1"
) -> pd.DataFrame:
    """Load responses from JSONL.

    Args:
        config: Dataset configuration
        question_part: Which question part to load (question_part_1, question_part_2)

    Returns:
        DataFrame with response_id and response columns
    """
    responses_path = config.local_path / "inputs" / question_part / "responses.jsonl"
    return pd.read_json(responses_path, lines=True)


def _load_question(
    config: DatasetConfig, question_part: str = "question_part_1"
) -> str:
    """Load question text.

    Args:
        config: Dataset configuration
        question_part: Which question part to load

    Returns:
        Question text string (includes scale_statement for agree/disagree questions)
    """
    question_path = config.local_path / "inputs" / question_part / "question.json"
    with open(question_path) as f:
        data = json.load(f)
    question_text = data["question_text"]

    # For agree/disagree questions, append the scale statement
    if data.get("scale_statement"):
        question_text = f'{question_text}\n\nStatement: "{data["scale_statement"]}"'

    return question_text


def _load_themes(
    config: DatasetConfig, question_part: str = "question_part_1"
) -> list[dict]:
    """Load themes.

    Args:
        config: Dataset configuration
        question_part: Which question part to load

    Returns:
        List of theme dicts with topic_id, topic_label, topic_description
    """
    # Find the most recent output date
    outputs_dir = config.local_path / "outputs" / "mapping"
    date_dirs = sorted(outputs_dir.iterdir(), reverse=True)
    if not date_dirs:
        raise FileNotFoundError(f"No output dates found in {outputs_dir}")

    themes_path = date_dirs[0] / question_part / "themes.json"
    with open(themes_path) as f:
        return json.load(f)


def _load_sentiment(
    config: DatasetConfig, question_part: str = "question_part_1"
) -> dict[str, str]:
    """Load sentiment labels.

    Args:
        config: Dataset configuration
        question_part: Which question part to load

    Returns:
        Dict mapping response_id to position (AGREE/DISAGREE/UNCLEAR)
    """
    outputs_dir = config.local_path / "outputs" / "mapping"
    date_dirs = sorted(outputs_dir.iterdir(), reverse=True)
    if not date_dirs:
        raise FileNotFoundError(f"No output dates found in {outputs_dir}")

    sentiment_path = date_dirs[0] / question_part / "sentiment.jsonl"
    df = pd.read_json(sentiment_path, lines=True)

    return dict(zip(df["response_id"].astype(str), df["position"]))


def _load_mapping(
    config: DatasetConfig, question_part: str = "question_part_1"
) -> dict[str, list[str]]:
    """Load theme mappings.

    Args:
        config: Dataset configuration
        question_part: Which question part to load

    Returns:
        Dict mapping response_id to list of topic_ids (labels)
    """
    outputs_dir = config.local_path / "outputs" / "mapping"
    date_dirs = sorted(outputs_dir.iterdir(), reverse=True)
    if not date_dirs:
        raise FileNotFoundError(f"No output dates found in {outputs_dir}")

    mapping_path = date_dirs[0] / question_part / "mapping.jsonl"
    df = pd.read_json(mapping_path, lines=True)

    return dict(zip(df["response_id"].astype(str), df["labels"]))


def _get_question_parts(config: DatasetConfig) -> list[str]:
    """Get available question parts for a dataset.

    Args:
        config: Dataset configuration

    Returns:
        List of question part directory names
    """
    inputs_dir = config.local_path / "inputs"
    return sorted(
        [
            d.name
            for d in inputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("question_part")
        ]
    )


def load_local_generation_data(config: DatasetConfig) -> list[dict]:
    """Load generation eval data from local files.

    Args:
        config: Dataset configuration

    Returns:
        List of dicts (one per question) with input and expected_output
    """
    items = []
    for question_part in _get_question_parts(config):
        question = _load_question(config, question_part)
        responses = _load_responses(config, question_part)
        themes = _load_themes(config, question_part)

        items.append(
            {
                "input": {
                    "question": question,
                    "responses": responses[["response_id", "response"]].to_dict(
                        orient="records"
                    ),
                },
                "expected_output": {"themes": themes},
                "metadata": {"question_part": question_part},
            }
        )

    return items


def load_local_sentiment_data(config: DatasetConfig) -> list[dict]:
    """Load sentiment eval data from local files.

    Args:
        config: Dataset configuration

    Returns:
        List of dicts (one per question) with input and expected_output
    """
    items = []
    for question_part in _get_question_parts(config):
        question = _load_question(config, question_part)
        responses = _load_responses(config, question_part)
        positions = _load_sentiment(config, question_part)

        items.append(
            {
                "input": {
                    "question": question,
                    "responses": responses[["response_id", "response"]].to_dict(
                        orient="records"
                    ),
                },
                "expected_output": {"positions": positions},
                "metadata": {"question_part": question_part},
            }
        )

    return items


def load_local_mapping_data(config: DatasetConfig) -> list[dict]:
    """Load mapping eval data from local files.

    Args:
        config: Dataset configuration

    Returns:
        List of dicts (one per question) with input and expected_output
    """
    items = []
    for question_part in _get_question_parts(config):
        question = _load_question(config, question_part)
        responses = _load_responses(config, question_part)
        themes = _load_themes(config, question_part)
        mappings = _load_mapping(config, question_part)

        items.append(
            {
                "input": {
                    "question": question,
                    "topics": themes,
                    "responses": responses[["response_id", "response"]].to_dict(
                        orient="records"
                    ),
                },
                "expected_output": {"mappings": mappings},
                "metadata": {"question_part": question_part},
            }
        )

    return items


def load_local_condensation_data(config: DatasetConfig) -> list[dict]:
    """Load condensation eval data from local files.

    Note: Uses the generated themes as input for condensation testing.

    Args:
        config: Dataset configuration

    Returns:
        List of dicts with input (themes) and expected_output
    """
    items = []
    for question_part in _get_question_parts(config):
        question = _load_question(config, question_part)
        themes = _load_themes(config, question_part)

        items.append(
            {
                "input": {
                    "question": question,
                    "themes": [
                        {
                            "topic_label": t["topic_label"],
                            "topic_description": t["topic_description"],
                        }
                        for t in themes
                    ],
                },
                "expected_output": None,  # Qualitative evaluation
                "metadata": {"question_part": question_part},
            }
        )

    return items


def load_local_refinement_data(config: DatasetConfig) -> list[dict]:
    """Load refinement eval data from local files.

    Note: Uses the generated themes as input for refinement testing.

    Args:
        config: Dataset configuration

    Returns:
        List of dicts with input (themes) and expected_output
    """
    items = []
    for question_part in _get_question_parts(config):
        question = _load_question(config, question_part)
        themes = _load_themes(config, question_part)

        items.append(
            {
                "input": {
                    "question": question,
                    "themes": [
                        {
                            "topic_label": t["topic_label"],
                            "topic_description": t["topic_description"],
                        }
                        for t in themes
                    ],
                },
                "expected_output": None,  # Qualitative evaluation
                "metadata": {"question_part": question_part},
            }
        )

    return items


def load_local_data(config: DatasetConfig) -> list[dict]:
    """Load data from local files for fallback mode.

    Args:
        config: Dataset configuration

    Returns:
        List of dataset items with input and expected_output
    """
    loaders = {
        "generation": load_local_generation_data,
        "sentiment": load_local_sentiment_data,
        "mapping": load_local_mapping_data,
        "condensation": load_local_condensation_data,
        "refinement": load_local_refinement_data,
    }

    loader = loaders.get(config.stage)
    if not loader:
        raise ValueError(f"No loader for stage: {config.stage}")

    return loader(config)
