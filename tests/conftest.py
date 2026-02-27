from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


@pytest.fixture()
def mock_llm():
    mock = AsyncMock()
    return mock


@pytest.fixture()
def mock_sync_llm():
    mock = MagicMock()
    return mock


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"response_id": [1, 2], "text": ["response1", "response2"]})


@pytest.fixture
def sample_themes_df():
    """Create a sample DataFrame with themes for testing."""
    df = pd.DataFrame(
        {
            "topic_id": ["A", "B", "C", "D"],
            "topic_label": ["Theme A", "Theme B", "Theme C", "Theme D"],
            "topic_description": [
                "Description of theme A",
                "Description of theme B",
                "Description of theme C",
                "Description of theme D",
            ],
            "source_topic_count": [5, 3, 7, 2],
        }
    )
    df["response_id"] = range(1, len(df) + 1)
    return df


@pytest.fixture()
def sample_responses_df():
    return pd.DataFrame(
        {
            "response_id": [1, 2],
            "response": ["response1", "response2"],
        }
    )
