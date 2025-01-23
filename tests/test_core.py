import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from themefinder import find_themes
from themefinder.themefinder_logging import logger


@pytest.mark.asyncio()
async def test_find_themes(mock_llm, sample_df):
    """Test the complete theme finding pipeline with mocked LLM responses."""
    mock_llm.ainvoke.side_effect = [
        MagicMock(
            content='{"responses": [{"response_id": 1, "position": "agreement", "text": "response1"}, {"response_id": 2, "position": "disagreement", "text": "response2"}]}'
        ),
        MagicMock(content='{"responses": [{"themes": ["theme1", "theme2"]}]}'),
        MagicMock(content='{"responses": [{"themes": ["theme3", "theme4"]}]}'),
        MagicMock(
            content='{"responses": [{"condensed_themes": ["main_theme1", "main_theme2"]}]}'
        ),
        MagicMock(content='{"responses": [{"topic_id": "label1", "topic": "desc1"}]}'),
        MagicMock(
            content=json.dumps(
                {
                    "responses": [
                        {
                            "response_id": 1,
                            "reason": "reason1",
                            "label": ["label1"],
                        },
                        {
                            "response_id": 2,
                            "reason": "reason2",
                            "label": ["label2"],
                        },
                    ]
                }
            )
        ),
    ]
    result = await find_themes(sample_df, mock_llm, question="test question")
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in [
            "sentiment",
            "topics",
            "condensed_topics",
            "mapping",
            "question",
            "refined_topics",
        ]
    )
    assert mock_llm.ainvoke.call_count == 6


@pytest.mark.asyncio()
async def test_find_themes_verbose_control(mock_llm, sample_df):
    """Test that the verbose parameter correctly controls logging output."""
    # Set up mock responses
    mock_llm.ainvoke.side_effect = [
        MagicMock(
            content='{"responses": [{"response_id": 1, "position": "agreement", "text": "response1"}, {"response_id": 2, "position": "disagreement", "text": "response2"}]}'
        ),
        MagicMock(content='{"responses": [{"themes": ["theme1", "theme2"]}]}'),
        MagicMock(content='{"responses": [{"themes": ["theme3", "theme4"]}]}'),
        MagicMock(
            content='{"responses": [{"condensed_themes": ["main_theme1", "main_theme2"]}]}'
        ),
        MagicMock(content='{"responses": [{"topic_id": "label1", "topic": "desc1"}]}'),
        MagicMock(
            content=json.dumps(
                {
                    "responses": [
                        {
                            "response_id": 1,
                            "reason": "reason1",
                            "label": ["label1"],
                        },
                        {
                            "response_id": 2,
                            "reason": "reason2",
                            "label": ["label2"],
                        },
                    ]
                }
            )
        ),
    ]

    # Test with verbose=False (default)
    with patch.object(logger, "setLevel") as mock_set_level:
        await find_themes(sample_df, mock_llm, question="test question")
        # Should set to CRITICAL when verbose is False
        mock_set_level.assert_called_with(logging.CRITICAL)

    # Reset mock responses for second test
    mock_llm.ainvoke.side_effect = [
        MagicMock(
            content='{"responses": [{"response_id": 1, "position": "agreement", "text": "response1"}, {"response_id": 2, "position": "disagreement", "text": "response2"}]}'
        ),
        MagicMock(content='{"responses": [{"themes": ["theme1", "theme2"]}]}'),
        MagicMock(content='{"responses": [{"themes": ["theme3", "theme4"]}]}'),
        MagicMock(
            content='{"responses": [{"condensed_themes": ["main_theme1", "main_theme2"]}]}'
        ),
        MagicMock(content='{"responses": [{"topic_id": "label1", "topic": "desc1"}]}'),
        MagicMock(
            content=json.dumps(
                {
                    "responses": [
                        {
                            "response_id": 1,
                            "reason": "reason1",
                            "label": ["label1"],
                        },
                        {
                            "response_id": 2,
                            "reason": "reason2",
                            "label": ["label2"],
                        },
                    ]
                }
            )
        ),
    ]

    # Test with verbose=True
    with patch.object(logger, "setLevel") as mock_set_level:
        await find_themes(sample_df, mock_llm, question="test question", verbose=True)
        # Should set to INFO when verbose is True
        mock_set_level.assert_called_with(logging.INFO)
