import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.prompts import PromptTemplate

from themefinder import (
    find_themes,
    sentiment_analysis,
    theme_condensation,
    theme_generation,
    theme_mapping,
    theme_refinement,
)
from themefinder.core import theme_target_alignment
from themefinder.llm_batch_processor import batch_and_run


@pytest.mark.asyncio
async def test_batch_and_run_missing_id(mock_llm):
    """Test batch_and_run where the mocked return does not contain an expected id."""
    sample_df = pd.DataFrame(
        {"response_id": [1, 2], "response": ["response 1", "response 2"]}
    )
    mock_llm.ainvoke.side_effect = [
        # First Mock should contain 1 and 2 but doesn't
        MagicMock(
            content=json.dumps(
                {"responses": [{"response_id": 1, "position": "positive"}]}
            )
        ),
        # Next 2 are when batch size == 1
        MagicMock(
            content=json.dumps(
                {"responses": [{"response_id": 1, "position": "positive"}]}
            )
        ),
        MagicMock(
            content=json.dumps(
                {"responses": [{"response_id": 2, "position": "negative"}]}
            )
        ),
    ]
    result = await batch_and_run(
        responses_df=sample_df,
        prompt_template=PromptTemplate.from_template(
            template="this is a fake template"
        ),
        llm=mock_llm,
        batch_size=2,
        response_id_integrity_check=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert "response_id" in result.columns
    assert "position" in result.columns
    assert len(result) == 2
    assert 1 in result["response_id"].to_list()
    assert 2 in result["response_id"].to_list()
    assert mock_llm.ainvoke.call_count == 3


async def test_sentiment_analysis(mock_llm, sample_df):
    """Test sentiment analysis with mocked LLM responses."""
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "position": "positive"},
                    {"response_id": 2, "position": "negative"},
                ]
            }
        )
    )
    result = await sentiment_analysis(
        sample_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "position" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_generation(mock_llm, sample_sentiment_df):
    """Test theme generation with mocked LLM responses."""
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "themes": ["theme1", "theme2"]},
                    {"response_id": 2, "themes": ["theme3", "theme4"]},
                ]
            }
        )
    )
    result = await theme_generation(
        sample_sentiment_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "themes" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_condensation(mock_llm):
    """Test theme condensation with mocked LLM responses."""
    # Initial_df has 4 themes. Batch size is 2 so it will need to condense in two batches.
    initial_df = pd.DataFrame({"theme": [f"theme{i}" for i in range(1, 5)]})
    mock_llm.ainvoke.side_effect = [
        # Each batch of two condenses to 1 which is under the batch size.
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}]})),
        MagicMock(content=json.dumps({"responses": [{"theme": "B"}]})),
        # Final condesation goes down to 1.
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}]})),
    ]
    result = await theme_condensation(
        initial_df, mock_llm, question="test question", batch_size=2
    )

    assert isinstance(result, pd.DataFrame)
    assert "theme" in result.columns
    assert len(result) == 1
    assert mock_llm.ainvoke.call_count == 3


@pytest.mark.asyncio
async def test_theme_condensation_when_condensation_stops(mock_llm):
    """Test theme condensation with mocked LLM responses."""
    # Initial_df has 4 themes. Batch size is 2 so it will need to condense in two batches.
    initial_df = pd.DataFrame({"theme": [f"theme{i}" for i in range(1, 5)]})
    mock_llm.ainvoke.side_effect = [
        # Each batch of two doesn't condense.
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}, {"theme": "B"}]})),
        MagicMock(content=json.dumps({"responses": [{"theme": "C"}, {"theme": "D"}]})),
        # This should trigger the final attempt (which has two batches as it is still at 4)
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}, {"theme": "B"}]})),
        MagicMock(content=json.dumps({"responses": [{"theme": "C"}, {"theme": "D"}]})),
    ]
    result = await theme_condensation(
        initial_df, mock_llm, question="test question", batch_size=2
    )

    assert isinstance(result, pd.DataFrame)
    assert "theme" in result.columns
    assert len(result) == 4
    assert mock_llm.ainvoke.call_count == 4


@pytest.mark.asyncio
async def test_theme_refinement(mock_llm):
    """Test theme refinement with mocked LLM responses."""
    condensed_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"topic_id": "1", "topic": "refined_theme1"},
                    {"topic_id": "2", "topic": "refined_theme2"},
                ]
            }
        )
    )
    result = await theme_refinement(
        condensed_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "topic_id" in result.columns
    assert "topic" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_target_alignment(mock_llm):
    """Test theme target alignment with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"topic_id": "1", "topic": "aligned_theme1"},
                    {"topic_id": "2", "topic": "aligned_theme2"},
                ]
            }
        )
    )
    result = await theme_target_alignment(
        refined_df, mock_llm, question="test question", target_n_themes=2, batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "topic_id" in result.columns
    assert "topic" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_mapping(mock_llm, sample_sentiment_df):
    """Test theme mapping with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "reason": ["reason1"], "label": ["label1"]},
                    {"response_id": 2, "reason": ["reason2"], "label": ["label2"]},
                ]
            }
        )
    )
    result = await theme_mapping(
        sample_sentiment_df,
        mock_llm,
        question="test question",
        refined_themes_df=refined_df,
        batch_size=2,
    )
    assert isinstance(result, pd.DataFrame)
    assert "response_id" in result.columns
    assert "reason" in result.columns
    assert "label" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_find_themes(monkeypatch):
    # Dummy async functions returning simple DataFrames
    async def dummy_sentiment_analysis(responses_df, llm, question, system_prompt):
        return pd.DataFrame({"sentiment": ["positive"]})

    async def dummy_theme_generation(sentiment_df, llm, question, system_prompt):
        return pd.DataFrame({"theme": ["theme1"]})

    async def dummy_theme_condensation(theme_df, llm, question, system_prompt):
        return pd.DataFrame({"condensed_theme": ["condensed_theme1"]})

    async def dummy_theme_refinement(condensed_theme_df, llm, question, system_prompt):
        return pd.DataFrame({"refined_theme": ["refined_theme1"]})

    async def dummy_theme_target_alignment(
        refined_theme_df, llm, question, target_n_themes, system_prompt
    ):
        return pd.DataFrame({"refined_theme": [f"aligned_theme_for_{target_n_themes}"]})

    async def dummy_theme_mapping(
        sentiment_df, llm, question, refined_themes_df, system_prompt
    ):
        return pd.DataFrame({"mapping": ["mapped_theme1"]})

    # Patch the internal functions so that find_themes uses the dummy versions.
    monkeypatch.setattr("themefinder.core.sentiment_analysis", dummy_sentiment_analysis)
    monkeypatch.setattr("themefinder.core.theme_generation", dummy_theme_generation)
    monkeypatch.setattr("themefinder.core.theme_condensation", dummy_theme_condensation)
    monkeypatch.setattr("themefinder.core.theme_refinement", dummy_theme_refinement)
    monkeypatch.setattr(
        "themefinder.core.theme_target_alignment", dummy_theme_target_alignment
    )
    monkeypatch.setattr("themefinder.core.theme_mapping", dummy_theme_mapping)

    # Prepare a dummy responses DataFrame and parameters.
    responses_df = pd.DataFrame({"response": ["This is a test response"]})
    dummy_llm = object()  # Dummy LLM; our dummy functions don't depend on it.
    question = "Test question"
    target_n_themes = 3
    system_prompt = "Dummy system prompt"
    verbose = False

    result = await find_themes(
        responses_df,
        dummy_llm,
        question,
        target_n_themes=target_n_themes,
        system_prompt=system_prompt,
        verbose=verbose,
    )

    # Verify that the returned dictionary contains the expected keys.
    expected_keys = {
        "question",
        "sentiment",
        "themes",
        "condensed_themes",
        "refined_themes",
        "mapping",
    }
    assert set(result.keys()) == expected_keys

    # Check that the question is correctly passed through.
    assert result["question"] == question

    # Verify that each stage returns a DataFrame with content.
    for key in ["sentiment", "themes", "condensed_themes", "refined_themes", "mapping"]:
        df = result[key]
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
