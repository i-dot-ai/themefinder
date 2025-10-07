import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.prompts import PromptTemplate

from themefinder import (
    sentiment_analysis,
    theme_condensation,
    theme_generation,
    theme_mapping,
    theme_refinement,
    theme_target_alignment,
    find_themes,
)
from themefinder.llm_batch_processor import batch_and_run
from themefinder.models import (
    CondensedTheme,
    Position,
    SentimentAnalysisOutput,
    SentimentAnalysisResponses,
    Theme,
    ThemeCondensationResponses,
    ThemeGenerationResponses,
    ThemeMappingOutput,
    ThemeMappingResponses,
)


@pytest.mark.asyncio
async def test_batch_and_run_missing_id(mock_llm, sample_df):
    """Test batch_and_run where the mocked return does not contain an expected id."""
    first_response = [{"response_id": 1, "position": "positive"}]
    retry_response = [{"response_id": 2, "position": "negative"}]

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.side_effect = [(first_response, [2]), (retry_response, [])]

        result, failed_df = await batch_and_run(
            input_df=sample_df,
            prompt_template=PromptTemplate.from_template(
                template="this is a fake template"
            ),
            llm=mock_llm,
            batch_size=2,
            integrity_check=True,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 1 in result["response_id"].to_list()
        assert 2 in result["response_id"].to_list()
        assert mock_call_llm.await_count == 2
        assert failed_df.empty


@pytest.mark.asyncio
async def test_sentiment_analysis(mock_llm, sample_df):
    """Test sentiment analysis with mocked LLM responses."""
    mock_response = SentimentAnalysisResponses(
        responses=[
            SentimentAnalysisOutput(response_id=1, position=Position.AGREEMENT),
            SentimentAnalysisOutput(response_id=2, position=Position.DISAGREEMENT),
        ]
    )

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.return_value = (
            [
                mock_response.responses[0].model_dump(),
                mock_response.responses[1].model_dump(),
            ],
            [],
        )

        result, _ = await sentiment_analysis(
            sample_df, mock_llm, question="test question", batch_size=2
        )

        assert isinstance(result, pd.DataFrame)
        assert "position" in result.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_theme_generation(mock_llm, sample_sentiment_df):
    """Test theme generation with mocked LLM responses."""
    mock_themes = ThemeGenerationResponses(
        responses=[
            Theme(
                topic_label="Theme 1",
                topic_description="Description of theme 1",
                position=Position.AGREEMENT,
            ),
            Theme(
                topic_label="Theme 2",
                topic_description="Description of theme 2",
                position=Position.DISAGREEMENT,
            ),
        ]
    )

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.return_value = (
            [theme.model_dump() for theme in mock_themes.responses],
            [],
        )

        result, _ = await theme_generation(
            sample_sentiment_df, mock_llm, question="test question", batch_size=2
        )

        assert isinstance(result, pd.DataFrame)
        assert "topic_label" in result.columns
        assert "topic_description" in result.columns
        assert "position" in result.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_theme_condensation_basic(mock_llm, sample_themes_df):
    """Test theme condensation with a single batch."""
    mock_condensed_themes = ThemeCondensationResponses(
        responses=[
            CondensedTheme(
                topic_label="Combined Theme AB",
                topic_description="Combined description of A and B",
                source_topic_count=8,
            ),
            CondensedTheme(
                topic_label="Combined Theme CD",
                topic_description="Combined description of C and D",
                source_topic_count=9,
            ),
        ]
    )

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.return_value = (
            [theme.model_dump() for theme in mock_condensed_themes.responses],
            [],
        )

        result_df, errors_df = await theme_condensation(
            sample_themes_df,
            mock_llm,
            question="What are your thoughts on this product?",
            batch_size=10,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "topic_label" in result_df.columns
        assert "topic_description" in result_df.columns
        assert "source_topic_count" in result_df.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_theme_condensation_recursive(mock_llm):
    """Test recursive theme condensation when the number of themes exceeds batch size."""
    large_themes_df = pd.DataFrame(
        {
            "topic_label": [f"Theme {i}" for i in range(100)],
            "topic_description": [f"Description of theme {i}" for i in range(100)],
            "source_topic_count": [i + 1 for i in range(100)],
        }
    )
    large_themes_df["response_id"] = range(1, len(large_themes_df) + 1)

    first_batch_responses = [
        {
            "topic_label": f"Combined Theme {i}",
            "topic_description": f"Combined description {i}",
            "source_topic_count": i * 2 + 3,
        }
        for i in range(50)
    ]

    second_batch_responses = [
        {
            "topic_label": f"Condensed Theme {i}",
            "topic_description": f"Condensed description {i}",
            "source_topic_count": i * 4 + 6,
        }
        for i in range(25)
    ]

    final_batch_responses = second_batch_responses.copy()

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.side_effect = [
            (first_batch_responses, []),
            (second_batch_responses, []),
            (final_batch_responses, []),
        ]

        result_df, errors_df = await theme_condensation(
            large_themes_df,
            mock_llm,
            question="What are your thoughts on this product?",
            batch_size=30,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 25
        assert mock_call_llm.await_count == 3


@pytest.mark.asyncio
async def test_theme_condensation_no_further_reduction(mock_llm):
    """Test when themes can't be condensed further."""
    themes_df = pd.DataFrame(
        {
            "topic_label": ["Theme A", "Theme B", "Theme C"],
            "topic_description": ["Desc A", "Desc B", "Desc C"],
            "source_topic_count": [5, 3, 7],
        }
    )
    themes_df["response_id"] = range(1, len(themes_df) + 1)

    original_responses = [
        {
            "topic_label": "Theme A",
            "topic_description": "Desc A",
            "source_topic_count": 5,
        },
        {
            "topic_label": "Theme B",
            "topic_description": "Desc B",
            "source_topic_count": 3,
        },
        {
            "topic_label": "Theme C",
            "topic_description": "Desc C",
            "source_topic_count": 7,
        },
    ]

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.side_effect = [(original_responses, []), (original_responses, [])]

        result_df, _ = await theme_condensation(
            themes_df, mock_llm, question="test question", batch_size=2
        )

        assert mock_call_llm.await_count == 2
        assert len(result_df) == 3
        original_labels = set(themes_df["topic_label"])
        result_labels = set(result_df["topic_label"])
        assert original_labels == result_labels


@pytest.mark.asyncio
async def test_theme_refinement(mock_llm):
    """Test theme refinement with mocked LLM responses."""
    condensed_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})

    mock_response = {
        "responses": [
            {"topic_id": "1", "topic": "refined_theme1"},
            {"topic_id": "2", "topic": "refined_theme2"},
        ]
    }

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.return_value = (
            [mock_response["responses"][0], mock_response["responses"][1]],
            [],
        )

        result, _ = await theme_refinement(
            condensed_df, mock_llm, question="test question", batch_size=2
        )

        assert isinstance(result, pd.DataFrame)
        assert "topic_id" in result.columns
        assert "topic" in result.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_theme_target_alignment(mock_llm):
    """Test theme target alignment with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})

    mock_response = {
        "responses": [
            {"topic_id": "1", "topic": "aligned_theme1"},
            {"topic_id": "2", "topic": "aligned_theme2"},
        ]
    }

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.return_value = (
            [mock_response["responses"][0], mock_response["responses"][1]],
            [],
        )

        result, _ = await theme_target_alignment(
            refined_df,
            mock_llm,
            question="test question",
            target_n_themes=2,
            batch_size=2,
        )

        assert isinstance(result, pd.DataFrame)
        assert "topic_id" in result.columns
        assert "topic" in result.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_theme_mapping(mock_llm, sample_sentiment_df):
    """Test theme mapping with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})

    mock_response = ThemeMappingResponses(
        responses=[
            ThemeMappingOutput(
                response_id=1,
                labels=["label1"],
            ),
            ThemeMappingOutput(
                response_id=2,
                labels=["label2"],
            ),
        ]
    )

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.return_value = (
            [
                mock_response.responses[0].model_dump(),
                mock_response.responses[1].model_dump(),
            ],
            [],
        )

        result, _ = await theme_mapping(
            sample_sentiment_df,
            mock_llm,
            question="test question",
            refined_themes_df=refined_df,
            batch_size=2,
        )

        assert isinstance(result, pd.DataFrame)
        assert "response_id" in result.columns
        assert "labels" in result.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_find_themes(mock_llm, sample_df):
    """Test find_themes with mocked LLM responses."""
    input_df = sample_df.copy()
    input_df = input_df.rename(columns={"text": "response"})

    sentiment_responses = [
        SentimentAnalysisOutput(
            response_id=1, position=Position.AGREEMENT
        ).model_dump(),
        SentimentAnalysisOutput(
            response_id=2, position=Position.DISAGREEMENT
        ).model_dump(),
    ]

    theme_generation_responses = [
        Theme(
            topic_label="Theme 1",
            topic_description="Description 1",
            position=Position.AGREEMENT,
        ).model_dump(),
        Theme(
            topic_label="Theme 2",
            topic_description="Description 2",
            position=Position.DISAGREEMENT,
        ).model_dump(),
    ]

    theme_condensation_responses = [
        CondensedTheme(
            topic_label="Condensed Theme",
            topic_description="Combined description",
            source_topic_count=2,
        ).model_dump()
    ]

    theme_refinement_responses = [{"topic_id": "1", "topic": "Refined Theme"}]

    theme_alignment_responses = [{"topic_id": "1", "topic": "Aligned Theme"}]

    theme_mapping_responses = [
        ThemeMappingOutput(
            response_id=1, reasons=["reason1"], labels=["label1"], stances=["POSITIVE"]
        ).model_dump(),
        ThemeMappingOutput(
            response_id=2, reasons=["reason2"], labels=["label2"], stances=["NEGATIVE"]
        ).model_dump(),
    ]

    detail_detection_responses = [
        {"response_id": 1, "detailed": True, "explanation": "Detailed explanation"},
        {
            "response_id": 2,
            "detailed": False,
            "explanation": "Not detailed explanation",
        },
    ]

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.side_effect = [
            (sentiment_responses, []),
            (theme_generation_responses, []),
            (theme_condensation_responses, []),
            (theme_refinement_responses, []),
            (theme_alignment_responses, []),
            (theme_mapping_responses, []),
            (detail_detection_responses, []),
        ]

        result = await find_themes(
            input_df,
            mock_llm,
            question="test question",
            target_n_themes=1,
            verbose=False,
        )

        assert mock_call_llm.await_count == 7

        expected_keys = [
            "question",
            "sentiment",
            "themes",
            "mapping",
            "detailed_responses",
            "unprocessables",
        ]
        assert all(key in result for key in expected_keys)

        assert result["question"] == "test question"

        mock_call_llm.reset_mock()

        mock_call_llm.side_effect = [
            (sentiment_responses, []),
            (theme_generation_responses, []),
            (theme_condensation_responses, []),
            (theme_refinement_responses, []),
            (theme_mapping_responses, []),
            (detail_detection_responses, []),
        ]

        _ = await find_themes(
            input_df,
            mock_llm,
            question="test question",
            target_n_themes=None,
            verbose=False,
        )

        assert mock_call_llm.await_count == 6
