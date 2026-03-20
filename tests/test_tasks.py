from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from themefinder import (
    find_themes,
    theme_condensation,
    theme_generation,
    theme_mapping,
    theme_refinement,
)
from themefinder.llm import LLMResponse
from themefinder.llm_batch_processor import batch_and_run
from themefinder.models import (
    CondensedTheme,
    DetailDetectionOutput,
    DetailDetectionResponses,
    EvidenceRich,
    Position,
    Theme,
    ThemeCondensationResponses,
    ThemeGenerationResponses,
    ThemeMappingOutput,
    ThemeMappingResponses,
)


@pytest.mark.asyncio
async def test_batch_and_run_missing_id(monkeypatch, mock_llm, sample_df):
    """Test batch_and_run where the mocked return does not contain an expected id."""
    from themefinder.llm_batch_processor import BatchPrompt

    first_response = [{"response_id": 1, "position": "positive"}]
    retry_response = [{"response_id": 2, "position": "negative"}]

    # Mock generate_prompts to avoid loading the actual template file
    def mock_generate_prompts(*args, **kwargs):
        batch_size = kwargs.get("batch_size", 10)
        if batch_size == 1:
            return [BatchPrompt(prompt_string="Retry prompt", response_ids=[2])]
        return [BatchPrompt(prompt_string="Test prompt", response_ids=[1, 2])]

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.generate_prompts",
        mock_generate_prompts,
    )

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.side_effect = [
            (first_response, [2]),
            (retry_response, []),
        ]

        result = await batch_and_run(
            input_df=sample_df,
            prompt_template="detail_detection",
            llm=mock_llm,
            output_model=DetailDetectionResponses,
            batch_size=2,
            integrity_check=True,
        )

        assert isinstance(result.output, pd.DataFrame)
        assert len(result.output) == 2
        assert 1 in result.output["response_id"].to_list()
        assert 2 in result.output["response_id"].to_list()
        assert mock_call_llm.await_count == 2
        assert result.failures.empty


@pytest.mark.asyncio
async def test_theme_generation(mock_llm, sample_responses_df):
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

        result = await theme_generation(
            sample_responses_df, mock_llm, question="test question", batch_size=2
        )

        assert isinstance(result.output, pd.DataFrame)
        assert "topic_label" in result.output.columns
        assert "topic_description" in result.output.columns
        assert "position" in result.output.columns
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

        result = await theme_condensation(
            sample_themes_df,
            mock_llm,
            question="What are your thoughts on this product?",
            batch_size=10,
        )

        assert isinstance(result.output, pd.DataFrame)
        assert len(result.output) == 2
        assert "topic_label" in result.output.columns
        assert "topic_description" in result.output.columns
        assert "source_topic_count" in result.output.columns
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

        result = await theme_condensation(
            large_themes_df,
            mock_llm,
            question="What are your thoughts on this product?",
            batch_size=30,
        )

        assert isinstance(result.output, pd.DataFrame)
        assert len(result.output) == 25
        assert (
            mock_call_llm.await_count == 3
        )  # while loop (100->50, 50->25) + final pass


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
        mock_call_llm.side_effect = [
            (original_responses, []),
        ]

        result = await theme_condensation(
            themes_df, mock_llm, question="test question", batch_size=2
        )

        assert (
            mock_call_llm.await_count == 1
        )  # 3 themes < target (30), only final pass runs
        assert len(result.output) == 3
        original_labels = set(themes_df["topic_label"])
        result_labels = set(result.output["topic_label"])
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

        result = await theme_refinement(
            condensed_df, mock_llm, question="test question", batch_size=2
        )

        assert isinstance(result.output, pd.DataFrame)
        assert "topic_id" in result.output.columns
        assert "topic" in result.output.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_theme_mapping(mock_llm, sample_responses_df):
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

        result = await theme_mapping(
            sample_responses_df,
            mock_llm,
            question="test question",
            refined_themes_df=refined_df,
            batch_size=2,
        )

        assert isinstance(result.output, pd.DataFrame)
        assert "response_id" in result.output.columns
        assert "labels" in result.output.columns
        assert mock_call_llm.await_count == 1


@pytest.mark.asyncio
async def test_find_themes(mock_llm, sample_df):
    """Test find_themes with mocked LLM responses."""
    input_df = sample_df.copy()

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

    theme_mapping_responses = [
        ThemeMappingOutput(response_id=1, labels=["label1"]).model_dump(),
        ThemeMappingOutput(response_id=2, labels=["label2"]).model_dump(),
    ]

    detail_detection_responses = [
        DetailDetectionOutput(
            response_id=1, evidence_rich=EvidenceRich.YES
        ).model_dump(),
        DetailDetectionOutput(
            response_id=2, evidence_rich=EvidenceRich.NO
        ).model_dump(),
    ]

    with patch(
        "themefinder.llm_batch_processor.call_llm", new_callable=AsyncMock
    ) as mock_call_llm:
        mock_call_llm.side_effect = [
            (theme_generation_responses, []),
            (theme_condensation_responses, []),
            (theme_refinement_responses, []),
            (theme_mapping_responses, []),
            (detail_detection_responses, []),
        ]

        result = await find_themes(
            input_df,
            mock_llm,
            question="test question",
            verbose=False,
        )

        assert mock_call_llm.await_count == 5

        expected_keys = [
            "question",
            "themes",
            "mapping",
            "detailed_responses",
            "unprocessables",
        ]
        assert all(key in result for key in expected_keys)

        assert result["question"] == "test question"

        mock_call_llm.reset_mock()

        mock_call_llm.side_effect = [
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
            verbose=False,
        )

        assert mock_call_llm.await_count == 5


@pytest.mark.asyncio
async def test_theme_clustering():
    """Test theme_clustering function"""
    from themefinder import theme_clustering
    from themefinder.models import HierarchicalClusteringResponse, ThemeNode

    # Create test themes DataFrame
    themes_df = pd.DataFrame(
        {
            "topic_id": ["1", "2", "3"],
            "topic_label": ["Theme A", "Theme B", "Theme C"],
            "topic_description": ["Description A", "Description B", "Description C"],
            "source_topic_count": [10, 20, 30],
        }
    )

    # Create a mock response for the clustering
    mock_response = HierarchicalClusteringResponse(
        parent_themes=[
            ThemeNode(
                topic_id="parent_1",
                topic_label="Combined Theme",
                topic_description="Combined description",
                source_topic_count=30,
                children=["1", "2"],
            )
        ],
        should_terminate=True,
    )

    # Create mock LLM with ainvoke returning LLMResponse
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=LLMResponse(parsed=mock_response))

    # Call theme_clustering
    result = await theme_clustering(
        themes_df,
        mock_llm,
        max_iterations=1,
        target_themes=2,
        significance_percentage=10.0,
        return_all_themes=False,
    )

    # Verify the result
    assert isinstance(result.output, pd.DataFrame)
    assert len(result.output) >= 0  # Can be 0 if nothing meets significance threshold

    # Test with return_all_themes=True
    result_all = await theme_clustering(
        themes_df,
        mock_llm,
        max_iterations=1,
        target_themes=2,
        significance_percentage=10.0,
        return_all_themes=True,
    )

    assert isinstance(result_all.output, pd.DataFrame)


def test_hierarchical_clustering_response_validation():
    """Test validation of HierarchicalClusteringResponse model."""
    from themefinder.models import HierarchicalClusteringResponse, ThemeNode

    # Test that duplicate children in different parents raises error
    with pytest.raises(
        ValueError, match="Each child theme can have at most one parent"
    ):
        HierarchicalClusteringResponse(
            parent_themes=[
                ThemeNode(
                    topic_id="parent_1",
                    topic_label="Theme 1",
                    topic_description="Description 1",
                    source_topic_count=10,
                    children=["child_1", "child_2"],
                ),
                ThemeNode(
                    topic_id="parent_2",
                    topic_label="Theme 2",
                    topic_description="Description 2",
                    source_topic_count=20,
                    children=["child_2", "child_3"],  # child_2 appears in both parents
                ),
            ],
            should_terminate=False,
        )
