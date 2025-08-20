"""Tests for CrossCuttingThemesAgent."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from themefinder.models import (
    CrossCuttingThemeIdentificationResponse,
    CrossCuttingThemeDefinition,
    CrossCuttingThemeMappingResponse,
    CrossCuttingThemeMapping,
)
from themefinder.advanced_tasks.cross_cutting_themes_agent import (
    CrossCuttingThemesAgent,
)


@pytest.fixture
def sample_questions_themes():
    """Create sample questions_themes data for testing."""
    return {
        1: pd.DataFrame(
            [
                {"topic_id": "A", "topic": "Employment Support: Help finding jobs"},
                {
                    "topic_id": "B",
                    "topic": "Financial Assistance: Support with benefits",
                },
                {
                    "topic_id": "C",
                    "topic": "Mental Health: Access to mental health services",
                },
            ]
        ),
        2: pd.DataFrame(
            [
                {
                    "topic_id": "A",
                    "topic": "Housing Support: Affordable housing solutions",
                },
                {
                    "topic_id": "B",
                    "topic": "Employment Training: Skills development programs",
                },
                {
                    "topic_id": "C",
                    "topic": "Healthcare Access: Improved medical services",
                },
            ]
        ),
        3: pd.DataFrame(
            [
                {
                    "topic_id": "A",
                    "topic": "Job Placement: Direct job placement assistance",
                },
                {
                    "topic_id": "B",
                    "topic": "Mental Wellbeing: Mental health support programs",
                },
                {"topic_id": "C", "topic": "Transport: Public transport improvements"},
            ]
        ),
    }


@pytest.fixture
def sample_question_strings():
    """Create sample question strings for testing."""
    return {
        "1": "What employment support do you need?",
        "2": "What housing support would be helpful?",
        "3": "What additional services are important?",
    }


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MagicMock()


class TestCrossCuttingThemesAgent:
    """Test cases for CrossCuttingThemesAgent class."""

    def test_agent_initialization(self, sample_questions_themes, mock_llm):
        """Test agent initialization."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=5
        )

        assert agent.llm == mock_llm
        assert agent.questions_themes == sample_questions_themes
        assert agent.n_concepts == 5
        assert agent.total_themes == 9  # 3 questions * 3 themes each
        assert agent.concepts == []
        assert agent.concept_assignments == {}

    def test_agent_initialization_with_question_strings(
        self, sample_questions_themes, sample_question_strings, mock_llm
    ):
        """Test agent initialization with question strings."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm,
            questions_themes=sample_questions_themes,
            question_strings=sample_question_strings,
            n_concepts=3,
        )

        assert agent.question_strings == sample_question_strings
        assert agent.n_concepts == 3

    def test_empty_questions_themes_raises_error(self, mock_llm):
        """Test that empty questions_themes raises ValueError."""
        with pytest.raises(ValueError, match="questions_themes cannot be empty"):
            CrossCuttingThemesAgent(llm=mock_llm, questions_themes={}, n_concepts=5)

    def test_format_questions_and_themes(
        self, sample_questions_themes, sample_question_strings, mock_llm
    ):
        """Test formatting questions and themes for prompts."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm,
            questions_themes=sample_questions_themes,
            question_strings=sample_question_strings,
            n_concepts=5,
        )

        formatted = agent._format_questions_and_themes()

        # Check that all questions are included
        assert "What employment support do you need?" in formatted
        assert "What housing support would be helpful?" in formatted
        assert "What additional services are important?" in formatted

        # Check that themes are included
        assert "Employment Support: Help finding jobs" in formatted
        assert "Housing Support: Affordable housing solutions" in formatted
        assert "Transport: Public transport improvements" in formatted

    def test_format_questions_and_themes_without_strings(
        self, sample_questions_themes, mock_llm
    ):
        """Test formatting when question strings are not provided."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=5
        )

        formatted = agent._format_questions_and_themes()

        # Should use default Question format
        assert "Question 1" in formatted
        assert "Question 2" in formatted
        assert "Question 3" in formatted

    def test_identify_concepts(self, sample_questions_themes, mock_llm):
        """Test concept identification."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=2
        )

        # Mock the structured LLM response
        mock_response = CrossCuttingThemeIdentificationResponse(
            themes=[
                CrossCuttingThemeDefinition(
                    name="Employment and Training",
                    description="Themes related to employment support and training programs.",
                ),
                CrossCuttingThemeDefinition(
                    name="Health and Wellbeing",
                    description="Themes related to mental health and healthcare access.",
                ),
            ]
        )

        mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

        concepts = agent.identify_concepts()

        assert len(concepts) == 2
        assert concepts[0]["name"] == "Employment and Training"
        assert concepts[1]["name"] == "Health and Wellbeing"
        assert agent.concepts == concepts

    def test_map_themes_to_concepts_without_concepts_raises_error(
        self, sample_questions_themes, mock_llm
    ):
        """Test that mapping themes without identifying concepts first raises error."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=2
        )

        with pytest.raises(ValueError, match="Must call identify_concepts\\(\\) first"):
            agent.map_themes_to_concepts()

    def test_map_themes_to_concepts(self, sample_questions_themes, mock_llm):
        """Test mapping themes to concepts."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=2
        )

        # Set up concepts
        agent.concepts = [
            {"name": "Employment Support", "description": "Employment themes"},
            {"name": "Health Services", "description": "Health and wellbeing themes"},
        ]

        # Mock the mapping responses for each question
        mapping_responses = [
            CrossCuttingThemeMappingResponse(
                mappings=[
                    CrossCuttingThemeMapping(
                        theme_name="Employment Support", theme_ids=["A"]
                    ),
                    CrossCuttingThemeMapping(
                        theme_name="Health Services", theme_ids=["C"]
                    ),
                ]
            ),
            CrossCuttingThemeMappingResponse(
                mappings=[
                    CrossCuttingThemeMapping(
                        theme_name="Employment Support", theme_ids=["B"]
                    ),
                    CrossCuttingThemeMapping(
                        theme_name="Health Services", theme_ids=["C"]
                    ),
                ]
            ),
            CrossCuttingThemeMappingResponse(
                mappings=[
                    CrossCuttingThemeMapping(
                        theme_name="Employment Support", theme_ids=["A"]
                    ),
                    CrossCuttingThemeMapping(
                        theme_name="Health Services", theme_ids=["B"]
                    ),
                ]
            ),
        ]

        mock_llm.with_structured_output.return_value.invoke.side_effect = (
            mapping_responses
        )

        assignments = agent.map_themes_to_concepts()

        # Check that assignments were created
        assert "Employment Support" in assignments
        assert "Health Services" in assignments

        # Check Employment Support assignments
        employment_assignments = assignments["Employment Support"]
        assert len(employment_assignments) == 3

        # Check Health Services assignments
        health_assignments = assignments["Health Services"]
        assert len(health_assignments) == 3

    def test_analyze_full_workflow(self, sample_questions_themes, mock_llm):
        """Test the complete analyze workflow."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=2
        )

        # Mock concept identification response
        concepts_response = CrossCuttingThemeIdentificationResponse(
            themes=[
                CrossCuttingThemeDefinition(
                    name="Employment Support",
                    description="Employment and training themes.",
                ),
            ]
        )

        # Mock mapping response
        mapping_response = CrossCuttingThemeMappingResponse(
            mappings=[
                CrossCuttingThemeMapping(
                    theme_name="Employment Support", theme_ids=["A", "B"]
                ),
            ]
        )

        mock_llm.with_structured_output.return_value.invoke.side_effect = [
            concepts_response,
            mapping_response,
            mapping_response,
            mapping_response,
        ]

        results = agent.analyze()

        assert "concepts" in results
        assert "assignments" in results
        assert len(results["concepts"]) == 1
        assert "Employment Support" in results["assignments"]

    def test_get_results_as_dataframe(self, sample_questions_themes, mock_llm):
        """Test converting results to DataFrame."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=2
        )

        # Set up mock data
        agent.concepts = [
            {"name": "Employment Support", "description": "Employment themes"},
        ]
        agent.concept_assignments = {
            "Employment Support": [
                {
                    "question_id": 1,
                    "theme_id": "A",
                    "theme_text": "Employment Support: Help finding jobs",
                },
                {
                    "question_id": 2,
                    "theme_id": "B",
                    "theme_text": "Employment Training: Skills development programs",
                },
            ]
        }

        df = agent.get_results_as_dataframe()

        assert len(df) == 1
        assert df.iloc[0]["name"] == "Employment Support"
        assert df.iloc[0]["n_themes"] == 2
        assert df.iloc[0]["n_questions"] == 2
        assert 1 in df.iloc[0]["themes"]
        assert 2 in df.iloc[0]["themes"]

    def test_get_statistics(self, sample_questions_themes, mock_llm):
        """Test getting analysis statistics."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=2
        )

        # Set up mock data
        agent.concepts = [
            {"name": "Employment Support", "description": "Employment themes"},
            {"name": "Health Services", "description": "Health themes"},
        ]
        agent.concept_assignments = {
            "Employment Support": [
                {"question_id": 1, "theme_id": "A", "theme_text": "Theme A"},
                {"question_id": 2, "theme_id": "B", "theme_text": "Theme B"},
            ],
            "Health Services": [
                {"question_id": 1, "theme_id": "C", "theme_text": "Theme C"},
            ],
        }

        stats = agent.get_statistics()

        assert stats["total_themes"] == 9  # 3 questions * 3 themes each
        assert stats["used_themes"] == 3  # 3 unique (question_id, theme_id) pairs
        assert stats["unused_themes"] == 6  # 9 - 3
        assert stats["utilization_rate"] == 3 / 9
        assert stats["n_concepts"] == 2
        assert stats["n_questions"] == 3
        assert stats["concepts_with_themes"] == 2  # Both concepts have themes

    def test_refine_concept_descriptions_without_mapping_raises_error(
        self, sample_questions_themes, mock_llm
    ):
        """Test that refining descriptions without mapping first raises error."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=2
        )

        with pytest.raises(
            ValueError, match="Must call map_themes_to_concepts\\(\\) first"
        ):
            agent.refine_concept_descriptions()

    def test_refine_concept_descriptions(self, sample_questions_themes, mock_llm):
        """Test refining concept descriptions."""
        agent = CrossCuttingThemesAgent(
            llm=mock_llm, questions_themes=sample_questions_themes, n_concepts=1
        )

        # Set up mock data
        agent.concepts = [
            {"name": "Employment Support", "description": "Original description"},
        ]
        agent.concept_assignments = {
            "Employment Support": [
                {
                    "question_id": 1,
                    "theme_id": "A",
                    "theme_text": "Employment Support: Help finding jobs",
                },
                {
                    "question_id": 2,
                    "theme_id": "B",
                    "theme_text": "Employment Training: Skills development programs",
                },
            ]
        }

        # Mock LLM response for description refinement
        mock_response = MagicMock()
        mock_response.content = "Refined description based on assigned themes."
        mock_llm.invoke.return_value = mock_response

        descriptions = agent.refine_concept_descriptions()

        assert "Employment Support" in descriptions
        assert (
            descriptions["Employment Support"]
            == "Refined description based on assigned themes."
        )
        assert agent.concept_descriptions == descriptions
