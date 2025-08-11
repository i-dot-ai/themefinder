"""Tests for cross_cutting_themes module."""

import json
from unittest.mock import MagicMock, patch
from typing import List, Dict

import pandas as pd
import pytest

from themefinder.models import (
    CrossCuttingThemesResponse,
    CrossCuttingTheme,
    ConstituentTheme,
    CrossCuttingThemeReviewResponse,
    CrossCuttingThemeAddition,
)
from themefinder.advanced_tasks.cross_cutting_themes import (
    analyze_cross_cutting_themes,
    _format_themes_for_prompt,
    _get_used_themes,
    _get_unused_themes,
    _step1_identify_cross_cutting_themes,
    _step2_review_unused_themes,
)


@pytest.fixture
def sample_questions_themes():
    """Create sample questions_themes data for testing."""
    return {
        1: pd.DataFrame([
            {"topic_id": "A", "topic": "Employment Support: Help finding and maintaining jobs"},
            {"topic_id": "B", "topic": "Financial Assistance: Support with benefits and costs"},
            {"topic_id": "C", "topic": "Mental Health: Access to mental health services"},
        ]),
        2: pd.DataFrame([
            {"topic_id": "A", "topic": "Housing Support: Affordable housing solutions"},
            {"topic_id": "B", "topic": "Employment Training: Skills development programs"},
            {"topic_id": "C", "topic": "Healthcare Access: Improved medical services"},
        ]),
        3: pd.DataFrame([
            {"topic_id": "A", "topic": "Job Placement: Direct job placement assistance"},
            {"topic_id": "B", "topic": "Mental Wellbeing: Mental health support programs"},
            {"topic_id": "C", "topic": "Transport: Public transport improvements"},
        ]),
    }


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    return llm


def test_format_themes_for_prompt(sample_questions_themes):
    """Test formatting themes data for LLM prompt."""
    result = _format_themes_for_prompt(sample_questions_themes)
    
    # Check that all themes are included
    assert "Question 1, Theme A: Employment Support - Help finding and maintaining jobs" in result
    assert "Question 2, Theme B: Employment Training - Skills development programs" in result
    assert "Question 3, Theme C: Transport - Public transport improvements" in result
    
    # Check proper formatting
    lines = result.split("\n")
    assert len(lines) == 9  # 3 questions * 3 themes each


def test_format_themes_handles_missing_colon(sample_questions_themes):
    """Test formatting themes when topic doesn't contain colon."""
    # Modify one theme to not have a colon
    sample_questions_themes[1].loc[0, "topic"] = "Simple Theme"
    
    result = _format_themes_for_prompt(sample_questions_themes)
    
    # Should handle gracefully by using same text for label and description
    assert "Question 1, Theme A: Simple Theme - Simple Theme" in result


def test_get_used_themes():
    """Test extracting used themes from groups."""
    groups = [
        {
            "name": "Employment Support",
            "description": "Themes about employment",
            "themes": [
                {"question_number": 1, "theme_key": "A"},
                {"question_number": 2, "theme_key": "B"},
            ]
        },
        {
            "name": "Mental Health",
            "description": "Mental health themes",
            "themes": [
                {"question_number": 1, "theme_key": "C"},
                {"question_number": 3, "theme_key": "B"},
            ]
        }
    ]
    
    used = _get_used_themes(groups)
    
    assert (1, "A") in used
    assert (2, "B") in used
    assert (1, "C") in used
    assert (3, "B") in used
    assert len(used) == 4


def test_get_unused_themes(sample_questions_themes):
    """Test getting unused themes."""
    used_themes = {(1, "A"), (2, "B"), (3, "C")}
    
    unused = _get_unused_themes(sample_questions_themes, used_themes)
    
    # Should have 9 total - 3 used = 6 unused
    assert len(unused) == 6
    
    # Check structure of unused theme
    theme_keys = [(t["question_number"], t["theme_key"]) for t in unused]
    assert (1, "B") in theme_keys
    assert (1, "C") in theme_keys
    assert (2, "A") in theme_keys
    assert (2, "C") in theme_keys
    assert (3, "A") in theme_keys
    assert (3, "B") in theme_keys
    
    # Check that used themes are not in unused
    assert (1, "A") not in theme_keys
    assert (2, "B") not in theme_keys
    assert (3, "C") not in theme_keys


@patch('themefinder.advanced_tasks.cross_cutting_themes.load_prompt_from_file')
def test_step1_identify_cross_cutting_themes(mock_load_prompt, mock_llm):
    """Test Step 1: identifying initial cross-cutting themes."""
    # Setup mock prompt
    mock_load_prompt.return_value = "{system_prompt}\n{themes_data}"
    
    # Setup mock LLM response
    mock_response = CrossCuttingThemesResponse(
        cross_cutting_themes=[
            CrossCuttingTheme(
                name="Employment Support",
                description="Themes about employment assistance",
                themes=[
                    ConstituentTheme(question_number=1, theme_key="A"),
                    ConstituentTheme(question_number=2, theme_key="B"),
                    ConstituentTheme(question_number=3, theme_key="A"),
                ]
            ),
            CrossCuttingTheme(
                name="Mental Health",
                description="Mental health support themes",
                themes=[
                    ConstituentTheme(question_number=1, theme_key="C"),
                    ConstituentTheme(question_number=3, theme_key="B"),
                ]
            )
        ]
    )
    
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
    
    themes_data = "Question 1, Theme A: Employment Support\nQuestion 2, Theme B: Training"
    result = _step1_identify_cross_cutting_themes(themes_data, mock_llm, "system prompt")
    
    assert len(result) == 2
    assert result[0]["name"] == "Employment Support"
    assert len(result[0]["themes"]) == 3
    assert result[1]["name"] == "Mental Health"
    assert len(result[1]["themes"]) == 2


@patch('themefinder.advanced_tasks.cross_cutting_themes.load_prompt_from_file')
def test_step2_review_unused_themes(mock_load_prompt, mock_llm):
    """Test Step 2: reviewing unused themes."""
    # Setup initial groups
    initial_groups = [
        {
            "name": "Employment Support",
            "description": "Employment themes",
            "themes": [
                {"question_number": 1, "theme_key": "A"},
                {"question_number": 3, "theme_key": "A"},
            ]
        }
    ]
    
    # Setup unused themes
    unused_themes = [
        {
            "question_number": 2,
            "theme_key": "B",
            "label": "Employment Training",
            "description": "Skills development"
        }
    ]
    
    # Setup mock prompt
    mock_load_prompt.return_value = "{system_prompt}\n{existing_themes}\n{unused_themes}"
    
    # Setup mock LLM response - adding theme from Q2
    mock_response = CrossCuttingThemeReviewResponse(
        additions=[
            CrossCuttingThemeAddition(
                cross_cutting_theme_name="Employment Support",
                question_number=2,
                theme_key="B",
                justification="Employment training fits with employment support"
            )
        ]
    )
    
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
    
    result = _step2_review_unused_themes(initial_groups, unused_themes, mock_llm, "system prompt")
    
    # Should have added the theme to the group
    assert len(result) == 1
    assert len(result[0]["themes"]) == 3
    
    # Check that Q2-B was added
    themes = result[0]["themes"]
    assert {"question_number": 2, "theme_key": "B"} in themes


@patch('themefinder.advanced_tasks.cross_cutting_themes.load_prompt_from_file')
def test_step2_prevents_duplicate_questions(mock_load_prompt, mock_llm):
    """Test that Step 2 doesn't add multiple themes from same question."""
    # Setup initial groups with Q2 already present
    initial_groups = [
        {
            "name": "Employment Support",
            "description": "Employment themes",
            "themes": [
                {"question_number": 1, "theme_key": "A"},
                {"question_number": 2, "theme_key": "A"},  # Q2 already has a theme
            ]
        }
    ]
    
    # Setup unused themes
    unused_themes = [
        {
            "question_number": 2,
            "theme_key": "B",  # Another theme from Q2
            "label": "Employment Training",
            "description": "Skills development"
        }
    ]
    
    # Setup mock prompt
    mock_load_prompt.return_value = "{system_prompt}\n{existing_themes}\n{unused_themes}"
    
    # Setup mock LLM response - trying to add another theme from Q2
    mock_response = CrossCuttingThemeReviewResponse(
        additions=[
            CrossCuttingThemeAddition(
                cross_cutting_theme_name="Employment Support",
                question_number=2,
                theme_key="B",
                justification="Employment training fits with employment support"
            )
        ]
    )
    
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
    
    result = _step2_review_unused_themes(initial_groups, unused_themes, mock_llm, "system prompt")
    
    # Should NOT have added the theme since Q2 is already represented
    assert len(result) == 1
    assert len(result[0]["themes"]) == 2  # Still only 2 themes
    
    # Q2-B should not be in the themes
    theme_keys = [(t["question_number"], t["theme_key"]) for t in result[0]["themes"]]
    assert (2, "B") not in theme_keys


@patch('themefinder.advanced_tasks.cross_cutting_themes._step2_review_unused_themes')
@patch('themefinder.advanced_tasks.cross_cutting_themes._step1_identify_cross_cutting_themes')
def test_analyze_cross_cutting_themes_full_flow(mock_step1, mock_step2, sample_questions_themes, mock_llm):
    """Test the full analysis flow."""
    # Setup Step 1 mock response
    initial_groups = [
        {
            "name": "Employment Support",
            "description": "Employment assistance themes",
            "themes": [
                {"question_number": 1, "theme_key": "A"},
                {"question_number": 2, "theme_key": "B"},
                {"question_number": 3, "theme_key": "A"},
            ]
        },
        {
            "name": "Mental Health",
            "description": "Mental health support",
            "themes": [
                {"question_number": 1, "theme_key": "C"},
                {"question_number": 3, "theme_key": "B"},
            ]
        },
        {
            "name": "Small Group",
            "description": "Too small",
            "themes": [
                {"question_number": 2, "theme_key": "C"},
            ]
        }
    ]
    mock_step1.return_value = initial_groups
    
    # Setup Step 2 mock response (just return the same groups)
    mock_step2.return_value = initial_groups
    
    # Run analysis with min_themes=2
    result = analyze_cross_cutting_themes(
        sample_questions_themes,
        mock_llm,
        "system prompt",
        min_themes=2
    )
    
    # Should filter out "Small Group" which has only 1 theme
    assert len(result) == 2
    assert result[0]["name"] == "Employment Support"
    assert result[1]["name"] == "Mental Health"
    
    # Verify Step 1 was called with formatted themes
    mock_step1.assert_called_once()
    call_args = mock_step1.call_args[0]
    assert "Question 1, Theme A: Employment Support" in call_args[0]
    
    # Verify Step 2 was called with unused themes
    mock_step2.assert_called_once()


def test_analyze_cross_cutting_themes_no_themes(mock_llm):
    """Test handling when no cross-cutting themes are found."""
    with patch('themefinder.advanced_tasks.cross_cutting_themes._step1_identify_cross_cutting_themes') as mock_step1:
        mock_step1.return_value = []
        
        result = analyze_cross_cutting_themes(
            {1: pd.DataFrame([{"topic_id": "A", "topic": "Test: Theme"}])},
            mock_llm,
            "system prompt",
            min_themes=3
        )
        
        assert result == []


def test_analyze_cross_cutting_themes_empty_input(mock_llm):
    """Test handling of empty input."""
    result = analyze_cross_cutting_themes(
        {},
        mock_llm,
        "system prompt",
        min_themes=3
    )
    
    # Should handle empty input gracefully
    assert result == []


@patch('themefinder.advanced_tasks.cross_cutting_themes._step1_identify_cross_cutting_themes')
def test_analyze_cross_cutting_themes_no_unused_themes(mock_step1, sample_questions_themes, mock_llm):
    """Test the case where all themes are used in step 1, so no step 2 needed."""
    # Setup Step 1 to return groups that use all themes
    all_themes_used_groups = [
        {
            "name": "Employment Support",
            "description": "Employment assistance themes",
            "themes": [
                {"question_number": 1, "theme_key": "A"},
                {"question_number": 1, "theme_key": "B"},
                {"question_number": 1, "theme_key": "C"},
                {"question_number": 2, "theme_key": "A"},
                {"question_number": 2, "theme_key": "B"},
                {"question_number": 2, "theme_key": "C"},
                {"question_number": 3, "theme_key": "A"},
                {"question_number": 3, "theme_key": "B"},
            ]
        }
    ]
    mock_step1.return_value = all_themes_used_groups
    
    # Run analysis - should skip step 2 because all themes are used
    result = analyze_cross_cutting_themes(
        sample_questions_themes,
        mock_llm,
        "system prompt",
        min_themes=5
    )
    
    # Should return the groups with all themes used, and step 2 should not be called
    assert len(result) == 1
    assert result[0]["name"] == "Employment Support"
    
    # Verify Step 1 was called
    mock_step1.assert_called_once()


def test_cross_cutting_theme_model_validation():
    """Test validation of CrossCuttingTheme model."""
    from themefinder.models import CrossCuttingTheme, ConstituentTheme
    
    # Test that a cross-cutting theme with only 1 constituent theme raises error
    with pytest.raises(ValueError, match="Cross-cutting theme must include at least 2 constituent themes"):
        CrossCuttingTheme(
            name="Test Theme",
            description="Test description",
            themes=[
                ConstituentTheme(question_number=1, theme_key="A")
            ]
        )
    
    # Test that a valid cross-cutting theme with 2+ themes works
    valid_theme = CrossCuttingTheme(
        name="Test Theme",
        description="Test description", 
        themes=[
            ConstituentTheme(question_number=1, theme_key="A"),
            ConstituentTheme(question_number=2, theme_key="B"),
        ]
    )
    assert valid_theme.name == "Test Theme"
    assert len(valid_theme.themes) == 2
    
    # Test that duplicate theme combinations raise error
    with pytest.raises(ValueError, match="Duplicate question_number and theme_key combinations found"):
        CrossCuttingTheme(
            name="Test Theme",
            description="Test description",
            themes=[
                ConstituentTheme(question_number=1, theme_key="A"),
                ConstituentTheme(question_number=1, theme_key="A"),  # Duplicate
            ]
        )


def test_cross_cutting_themes_response_validation():
    """Test validation of CrossCuttingThemesResponse model."""
    from themefinder.models import CrossCuttingThemesResponse, CrossCuttingTheme, ConstituentTheme
    
    # Test that duplicate theme names raise error
    theme1 = CrossCuttingTheme(
        name="Duplicate Name",
        description="First theme", 
        themes=[
            ConstituentTheme(question_number=1, theme_key="A"),
            ConstituentTheme(question_number=2, theme_key="B"),
        ]
    )
    theme2 = CrossCuttingTheme(
        name="Duplicate Name",  # Same name
        description="Second theme",
        themes=[
            ConstituentTheme(question_number=3, theme_key="C"),
            ConstituentTheme(question_number=4, theme_key="D"),
        ]
    )
    
    with pytest.raises(ValueError, match="Cross-cutting theme names must be unique"):
        CrossCuttingThemesResponse(cross_cutting_themes=[theme1, theme2])
    
    # Test that empty response is valid
    empty_response = CrossCuttingThemesResponse(cross_cutting_themes=[])
    assert len(empty_response.cross_cutting_themes) == 0