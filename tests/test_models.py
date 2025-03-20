import pytest
from pydantic import BaseModel, ValidationError

from themefinder.models import (
    SentimentAnalysisInput,
    SentimentAnalysisOutput,
    ThemeCondensationInput,
    ThemeCondensationOutput,
    ThemeGenerationInput,
    ThemeGenerationOutput,
    ThemeMappingOutput,
    ThemeMappingResponseInput,
    ThemeMappingThemeInput,
    ThemeRefinementInput,
    ThemeRefinementOutput,
    validate_mapping_reason_and_stance_lengths,
    validate_non_empty_fields,
    validate_position,
    validate_stances,
)


# Helper function to check validation errors
def raises_validation_error(model_class, data):
    with pytest.raises(ValidationError):
        model_class(**data)


# === Mock Model for Direct Validation Function Testing ===
class MockModel(BaseModel):
    response_id: int
    response: str
    position: str = None
    stances: list[str] = None
    labels: list[str] = None
    reasons: list[str] = None


# === Tests for validate_non_empty_fields ===
def test_validate_non_empty_fields_valid():
    model = MockModel(response_id=1, response="Valid response")
    validate_non_empty_fields(model)  # Should not raise error


def test_validate_non_empty_fields_invalid():
    model = MockModel(response_id=1, response="   ")
    with pytest.raises(ValueError, match="response cannot be empty or only whitespace"):
        validate_non_empty_fields(model)


# === Tests for validate_position ===
def test_validate_position_valid():
    model = MockModel(response_id=1, response="Valid response", position="agreement")
    validate_position(model)  # Should not raise error


def test_validate_position_invalid():
    model = MockModel(
        response_id=1, response="Valid response", position="invalid_value"
    )
    with pytest.raises(ValueError, match="position must be one of"):
        validate_position(model)


# === Tests for validate_stances ===
def test_validate_stances_valid():
    model = MockModel(
        response_id=1, response="Valid response", stances=["POSITIVE", "NEGATIVE"]
    )
    validate_stances(model)  # Should not raise error


def test_validate_stances_invalid():
    model = MockModel(response_id=1, response="Valid response", stances=["INVALID"])
    with pytest.raises(ValueError, match="stances must be one of"):
        validate_stances(model)


# === Tests for validate_mapping_reason_and_stance_lengths ===
def test_validate_mapping_reason_and_stance_lengths_valid():
    model = MockModel(
        response_id=1,
        response="Valid response",
        labels=["Label1", "Label2"],
        reasons=["Reason1", "Reason2"],
        stances=["POSITIVE", "NEGATIVE"],
    )
    validate_mapping_reason_and_stance_lengths(model)  # Should not raise error


def test_validate_mapping_reason_and_stance_lengths_invalid():
    model = MockModel(
        response_id=1,
        response="Valid response",
        labels=["Label1"],
        reasons=["Reason1", "Reason2"],  # Mismatch in length
        stances=["POSITIVE"],
    )
    with pytest.raises(
        ValueError, match="'reasons' must have the same length as 'labels'"
    ):
        validate_mapping_reason_and_stance_lengths(model)

    model = MockModel(
        response_id=1,
        response="Valid response",
        labels=["Label1"],
        reasons=["Reason1"],
        stances=["POSITIVE", "NEGATIVE"],  # Mismatch in length
    )
    with pytest.raises(
        ValueError, match="'stances' must have the same length as 'labels'"
    ):
        validate_mapping_reason_and_stance_lengths(model)


# === SentimentAnalysisInput ===
def test_sentiment_analysis_input_valid():
    model = SentimentAnalysisInput(response_id=1, response="This is a test response.")
    assert model.response_id == 1


def test_sentiment_analysis_input_invalid():
    raises_validation_error(
        SentimentAnalysisInput, {"response_id": 0, "response": "Valid text"}
    )
    raises_validation_error(
        SentimentAnalysisInput, {"response_id": 1, "response": "  "}
    )


# === SentimentAnalysisOutput ===
def test_sentiment_analysis_output_valid():
    model = SentimentAnalysisOutput(
        response_id=1, response="Test response", position="agreement"
    )
    assert model.position == "agreement"


def test_sentiment_analysis_output_invalid():
    raises_validation_error(
        SentimentAnalysisOutput,
        {"response_id": 1, "response": "Text", "position": "invalid"},
    )
    raises_validation_error(
        SentimentAnalysisOutput,
        {"response_id": 1, "response": "", "position": "agreement"},
    )


# === ThemeGenerationInput ===
def test_theme_generation_input_valid():
    model = ThemeGenerationInput(
        response_id=1, response="Test response", position="disagreement"
    )
    assert model.position == "disagreement"


def test_theme_generation_input_invalid():
    raises_validation_error(
        ThemeGenerationInput,
        {"response_id": 1, "response": "Valid", "position": "wrong_value"},
    )


# === ThemeGenerationOutput ===
def test_theme_generation_output_valid():
    model = ThemeGenerationOutput(
        topic_label="Label", topic_description="Desc", position="unclear"
    )
    assert model.topic_label == "Label"


def test_theme_generation_output_invalid():
    raises_validation_error(
        ThemeGenerationOutput,
        {"topic_label": "", "topic_description": "Valid", "position": "agreement"},
    )


# === ThemeCondensationInput ===
def test_theme_condensation_input_valid():
    model = ThemeCondensationInput(
        topic_label="Topic", topic_description="Description", position="agreement"
    )
    assert model.position == "agreement"


def test_theme_condensation_input_invalid():
    raises_validation_error(
        ThemeCondensationInput,
        {"topic_label": "Valid", "topic_description": "", "position": "agreement"},
    )
    raises_validation_error(
        ThemeCondensationInput,
        {"topic_label": "Valid", "topic_description": "Valid", "position": "wrong"},
    )


# === ThemeCondensationOutput ===
def test_theme_condensation_output_valid():
    model = ThemeCondensationOutput(
        topic_label="Topic", topic_description="Desc", source_topic_count=5
    )
    assert model.source_topic_count == 5


def test_theme_condensation_output_invalid():
    raises_validation_error(
        ThemeCondensationOutput,
        {"topic_label": "Label", "topic_description": "Desc", "source_topic_count": -1},
    )


# === ThemeRefinementInput ===
def test_theme_refinement_input_valid():
    model = ThemeRefinementInput(
        topic_label="Topic", topic_description="Desc", source_topic_count=3
    )
    assert model.source_topic_count == 3


def test_theme_refinement_input_invalid():
    raises_validation_error(
        ThemeRefinementInput,
        {"topic_label": "", "topic_description": "Valid", "source_topic_count": 2},
    )


# === ThemeRefinementOutput ===
def test_theme_refinement_output_valid():
    model = ThemeRefinementOutput(topic_id="123", topic="Test", source_topic_count=2)
    assert model.topic_id == "123"


def test_theme_refinement_output_invalid():
    raises_validation_error(
        ThemeRefinementOutput,
        {"topic_id": "", "topic": "Valid", "source_topic_count": 2},
    )


# === ThemeMappingResponseInput ===
def test_theme_mapping_response_input_valid():
    model = ThemeMappingResponseInput(
        response_id=1, response="Valid", position="disagreement"
    )
    assert model.response_id == 1


def test_theme_mapping_response_input_invalid():
    raises_validation_error(
        ThemeMappingResponseInput,
        {"response_id": 1, "response": "Valid", "position": "incorrect"},
    )


# === ThemeMappingThemeInput ===
def test_theme_mapping_theme_input_valid():
    model = ThemeMappingThemeInput(topic_id="123", topic="Valid")
    assert model.topic_id == "123"


def test_theme_mapping_theme_input_invalid():
    raises_validation_error(ThemeMappingThemeInput, {"topic_id": "", "topic": "Valid"})


# === ThemeMappingOutput ===
def test_theme_mapping_output_valid():
    model = ThemeMappingOutput(
        response_id=1,
        response="Valid response",
        position="agreement",
        labels=["Label1", "Label2"],
        reasons=["Reason1", "Reason2"],
        stances=["POSITIVE", "NEGATIVE"],
    )
    assert len(model.labels) == len(model.reasons) == len(model.stances)


def test_theme_mapping_output_invalid():
    raises_validation_error(
        ThemeMappingOutput,
        {
            "response_id": 1,
            "response": "Valid",
            "position": "wrong_value",
            "labels": ["Label1"],
            "reasons": ["Reason1", "Reason2"],  # Different length
            "stances": ["POSITIVE"],
        },
    )
    raises_validation_error(
        ThemeMappingOutput,
        {
            "response_id": 1,
            "response": "Valid",
            "position": "agreement",
            "labels": ["Label1"],
            "reasons": ["Reason1"],
            "stances": ["InvalidStance"],  # Invalid stance value
        },
    )
