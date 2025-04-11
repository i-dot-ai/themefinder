from pydantic import BaseModel, Field, model_validator


def validate_non_empty_fields(model: BaseModel) -> BaseModel:
    """
    Validate that all string fields in the model are non-empty (after stripping)
    and that list fields are not empty.

    Args:
        model (BaseModel): A Pydantic model instance.

    Returns:
        BaseModel: The same model if validation passes.

    Raises:
        ValueError: If any string field is empty or any list field is empty.
    """
    for field_name, value in model.__dict__.items():
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{field_name} cannot be empty or only whitespace")
        if isinstance(value, list) and not value:
            raise ValueError(f"{field_name} cannot be an empty list")
    return model


def validate_position(model: BaseModel) -> BaseModel:
    """
    Validate that the model's 'position' field is one of the allowed values.

    Args:
        model (BaseModel): A Pydantic model instance with a 'position' attribute.

    Returns:
        BaseModel: The same model if validation passes.

    Raises:
        ValueError: If the 'position' field is not one of the allowed values.
    """
    allowed_positions = {"AGREEMENT", "DISAGREEMENT", "UNCLEAR"}
    if model.position not in allowed_positions:
        raise ValueError(f"position must be one of {allowed_positions}")
    return model


def validate_detail_detection(model: BaseModel) -> BaseModel:
    """
    Validate that the model's 'evidence_rich' field is one of the allowed values.

    Args:
        model (BaseModel): A Pydantic model instance with an 'evidence_rich' attribute.

    Returns:
        BaseModel: The same model if validation passes.

    Raises:
        ValueError: If the 'evidence_rich' field is not one of the allowed values.
    """
    allowed_values = {"YES", "NO"}
    if model.evidence_rich not in allowed_values:
        raise ValueError(f"evidence_rich must be one of {allowed_values}")
    return model


def validate_stances(model: BaseModel) -> BaseModel:
    """
    Validate that every stance in the model's 'stances' field is allowed.

    Args:
        model (BaseModel): A Pydantic model instance with a 'stances' attribute.

    Returns:
        BaseModel: The same model if validation passes.

    Raises:
        ValueError: If any stance is not among the allowed stances.
    """
    allowed_stances = {"POSITIVE", "NEGATIVE"}
    for stance in model.stances:
        if stance not in allowed_stances:
            raise ValueError(f"stances must be one of {allowed_stances}")
    return model


def validate_mapping_stance_lengths(model: BaseModel) -> BaseModel:
    """
    Validate that the lengths of the model's 'stances' and 'labels' fields match.

    Args:
        model (BaseModel): A Pydantic model instance with 'stances' and 'labels' attributes.

    Returns:
        BaseModel: The same model if validation passes.

    Raises:
        ValueError: If the lengths of 'stances' and 'labels' do not match.
    """
    if len(model.stances) != len(model.labels):
        raise ValueError("'stances' must have the same length as 'labels'")
    return model


def validate_mapping_unique_labels(model: BaseModel) -> BaseModel:
    """
    Validate that the model's 'labels' field contains unique values.

    Args:
        model (BaseModel): A Pydantic model instance with a 'labels' attribute.

    Returns:
        BaseModel: The same model if validation passes.

    Raises:
        ValueError: If 'labels' contains duplicate values.
    """
    if len(model.labels) != len(set(model.labels)):
        raise ValueError("'labels' must be unique")
    return model


class SentimentAnalysisOutput(BaseModel):
    response_id: int = Field(gt=0)
    position: str

    @model_validator(mode="after")
    def run_validations(self) -> "SentimentAnalysisOutput":
        """
        Run all validations for SentimentAnalysisOutput.

        Validates that:
         - 'position' is one of the allowed values.
         - No fields are empty or only whitespace (for strings) and no lists are empty.
        """
        validate_position(self)
        validate_non_empty_fields(self)
        return self


class ThemeMappingOutput(BaseModel):
    response_id: int = Field(gt=0)
    labels: list[str]
    reasons: list[str]
    stances: list[str]

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeMappingOutput":
        """
        Run all validations for ThemeMappingOutput.

        Validates that:
         - 'stances' are only 'POSITIVE' or 'NEGATIVE'.
         - The 'stances' and 'labels' have matching lengths.
         - 'labels' are unique.
        """
        validate_stances(self)
        validate_mapping_stance_lengths(self)
        validate_mapping_unique_labels(self)
        return self


class DetailDetectionOutput(BaseModel):
    response_id: int = Field(gt=0)
    evidence_rich: str

    @model_validator(mode="after")
    def run_validations(self) -> "DetailDetectionOutput":
        """
        Run all validations for DetailDetectionOutput.
        """
        validate_non_empty_fields(self)
        validate_detail_detection(self)
        return self
