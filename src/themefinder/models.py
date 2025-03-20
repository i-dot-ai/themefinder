from pydantic import BaseModel, Field, model_validator


def validate_non_empty_fields(model):
    for field_name, value in model.__dict__.items():
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{field_name} cannot be empty or only whitespace")
        if isinstance(value, list) and not value:
            raise ValueError(f"{field_name} cannot be an empty list")
    return model


def validate_position(model):
    allowed_positions = {"agreement", "disagreement", "unclear"}
    if model.position not in allowed_positions:
        raise ValueError(f"position must be one of {allowed_positions}")
    return model


def validate_stances(model):
    allowed_stances = {"POSITIVE", "NEGATIVE"}
    for stance in model.stances:
        if stance not in allowed_stances:
            raise ValueError(f"stances must be one of {allowed_stances}")
    return model


def validate_mapping_reason_and_stance_lengths(model):
    if len(model.reasons) != len(model.labels):
        raise ValueError("'reasons' must have the same length as 'labels'")
    if len(model.stances) != len(model.labels):
        raise ValueError("'stances' must have the same length as 'labels'")
    return model


class SentimentAnalysisInput(BaseModel):
    response_id: int = Field(gt=0)
    response: str

    @model_validator(mode="after")
    def run_validations(self):
        validate_non_empty_fields(self)
        return self


class SentimentAnalysisOutput(BaseModel):
    response_id: int = Field(gt=0)
    response: str
    position: str

    @model_validator(mode="after")
    def run_validations(self):
        validate_position(self)
        validate_non_empty_fields(self)
        return self


class ThemeGenerationInput(BaseModel):
    response_id: int = Field(gt=0)
    response: str
    position: str

    @model_validator(mode="after")
    def run_validations(self):
        validate_position(self)
        validate_non_empty_fields(self)
        return self


class ThemeGenerationOutput(BaseModel):
    topic_label: str
    topic_description: str
    position: str

    @model_validator(mode="after")
    def run_validations(self):
        validate_position(self)
        validate_non_empty_fields(self)
        return self


class ThemeCondensationInput(BaseModel):
    topic_label: str
    topic_description: str
    position: str

    @model_validator(mode="after")
    def run_validations(self):
        validate_position(self)
        validate_non_empty_fields(self)
        return self


class ThemeCondensationOutput(BaseModel):
    topic_label: str
    topic_description: str
    source_topic_count: int = Field(ge=0)

    @model_validator(mode="after")
    def run_validations(self):
        validate_non_empty_fields(self)
        return self


class ThemeRefinementInput(BaseModel):
    topic_label: str
    topic_description: str
    source_topic_count: int = Field(ge=0)

    @model_validator(mode="after")
    def run_validations(self):
        validate_non_empty_fields(self)
        return self


class ThemeRefinementOutput(BaseModel):
    topic_id: str
    topic: str
    source_topic_count: int

    @model_validator(mode="after")
    def run_validations(self):
        validate_non_empty_fields(self)
        return self


class ThemeMappingResponseInput(BaseModel):
    response_id: int = Field(gt=0)
    response: str
    position: str

    @model_validator(mode="after")
    def run_validations(self):
        validate_position(self)
        validate_non_empty_fields(self)
        return self


class ThemeMappingThemeInput(BaseModel):
    topic_id: str
    topic: str

    @model_validator(mode="after")
    def run_validations(self):
        validate_non_empty_fields(self)
        return self


class ThemeMappingOutput(BaseModel):
    response_id: int = Field(ge=0)
    response: str
    position: str
    labels: list[str]
    reasons: list[str]
    stances: list[str]

    @model_validator(mode="after")
    def run_validations(self):
        validate_position(self)
        validate_stances(self)
        validate_mapping_reason_and_stance_lengths(self)
        validate_non_empty_fields(self)
        return self
