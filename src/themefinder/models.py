import logging
from typing import List, Optional, Annotated
from enum import Enum
from pydantic import BaseModel, Field, model_validator, AfterValidator

logger = logging.getLogger(__file__)


class Position(str, Enum):
    """Enum for valid position values"""

    AGREEMENT = "AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    UNCLEAR = "UNCLEAR"


class Stance(str, Enum):
    """Enum for valid stance values"""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class EvidenceRich(str, Enum):
    """Enum for valid evidence_rich values"""

    YES = "YES"
    NO = "NO"


class ValidatedModel(BaseModel):
    """Base model with common validation methods"""

    def validate_non_empty_fields(self) -> "ValidatedModel":
        """
        Validate that all string fields are non-empty and all list fields are not empty.
        """
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str) and not item.strip():
                        raise ValueError(
                            f"Item {i} in {field_name} cannot be empty or only whitespace"
                        )
        return self

    def validate_unique_items(
        self, field_name: str, transform_func: Optional[callable] = None
    ) -> "ValidatedModel":
        """
        Validate that a field contains unique values.

        Args:
            field_name: The name of the field to check for uniqueness
            transform_func: Optional function to transform items before checking uniqueness
                           (e.g., lowercasing strings)
        """
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' does not exist")
        items = getattr(self, field_name)
        if not isinstance(items, list):
            raise ValueError(f"Field '{field_name}' is not a list")
        if transform_func:
            transformed_items = [transform_func(item) for item in items]
        else:
            transformed_items = items
        if len(transformed_items) != len(set(transformed_items)):
            raise ValueError(f"'{field_name}' must contain unique values")
        return self

    def validate_unique_attribute_in_list(
        self, list_field: str, attr_name: str
    ) -> "ValidatedModel":
        """
        Validate that an attribute across all objects in a list field is unique.

        Args:
            list_field: The name of the list field containing objects
            attr_name: The attribute within each object to check for uniqueness
        """
        if not hasattr(self, list_field):
            raise ValueError(f"Field '{list_field}' does not exist")

        items = getattr(self, list_field)
        if not isinstance(items, list):
            raise ValueError(f"Field '{list_field}' is not a list")

        attr_values = []
        for item in items:
            if not hasattr(item, attr_name):
                raise ValueError(
                    f"Item in '{list_field}' does not have attribute '{attr_name}'"
                )
            attr_values.append(getattr(item, attr_name))
        if len(attr_values) != len(set(attr_values)):
            raise ValueError(
                f"'{attr_name}' must be unique across all items in '{list_field}'"
            )
        return self

    def validate_equal_lengths(self, *field_names) -> "ValidatedModel":
        """
        Validate that multiple list fields have the same length.

        Args:
            *field_names: Variable number of field names to check for equal lengths
        """
        if len(field_names) < 2:
            return self
        lengths = []
        for field_name in field_names:
            if not hasattr(self, field_name):
                raise ValueError(f"Field '{field_name}' does not exist")

            items = getattr(self, field_name)
            if not isinstance(items, list):
                raise ValueError(f"Field '{field_name}' is not a list")

            lengths.append(len(items))
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Fields {', '.join(field_names)} must all have the same length"
            )
        return self

    @model_validator(mode="after")
    def run_validations(self) -> "ValidatedModel":
        """
        Run common validations. Override in subclasses to add specific validations.
        """
        return self.validate_non_empty_fields()


class SentimentAnalysisOutput(ValidatedModel):
    """Model for sentiment analysis output"""

    response_id: int = Field(gt=0)
    position: Position


class SentimentAnalysisResponses(ValidatedModel):
    """Container for all sentiment analysis responses"""

    responses: List[SentimentAnalysisOutput]

    @model_validator(mode="after")
    def run_validations(self) -> "SentimentAnalysisResponses":
        """Validate that response_ids are unique"""
        self.validate_non_empty_fields()
        response_ids = [resp.response_id for resp in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("Response IDs must be unique")
        return self


def lower_case_strip_str(value: str) -> str:
    return value.lower().strip()


class Theme(ValidatedModel):
    """Model for a single extracted theme"""

    topic_label: Annotated[str, AfterValidator(lower_case_strip_str)] = Field(
        ..., description="Short label summarizing the topic in a few words"
    )
    topic_description: str = Field(
        ..., description="More detailed description of the topic in 1-2 sentences"
    )
    position: Position = Field(
        ...,
        description="SENTIMENT ABOUT THIS TOPIC (AGREEMENT, DISAGREEMENT, OR UNCLEAR)",
    )

    class Config:
        frozen = True


class ThemeGenerationResponses(BaseModel):
    """Container for all extracted themes"""

    responses: set[Theme] = Field(
        ..., description="List of extracted themes", min_length=1
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeGenerationResponses":
        """Ensure there are no duplicate themes"""
        labels = {theme.topic_label for theme in self.responses}

        def _reduce(topic_label: str):
            themes = list(
                filter(
                    lambda x: x.topic_label == topic_label,
                    self.responses,
                )
            )
            if len(themes) == 1:
                return themes[0]

            topic_description = ", ".join(t.topic_description for t in themes)
            logger.warning("compressing themes:" + topic_description)
            return Theme(
                topic_label=themes[0].topic_label,
                topic_description="\n".join(t.topic_description for t in themes),
                position=themes[0].position,
            )

        self.responses = [_reduce(label) for label in labels]

        return self


class CondensedTheme(ValidatedModel):
    """Model for a single condensed theme"""

    topic_label: Annotated[str, AfterValidator(lower_case_strip_str)] = Field(
        ..., description="Representative label for the condensed topic"
    )
    topic_description: str = Field(
        ...,
        description="Concise description incorporating key insights from constituent topics",
    )
    source_topic_count: int = Field(
        ..., gt=0, description="Sum of source_topic_counts from combined topics"
    )

    class Config:
        frozen = True


class ThemeCondensationResponses(BaseModel):
    """Container for all condensed themes"""

    responses: set[CondensedTheme] = Field(
        ..., description="List of condensed themes", min_length=1
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeCondensationResponses":
        """Ensure there are no duplicate themes"""
        labels = {theme.topic_label for theme in self.responses}

        def _reduce(topic_label: str) -> CondensedTheme:
            themes = list(
                filter(
                    lambda x: x.topic_label == topic_label,
                    self.responses,
                )
            )
            if len(themes) == 1:
                return themes[0]

            topic_description = "\n".join(t.topic_description for t in themes)
            logger.warning("compressing themes: " + topic_description)
            return CondensedTheme(
                topic_label=themes[0].topic_label,
                topic_description="\n".join(t.topic_description for t in themes),
                source_topic_count=sum(t.source_topic_count for t in themes),
            )

        self.responses = [_reduce(label) for label in labels]

        return self


class RefinedTheme(ValidatedModel):
    """Model for a single refined theme"""

    topic: str = Field(
        ..., description="Topic label and description combined with a colon separator"
    )
    source_topic_count: int = Field(
        ..., gt=0, description="Count of source topics combined"
    )

    @model_validator(mode="after")
    def run_validations(self) -> "RefinedTheme":
        """Run all validations for RefinedTheme"""
        self.validate_non_empty_fields()
        self.validate_topic_format()
        return self

    def validate_topic_format(self) -> "RefinedTheme":
        """
        Validate that topic contains a label and description separated by a colon.
        """
        if ":" not in self.topic:
            raise ValueError(
                "Topic must contain a label and description separated by a colon"
            )

        label, description = self.topic.split(":", 1)
        if not label.strip() or not description.strip():
            raise ValueError("Both label and description must be non-empty")

        word_count = len(label.strip().split())
        if word_count > 10:
            raise ValueError(f"Topic label must be under 10 words (found {word_count})")

        return self


class ThemeRefinementResponses(ValidatedModel):
    """Container for all refined themes"""

    responses: List[RefinedTheme] = Field(..., description="List of refined themes")

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeRefinementResponses":
        """Ensure there are no duplicate themes"""
        self.validate_non_empty_fields()
        topics = [theme.topic.lower().strip() for theme in self.responses]
        if len(topics) != len(set(topics)):
            raise ValueError("Duplicate topics detected")

        return self


class ThemeMappingOutput(ValidatedModel):
    """Model for theme mapping output"""

    response_id: int = Field(gt=0, description="Response ID, must be greater than 0")
    labels: List[str] = Field(..., description="List of theme labels")

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeMappingOutput":
        """
        Run all validations for ThemeMappingOutput.
        """
        self.validate_non_empty_fields()
        self.validate_unique_items("labels")
        return self


class ThemeMappingResponses(ValidatedModel):
    """Container for all theme mapping responses"""

    responses: List[ThemeMappingOutput] = Field(
        ..., description="List of theme mapping outputs"
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeMappingResponses":
        """
        Validate that response_ids are unique.
        """
        self.validate_non_empty_fields()
        response_ids = [resp.response_id for resp in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("Response IDs must be unique")
        return self


class DetailDetectionOutput(ValidatedModel):
    """Model for detail detection output"""

    response_id: int = Field(gt=0, description="Response ID, must be greater than 0")
    evidence_rich: EvidenceRich = Field(
        ..., description="Whether the response is evidence-rich (YES or NO)"
    )


class DetailDetectionResponses(ValidatedModel):
    """Container for all detail detection responses"""

    responses: List[DetailDetectionOutput] = Field(
        ..., description="List of detail detection outputs"
    )

    @model_validator(mode="after")
    def run_validations(self) -> "DetailDetectionResponses":
        """
        Validate that response_ids are unique.
        """
        self.validate_non_empty_fields()
        response_ids = [resp.response_id for resp in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("Response IDs must be unique")
        return self


class ThemeNode(ValidatedModel):
    """Model for topic nodes created during hierarchical clustering"""

    topic_id: str = Field(
        ...,
        description="Short alphabetic ID (e.g. 'A', 'B', 'C') - iteration prefix will be added automatically",
    )
    topic_label: str = Field(
        ..., description="4-5 word label encompassing merged child topics"
    )
    topic_description: str = Field(
        ..., description="1-2 sentences combining key aspects of child topics"
    )
    source_topic_count: int = Field(gt=0, description="Sum of all child topic counts")
    parent_id: Optional[str] = Field(
        default=None,
        description="Internal field: ID of parent topic node, managed by clustering agent, not set by LLM",
    )
    children: List[str] = Field(
        default_factory=list, description="List of topic_ids of merged child topics"
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeNode":
        """Validate topic node constraints"""
        if self.children:
            # Each parent must have at least 2 children
            if len(self.children) < 2:
                raise ValueError("Each topic node must have at least 2 children")
            # Validate children are unique
            if len(self.children) != len(set(self.children)):
                raise ValueError("Child topic IDs must be unique")

        return self


class HierarchicalClusteringResponse(ValidatedModel):
    """Model for hierarchical clustering agent response"""

    parent_themes: List[ThemeNode] = Field(
        default=[],
        description="List of parent themes created by merging similar themes",
    )
    should_terminate: bool = Field(
        ...,
        description="True if no more meaningful clustering is possible, false otherwise",
    )

    @model_validator(mode="after")
    def run_validations(self) -> "HierarchicalClusteringResponse":
        """Validate clustering response constraints"""
        self.validate_non_empty_fields()

        # Validate that no child appears in multiple parents
        all_children = []
        for parent in self.parent_themes:
            all_children.extend(parent.children)

        if len(all_children) != len(set(all_children)):
            raise ValueError("Each child theme can have at most one parent")

        return self


# Cross-Cutting Theme Identification Models


class CrossCuttingThemeDefinition(BaseModel):
    """Model for a high-level cross-cutting theme."""

    name: str = Field(
        ...,
        description="Short, descriptive name for the cross-cutting theme (3-7 words)",
    )
    description: str = Field(
        ...,
        description="2-sentence description of what this cross-cutting theme represents",
    )


class CrossCuttingThemeIdentificationResponse(BaseModel):
    """Response model for identifying cross-cutting themes."""

    themes: List[CrossCuttingThemeDefinition] = Field(
        default=[], description="List of identified cross-cutting themes"
    )


class CrossCuttingThemeMapping(BaseModel):
    """Model for mapping individual themes to a cross-cutting theme."""

    theme_name: str = Field(
        ..., description="Name of the cross-cutting theme this theme belongs to"
    )
    theme_ids: List[str] = Field(
        ...,
        description="List of theme IDs that belong to this cross-cutting theme (e.g., ['A', 'B', 'C'])",
    )


class CrossCuttingThemeMappingResponse(BaseModel):
    """Response model for mapping question themes to cross-cutting themes."""

    mappings: List[CrossCuttingThemeMapping] = Field(
        default=[], description="List of cross-cutting theme mappings for this question"
    )
