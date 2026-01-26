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


class SentimentAnalysisOutput(BaseModel):
    """Model for sentiment analysis output"""

    response_id: int = Field(gt=0)
    position: Position


class SentimentAnalysisResponses(BaseModel):
    """Container for all sentiment analysis responses"""

    responses: List[SentimentAnalysisOutput] = Field(min_length=1)

    @model_validator(mode="after")
    def run_validations(self) -> "SentimentAnalysisResponses":
        """Validate that response_ids are unique"""
        response_ids = [resp.response_id for resp in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("Response IDs must be unique")
        return self


def lower_case_strip_str(value: str) -> str:
    return value.lower().strip()


class Theme(BaseModel):
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

    responses: list[Theme] = Field(
        ..., description="List of extracted themes", min_length=1
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeGenerationResponses":
        """Ensure there are no duplicate themes"""
        self.responses = list(set(self.responses))

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

            topic_description = " ".join(t.topic_description for t in themes)
            logger.warning("compressing themes:" + topic_description)
            return Theme(
                topic_label=themes[0].topic_label,
                topic_description="\n".join(t.topic_description for t in themes),
                position=themes[0].position,
            )

        self.responses = [_reduce(label) for label in labels]

        return self


class CondensedTheme(BaseModel):
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

    responses: list[CondensedTheme] = Field(
        ..., description="List of condensed themes", min_length=1
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeCondensationResponses":
        """Ensure there are no duplicate themes"""
        self.responses = list(set(self.responses))

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

            topic_description = " ".join(t.topic_description for t in themes)
            logger.warning("compressing themes: " + topic_description)
            return CondensedTheme(
                topic_label=themes[0].topic_label,
                topic_description="\n".join(t.topic_description for t in themes),
                source_topic_count=sum(t.source_topic_count for t in themes),
            )

        self.responses = [_reduce(label) for label in labels]

        return self


class RefinedTheme(BaseModel):
    """Model for a single refined theme"""

    topic: str = Field(
        ...,
        description="Topic label and description combined with a colon separator",
        min_length=1,
    )
    source_topic_count: int = Field(
        ..., gt=0, description="Count of source topics combined"
    )

    @model_validator(mode="after")
    def run_validations(self) -> "RefinedTheme":
        """Run all validations for RefinedTheme"""
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


class ThemeRefinementResponses(BaseModel):
    """Container for all refined themes"""

    responses: List[RefinedTheme] = Field(
        ..., description="List of refined themes", min_length=1
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeRefinementResponses":
        """Ensure there are no duplicate themes"""
        topics = [theme.topic.lower().strip() for theme in self.responses]
        if len(topics) != len(set(topics)):
            raise ValueError("Duplicate topics detected")

        return self


class ThemeMappingOutput(BaseModel):
    """Model for theme mapping output"""

    response_id: int = Field(gt=0, description="Response ID, must be greater than 0")
    labels: set[str] = Field(..., description="List of theme labels", min_length=1)


class ThemeMappingResponses(BaseModel):
    """Container for all theme mapping responses"""

    responses: List[ThemeMappingOutput] = Field(
        ..., description="List of theme mapping outputs", min_length=1
    )

    @model_validator(mode="after")
    def run_validations(self) -> "ThemeMappingResponses":
        """
        Validate that response_ids are unique.
        """
        response_ids = [resp.response_id for resp in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("Response IDs must be unique")
        return self


class DetailDetectionOutput(BaseModel):
    """Model for detail detection output"""

    response_id: int = Field(gt=0, description="Response ID, must be greater than 0")
    evidence_rich: EvidenceRich = Field(
        ..., description="Whether the response is evidence-rich (YES or NO)"
    )


class DetailDetectionResponses(BaseModel):
    """Container for all detail detection responses"""

    responses: List[DetailDetectionOutput] = Field(
        ..., description="List of detail detection outputs", min_length=1
    )

    @model_validator(mode="after")
    def run_validations(self) -> "DetailDetectionResponses":
        """
        Validate that response_ids are unique.
        """
        response_ids = [resp.response_id for resp in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("Response IDs must be unique")
        return self


class ThemeNode(BaseModel):
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


class HierarchicalClusteringResponse(BaseModel):
    """Model for hierarchical clustering agent response"""

    parent_themes: List[ThemeNode] = Field(
        description="List of parent themes created by merging similar themes",
        min_length=1,
    )
    should_terminate: bool = Field(
        ...,
        description="True if no more meaningful clustering is possible, false otherwise",
    )

    @model_validator(mode="after")
    def run_validations(self) -> "HierarchicalClusteringResponse":
        """Validate clustering response constraints"""

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
