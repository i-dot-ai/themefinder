"""Configuration dataclasses for synthetic dataset generation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class NoiseLevel(Enum):
    """Noise injection intensity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResponseLength(Enum):
    """Response length categories with word count ranges."""

    SHORT = (5, 50)
    MEDIUM = (51, 250)
    LONG = (251, 1000)


class QuestionType(Enum):
    """UK government consultation question types."""

    OPEN_ENDED = "open_ended"  # "What are your views on..."
    AGREE_DISAGREE = "agree_disagree"  # 5-point Likert scale with free text
    YES_NO = "yes_no"  # Binary with "please explain" follow-up
    MULTIPLE_CHOICE = "multiple_choice"  # Select all that apply


class ResponseType(Enum):
    """Response stance/quality categories."""

    AGREE = "agree"
    DISAGREE = "disagree"
    NUANCED = "nuanced"
    OFF_TOPIC = "off_topic"
    LOW_QUALITY = "low_quality"


@dataclass
class DemographicField:
    """Single demographic or policy context field with values and distribution weights.

    Can represent either:
    - Fixed demographics (region, age, etc.) - is_policy_context=False
    - Policy-specific context (student loan status, etc.) - is_policy_context=True

    Policy context fields can have stance_modifiers to influence respondent disposition.
    """

    name: str
    display_name: str
    values: list[str]
    distribution: list[float]
    enabled: bool = True
    is_policy_context: bool = False  # True for LLM-generated policy-specific fields
    stance_modifiers: list[float] | None = (
        None  # Per-value: -0.1 to +0.1 (subtle influence)
    )

    def __post_init__(self) -> None:
        """Validate distribution and stance modifiers."""
        total = sum(self.distribution)
        if not (0.99 <= total <= 1.01):
            msg = f"Distribution for {self.name} must sum to 1.0, got {total}"
            raise ValueError(msg)
        if len(self.values) != len(self.distribution):
            msg = f"Values and distribution must have same length for {self.name}"
            raise ValueError(msg)
        if self.stance_modifiers is not None:
            if len(self.stance_modifiers) != len(self.values):
                msg = f"Stance modifiers must match values length for {self.name}"
                raise ValueError(msg)
            for mod in self.stance_modifiers:
                if not (-1.0 <= mod <= 1.0):
                    msg = (
                        f"Stance modifiers must be between -1.0 and 1.0 for {self.name}"
                    )
                    raise ValueError(msg)


@dataclass
class QuestionConfig:
    """Configuration for a single consultation question."""

    number: int
    text: str
    question_type: QuestionType = QuestionType.OPEN_ENDED
    multi_choice_options: Optional[list[str]] = None
    scale_statement: Optional[str] = None  # For agree/disagree: the statement to rate


@dataclass
class ResponseSpec:
    """Specification for generating a single response."""

    response_id: int
    themes: list[str]
    stances: list[str]
    response_type: ResponseType
    length: ResponseLength
    persona: dict[str, str]
    apply_noise: bool
    noise_type: Optional[str] = None


@dataclass
class GenerationConfig:
    """Complete configuration for synthetic dataset generation."""

    dataset_name: str
    topic: str
    n_responses: int
    questions: list[QuestionConfig]
    demographic_fields: list[DemographicField]
    noise_level: NoiseLevel = NoiseLevel.MEDIUM

    position_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "agree": 0.30,
            "disagree": 0.20,
            "nuanced": 0.20,
            "off_topic": 0.05,
            "low_quality": 0.05,
        }
    )

    length_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "short": 0.20,
            "medium": 0.60,
            "long": 0.20,
        }
    )

    multi_theme_ratio: float = 0.30

    @property
    def output_dir(self) -> Path:
        """Output directory path for generated dataset."""
        return Path(__file__).parent.parent / "data" / self.dataset_name
