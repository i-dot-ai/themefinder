"""UK demographic presets and sampling utilities for synthetic data generation."""

import numpy as np

from synthetic.config import DemographicField


def get_uk_demographic_presets() -> list[DemographicField]:
    """Return standard UK demographic field presets.

    Based on ONS data and UK government consultation standards.
    Fields marked enabled=True are included by default.

    Returns:
        List of DemographicField presets for UK consultations.
    """
    return [
        DemographicField(
            name="region",
            display_name="Do you live in:",
            values=["England", "Scotland", "Wales", "Northern Ireland"],
            distribution=[0.84, 0.08, 0.05, 0.03],
            enabled=True,
        ),
        DemographicField(
            name="age_group",
            display_name="What is your age group?",
            values=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            distribution=[0.11, 0.17, 0.17, 0.18, 0.17, 0.20],
            enabled=True,
        ),
        DemographicField(
            name="respondent_type",
            display_name="Which of the following best describes how you are responding to this consultation. Are you responding:",
            values=["As an individual", "As an organisation"],
            distribution=[0.85, 0.15],
            enabled=True,
        ),
        DemographicField(
            name="disability",
            display_name="Do you consider yourself to have a health condition or a disability?",
            values=["No", "Yes", "Prefer not to say"],
            distribution=[0.78, 0.17, 0.05],
            enabled=False,
        ),
        DemographicField(
            name="gender",
            display_name="What is your gender?",
            values=["Male", "Female", "Non-binary", "Prefer not to say"],
            distribution=[0.49, 0.49, 0.01, 0.01],
            enabled=False,
        ),
        DemographicField(
            name="employment",
            display_name="What is your employment status?",
            values=[
                "Employed full-time",
                "Employed part-time",
                "Self-employed",
                "Unemployed",
                "Retired",
                "Student",
                "Other",
            ],
            distribution=[0.40, 0.12, 0.13, 0.04, 0.21, 0.07, 0.03],
            enabled=False,
        ),
    ]


def sample_demographics(
    fields: list[DemographicField],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    """Sample demographic profiles for n respondents.

    Args:
        fields: List of demographic fields (only enabled ones are sampled).
        n_samples: Number of profiles to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of dicts mapping display_name to sampled value.
    """
    enabled_fields = [f for f in fields if f.enabled]
    profiles = []

    for _ in range(n_samples):
        profile = {}
        for demographic_field in enabled_fields:
            profile[demographic_field.display_name] = rng.choice(
                demographic_field.values,
                p=demographic_field.distribution,
            )
        profiles.append(profile)

    return profiles


def calculate_stance_modifier(
    profile: dict[str, str],
    fields: list[DemographicField],
) -> float:
    """Calculate total stance modifier for a profile based on policy context.

    Args:
        profile: Sampled profile mapping display_name to value.
        fields: All demographic fields (including policy context).

    Returns:
        Total stance modifier (-1.0 to +1.0), clamped.
    """
    total_modifier = 0.0

    for field in fields:
        if not field.is_policy_context or not field.stance_modifiers:
            continue

        # Find which value was sampled for this field
        sampled_value = profile.get(field.display_name)
        if sampled_value is None:
            continue

        try:
            value_index = field.values.index(sampled_value)
            total_modifier += field.stance_modifiers[value_index]
        except ValueError:
            # Value not found, skip
            continue

    # Clamp to reasonable range
    return max(-0.5, min(0.5, total_modifier))
