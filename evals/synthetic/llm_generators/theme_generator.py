"""Theme generation using LLM for synthetic consultation datasets.

Uses gpt-5-mini with medium reasoning for comprehensive theme discovery
across diverse demographic perspectives.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from synthetic.config import DemographicField


class Theme(BaseModel):
    """Single theme definition for structured output."""

    topic_id: str = Field(
        description="Theme identifier (A-Z, then AA-AZ for extended sets)"
    )
    topic_label: str = Field(description="Short theme name (2-5 words)")
    topic_description: str = Field(
        description="Detailed description explaining what responses in this theme discuss (1-3 sentences)"
    )


class ThemeSet(BaseModel):
    """Complete theme set returned by the LLM."""

    themes: list[Theme]


SYSTEM_PROMPT = """You are an expert analyst in UK public consultations and policy engagement.

Your task is to generate a COMPREHENSIVE set of themes that would realistically emerge from
public responses to a government consultation question.

## Demographic Perspectives to Consider in Your Analysis
Think deeply about how different groups would respond differently before generating themes:
- Age groups: Young adults (18-24) vs working age (25-54) vs retirees (65+)
- UK nations: England, Scotland, Wales, Northern Ireland - each with distinct policy contexts
- Urban vs rural residents
- Socioeconomic backgrounds: Different income levels, employment situations
- Those directly affected vs general public
- Individuals vs organisations/professional bodies
- People with disabilities or health conditions
- Different ethnic and cultural backgrounds

## Theme Categories to Cover
Ensure your themes span these categories where relevant:
- **Support themes**: Various reasons people agree with the proposal
- **Opposition themes**: Various reasons people disagree
- **Conditional themes**: "Yes, but..." or "Only if..." positions
- **Practical concerns**: Implementation challenges, costs, timelines
- **Stakeholder-specific impacts**: Effects on particular groups
- **Alternative proposals**: Different approaches people might suggest
- **Unintended consequences**: Concerns about knock-on effects
- **Ideological positions**: Principled stances (fairness, freedom, responsibility)
- **Evidence-based concerns**: Citing research, data, or precedents
- **Personal experience themes**: Based on lived experience

## Quality Requirements
- Each theme must be DISTINCT (no significant overlaps)
- Themes should be SPECIFIC to this policy topic
- Cover the FULL SPECTRUM of likely opinion
- Be REALISTIC about what UK citizens actually write in consultations
- Consider MINORITY viewpoints that may be less common but important

Generate as many themes as needed to comprehensively cover the topic. For simple questions,
this might be 10-15 themes. For complex, contentious topics, you may need 30-50+ themes.
Do not artificially limit yourself - be thorough."""


def _get_theme_generation_llm(callbacks: list | None = None) -> AzureChatOpenAI:
    """Create LLM instance optimised for theme generation.

    Uses gpt-5-mini with medium reasoning level for comprehensive analysis.

    Args:
        callbacks: LangChain callbacks for tracing.

    Returns:
        Configured AzureChatOpenAI instance.
    """
    return AzureChatOpenAI(
        azure_deployment="gpt-5-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
        reasoning_effort="high",
        callbacks=callbacks or [],
    )


def _generate_topic_ids(n: int) -> list[str]:
    """Generate topic IDs for n themes.

    Uses A-Z for first 26, then AA-AZ, BA-BZ, etc.

    Args:
        n: Number of IDs needed.

    Returns:
        List of topic ID strings.
    """
    ids = []
    for i in range(n):
        if i < 26:
            ids.append(chr(65 + i))  # A-Z
        else:
            # AA, AB, ..., AZ, BA, BB, ...
            first = chr(65 + ((i - 26) // 26))
            second = chr(65 + ((i - 26) % 26))
            ids.append(first + second)
    return ids


async def generate_themes(
    topic: str,
    question: str,
    demographic_fields: list[DemographicField],
    callbacks: list | None = None,
) -> list[dict]:
    """Generate comprehensive themes for a consultation question.

    Uses gpt-5-mini with medium reasoning to deeply analyse the policy topic
    and generate themes covering diverse demographic perspectives.

    Args:
        topic: Overall consultation topic.
        question: Specific question text.
        demographic_fields: Enabled demographic fields for perspective consideration.
        callbacks: LangChain callbacks for tracing.

    Returns:
        List of theme dicts with topic_id, topic_label, topic_description, topic.
    """
    llm = _get_theme_generation_llm(callbacks)
    structured_llm = llm.with_structured_output(ThemeSet)

    # Build demographic context for the prompt (guides reasoning, not output)
    demographic_context = _build_demographic_context(demographic_fields)

    human_prompt = f"""Analyse this UK government consultation question and generate a comprehensive theme framework.

## Consultation Topic
{topic}

## Question
{question}

## Demographic Context for This Consultation
The following demographic dimensions are being tracked for respondents. Consider how each group might respond differently:
{demographic_context}

## Your Task
1. Reason through the policy question - what are the key tensions, trade-offs, and stakeholder interests?
2. Consider how different demographic groups would approach this question differently
3. Generate a COMPREHENSIVE set of themes covering all likely response patterns
4. Ensure themes capture perspectives from across the demographic spectrum

Generate as many themes as the topic requires for comprehensive coverage. Simple questions
may need 10-15 themes; complex or contentious topics may need 30-50+. Do not artificially
limit the number - be thorough.

Use sequential IDs: A, B, C, ... Z, AA, AB, ... for themes."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    result = await structured_llm.ainvoke(messages)

    # Normalise topic IDs to ensure sequential ordering
    topic_ids = _generate_topic_ids(len(result.themes))

    # Build theme list with combined topic field
    themes = [
        {
            "topic_id": topic_ids[i],
            "topic_label": t.topic_label,
            "topic_description": t.topic_description,
            "topic": f"{t.topic_label}: {t.topic_description}",
        }
        for i, t in enumerate(result.themes)
    ]

    # Add special themes X and Y (always required per spec)
    themes.extend(
        [
            {
                "topic_id": "X",
                "topic_label": "None of the Above",
                "topic_description": "The response discusses a topic not covered by the listed themes (only use this if no other theme applies).",
                "topic": "None of the Above: The response discusses a topic not covered by the listed themes (only use this if no other theme applies).",
            },
            {
                "topic_id": "Y",
                "topic_label": "No Reason Given",
                "topic_description": "The response does not provide a substantive answer to the question.",
                "topic": "No Reason Given: The response does not provide a substantive answer to the question.",
            },
        ]
    )

    return themes


def _build_demographic_context(fields: list[DemographicField]) -> str:
    """Build demographic context string for the prompt.

    Args:
        fields: List of demographic fields (enabled ones).

    Returns:
        Formatted string describing demographic dimensions.
    """
    enabled = [f for f in fields if f.enabled]

    if not enabled:
        return "No specific demographic dimensions tracked - consider general UK population diversity."

    lines = []
    for field in enabled:
        values_str = ", ".join(field.values)
        lines.append(f"- **{field.display_name}**: {values_str}")

    return "\n".join(lines)
