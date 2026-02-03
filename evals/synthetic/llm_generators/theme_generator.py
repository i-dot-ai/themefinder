"""Theme generation using LLM for synthetic consultation datasets.

Uses fan-out parallelisation (10 calls) for diverse theme discovery,
then consolidates with light-touch rationalisation.
"""

import asyncio
import logging
import os
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from synthetic.config import DemographicField
from themefinder.structured_output import with_structured_output

logger = logging.getLogger(__name__)

# Number of parallel theme generation calls for diversity
FAN_OUT_COUNT = 10

# Retry configuration for transient LLM errors
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


class Theme(BaseModel):
    """Single theme definition for structured output."""

    topic_id: str = Field(
        description="Theme identifier (A-Z, then AA-AZ for extended sets)"
    )
    topic_label: str = Field(description="Short theme name (2-5 words)")
    topic_description: str = Field(
        description="Concise description of what responses in this theme argue (15-25 words max)"
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

## Description Format
- topic_label: 2-5 words (e.g., "Fiscal cost concerns")
- topic_description: ONE concise sentence, 15-25 words max
  - Good: "Opposition citing large Exchequer cost and pressure on public services"
  - Bad: "Submissions emphasising the large direct cost to the Exchequer, potential increases in public borrowing and the opportunity cost for other public services. Critics in this theme demand robust costing..."

Generate as many themes as needed to comprehensively cover the topic. For simple questions,
this might be 10-15 themes. For complex, contentious topics, you may need 30-50+ themes.
Do not artificially limit yourself - be thorough."""

CONSOLIDATION_SYSTEM_PROMPT = """You are an expert at consolidating and deduplicating theme lists.

You will receive a large list of themes generated from multiple parallel analyses of the same
consultation question. Your task is to:

1. **Remove exact or near-duplicates** - themes that express the same idea
2. **Merge highly similar themes** - combine themes that overlap significantly into one
3. **Preserve diversity** - keep distinct viewpoints even if only mentioned once
4. **Maintain quality** - ensure each final theme is clear and well-described

## Important Guidelines
- Be CONSERVATIVE with merging - when in doubt, keep themes separate
- Preserve minority/niche viewpoints - these are valuable for realistic consultation data
- Keep the original wording where possible - don't over-edit
- Aim for comprehensive coverage over conciseness

## Output Format
- topic_label: 2-5 words
- topic_description: ONE concise sentence, 15-25 words max
- Use sequential IDs: A, B, C, ... Z, AA, AB, ..."""


def _get_generation_llm(
    reasoning_effort: str = "medium",
    callbacks: list | None = None,
) -> AzureChatOpenAI:
    """Create LLM instance for theme generation.

    Args:
        reasoning_effort: Reasoning level (low, medium, high).
        callbacks: LangChain callbacks for tracing.

    Returns:
        Configured AzureChatOpenAI instance.
    """
    return AzureChatOpenAI(
        azure_deployment="gpt-5-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
        reasoning_effort=reasoning_effort,
        callbacks=callbacks or [],
        timeout=600,  # 10 minute timeout to prevent indefinite hangs (reasoning can be slow)
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
    on_fan_out_complete: Callable[[], None] | None = None,
) -> list[dict]:
    """Generate comprehensive themes for a consultation question.

    Uses fan-out parallelisation: 10 concurrent LLM calls generate diverse themes,
    then a consolidation step deduplicates and rationalises them.

    Args:
        topic: Overall consultation topic.
        question: Specific question text.
        demographic_fields: Enabled demographic fields for perspective consideration.
        callbacks: LangChain callbacks for tracing.
        on_fan_out_complete: Callback invoked after each fan-out call completes.

    Returns:
        List of theme dicts with topic_id, topic_label, topic_description, topic.
    """
    # Step 1: Fan-out - 10 parallel calls for diverse theme discovery
    raw_themes = await _fan_out_theme_generation(
        topic=topic,
        question=question,
        demographic_fields=demographic_fields,
        callbacks=callbacks,
        on_complete=on_fan_out_complete,
    )

    # Step 2: Consolidate - deduplicate and rationalise
    consolidated_themes = await _consolidate_themes(
        raw_themes=raw_themes,
        topic=topic,
        question=question,
        callbacks=callbacks,
    )

    # Add special themes X and Y (always required per spec)
    consolidated_themes.extend(
        [
            {
                "topic_id": "X",
                "topic_label": "None of the Above",
                "topic_description": "Response discusses a topic not covered by listed themes",
                "topic": "None of the Above: Response discusses a topic not covered by listed themes",
            },
            {
                "topic_id": "Y",
                "topic_label": "No Reason Given",
                "topic_description": "Response does not provide a substantive answer",
                "topic": "No Reason Given: Response does not provide a substantive answer",
            },
        ]
    )

    return consolidated_themes


async def _fan_out_theme_generation(
    topic: str,
    question: str,
    demographic_fields: list[DemographicField],
    callbacks: list | None = None,
    on_complete: Callable[[], None] | None = None,
) -> list[dict]:
    """Generate themes with fan-out parallelisation.

    Runs FAN_OUT_COUNT concurrent LLM calls to discover diverse themes.

    Args:
        topic: Consultation topic.
        question: Question text.
        demographic_fields: Demographic fields for context.
        callbacks: LangChain callbacks.
        on_complete: Callback for each completed call.

    Returns:
        Combined list of all themes from all calls (with duplicates).
    """
    llm = _get_generation_llm(reasoning_effort="medium", callbacks=callbacks)
    structured_llm = with_structured_output(llm, ThemeSet)

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

    async def single_call(call_id: int) -> list[dict]:
        """Execute a single theme generation call with retry logic."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                result = await structured_llm.ainvoke(messages)
                themes = [
                    {
                        "topic_label": t.topic_label,
                        "topic_description": t.topic_description,
                    }
                    for t in result.themes
                ]
                if on_complete:
                    on_complete()
                return themes
            except (ValidationError, Exception) as e:
                last_error = e
                error_type = type(e).__name__

                # Check if it's a retryable error
                is_validation_error = isinstance(e, ValidationError)
                is_connection_error = (
                    "ECONNRESET" in str(e)
                    or "ENOTFOUND" in str(e)
                    or "ECONNREFUSED" in str(e)
                    or "DNS" in str(e)
                    or "connection" in str(e).lower()
                )

                if is_validation_error or is_connection_error:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY_SECONDS * (2**attempt)
                        logger.warning(
                            f"Retryable error ({error_type}) in theme fan-out call {call_id}, "
                            f"attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                raise

        raise last_error  # type: ignore[misc]

    # Run FAN_OUT_COUNT calls in parallel - return_exceptions prevents one failure cancelling all
    tasks = [asyncio.create_task(single_call(i)) for i in range(FAN_OUT_COUNT)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failed calls and log errors
    all_themes = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                f"Theme fan-out call {i} failed: {type(result).__name__}: {result}"
            )
        else:
            all_themes.extend(result)

    if not all_themes:
        raise RuntimeError("All theme generation calls failed")

    return all_themes


async def _consolidate_themes(
    raw_themes: list[dict],
    topic: str,
    question: str,
    callbacks: list | None = None,
) -> list[dict]:
    """Consolidate and deduplicate themes from fan-out generation.

    Uses a light-touch LLM call to merge duplicates while preserving diversity.

    Args:
        raw_themes: Combined themes from all fan-out calls.
        topic: Consultation topic.
        question: Question text.
        callbacks: LangChain callbacks.

    Returns:
        Deduplicated and rationalised theme list.
    """
    # Use low reasoning for consolidation - it's mostly deduplication
    llm = _get_generation_llm(reasoning_effort="low", callbacks=callbacks)
    structured_llm = with_structured_output(llm, ThemeSet)

    # Format raw themes for the prompt
    themes_text = "\n".join(
        f"- {t['topic_label']}: {t['topic_description']}" for t in raw_themes
    )

    human_prompt = f"""## Consultation Context
Topic: {topic}
Question: {question}

## Raw Themes to Consolidate
The following {len(raw_themes)} themes were generated from multiple parallel analyses.
Consolidate them by removing duplicates and merging highly similar themes.

{themes_text}

## Your Task
1. Remove exact or near-duplicate themes
2. Merge themes that express essentially the same viewpoint
3. Preserve distinct minority viewpoints - don't over-consolidate
4. Output a clean, deduplicated theme list

Be CONSERVATIVE - when in doubt, keep themes separate. Diversity is valuable."""

    messages = [
        SystemMessage(content=CONSOLIDATION_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    # Retry loop for transient LLM errors
    result = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            result = await structured_llm.ainvoke(messages)
            break
        except (ValidationError, Exception) as e:
            last_error = e
            error_type = type(e).__name__

            is_validation_error = isinstance(e, ValidationError)
            is_connection_error = (
                "ECONNRESET" in str(e)
                or "ENOTFOUND" in str(e)
                or "ECONNREFUSED" in str(e)
                or "DNS" in str(e)
                or "connection" in str(e).lower()
            )

            if is_validation_error or is_connection_error:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_SECONDS * (2**attempt)
                    logger.warning(
                        f"Retryable error ({error_type}) in theme consolidation, "
                        f"attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
            raise

    if result is None:
        raise last_error  # type: ignore[misc]

    # Normalise topic IDs to ensure sequential ordering
    topic_ids = _generate_topic_ids(len(result.themes))

    # Build final theme list
    themes = [
        {
            "topic_id": topic_ids[i],
            "topic_label": t.topic_label,
            "topic_description": t.topic_description,
            "topic": f"{t.topic_label}: {t.topic_description}",
        }
        for i, t in enumerate(result.themes)
    ]

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
