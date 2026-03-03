"""Policy context field generation using LLM for synthetic consultation datasets.

Generates policy-specific respondent context questions (e.g., "Do you have a student loan?")
based on the consultation topic. These complement fixed demographics with topic-relevant
characteristics that influence how respondents answer.
"""

import asyncio
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from synthetic.config import DemographicField, QuestionConfig
from structured_output import with_structured_output

logger = logging.getLogger(__name__)

# Retry configuration for transient LLM errors
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


class ContextFieldOption(BaseModel):
    """Single option for a context field."""

    value: str = Field(description="The option text (e.g., 'Yes - currently repaying')")
    distribution_percent: float = Field(
        description="Estimated UK population percentage (0-100)"
    )
    stance_influence: float = Field(
        description="How this option affects stance: -0.1 (oppose) to +0.1 (support), 0 for neutral"
    )


class GeneratedContextField(BaseModel):
    """LLM-generated policy context field."""

    name: str = Field(
        description="Internal identifier (snake_case, e.g., 'student_loan_status')"
    )
    question_text: str = Field(
        description="Question as it would appear in a survey (e.g., 'Do you currently have a student loan?')"
    )
    options: list[ContextFieldOption] = Field(
        description="Available response options with distributions and stance influences"
    )
    rationale: str = Field(
        description="Brief explanation of why this context is relevant to the policy"
    )


class ContextFieldSet(BaseModel):
    """Complete set of generated context fields."""

    fields: list[GeneratedContextField]


SYSTEM_PROMPT = """You are an expert in UK public consultation design and survey methodology.

Your task is to generate POLICY-SPECIFIC CONTEXT QUESTIONS that would realistically be asked
in a UK government consultation survey to understand respondent backgrounds.

## What Makes a Good Context Question

1. **Directly relevant** to the policy topic - asks about characteristics that would affect
   how someone views the proposal
2. **Realistic distributions** - based on UK population statistics where available
3. **Clear stance influence** - some options naturally correlate with supporting or opposing
   the policy (though keep influences subtle: ±0.1 max)
4. **Inclusive options** - always include "Prefer not to say" where appropriate

## Stance Influence Guidelines

- Use small values: -0.1 to +0.1 (subtle influence, not deterministic)
- Positive = tends to support the policy
- Negative = tends to oppose the policy
- Zero = neutral (no influence on stance)

Example for "stopping interest on student loans":
- "Currently repaying student loan" → +0.08 (tends to support)
- "Never had student loan" → -0.05 (slightly tends to oppose)
- "Paid off student loan" → 0.0 (neutral - could go either way)

## Distribution Guidelines

- Distributions must sum to 100%
- Base on realistic UK statistics where possible
- Include reasonable estimates where data unavailable

## Output Format

Generate 3-5 context fields that capture the most important respondent characteristics
for understanding perspectives on this policy."""


def _get_context_llm(callbacks: list | None = None) -> AzureChatOpenAI:
    """Create LLM instance for context field generation.

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
        reasoning_effort="medium",
        callbacks=callbacks or [],
        timeout=600,  # 10 minute timeout to prevent indefinite hangs (reasoning can be slow)
    )


async def generate_context_fields(
    topic: str,
    questions: list[QuestionConfig],
    n_fields: int = 4,
    callbacks: list | None = None,
) -> list[DemographicField]:
    """Generate policy-specific context fields using LLM.

    Args:
        topic: Consultation topic.
        questions: List of consultation questions (for context).
        n_fields: Target number of context fields to generate (3-5).
        callbacks: LangChain callbacks for tracing.

    Returns:
        List of DemographicField objects with is_policy_context=True.
    """
    llm = _get_context_llm(callbacks)
    structured_llm = with_structured_output(llm, ContextFieldSet)

    # Format questions for context
    questions_text = "\n".join(f"{q.number}. {q.text}" for q in questions)

    human_prompt = f"""Generate {n_fields} policy-specific context questions for this UK government consultation.

## Consultation Topic
{topic}

## Consultation Questions
{questions_text}

## Your Task
1. Identify 3-5 respondent characteristics that are DIRECTLY RELEVANT to this policy
2. For each, create a survey question with realistic UK population distributions
3. Estimate subtle stance influences for each option (±0.1 max)
4. Explain why each context is relevant

Focus on characteristics that would genuinely affect how someone views this policy.
Avoid generic demographics (age, region, etc.) - those are handled separately."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
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
                        f"Retryable error ({error_type}) in context field generation, "
                        f"attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
            raise

    if result is None:
        raise last_error  # type: ignore[misc]

    # Convert to DemographicField objects
    demographic_fields = []
    for gen_field in result.fields:
        # Normalise distributions to sum to 1.0
        total_pct = sum(opt.distribution_percent for opt in gen_field.options)
        distribution = [
            opt.distribution_percent / total_pct for opt in gen_field.options
        ]

        demographic_fields.append(
            DemographicField(
                name=gen_field.name,
                display_name=gen_field.question_text,
                values=[opt.value for opt in gen_field.options],
                distribution=distribution,
                enabled=True,
                is_policy_context=True,
                stance_modifiers=[opt.stance_influence for opt in gen_field.options],
            )
        )

    return demographic_fields


async def regenerate_context_fields(
    topic: str,
    questions: list[QuestionConfig],
    feedback: str,
    n_fields: int = 4,
    callbacks: list | None = None,
) -> list[DemographicField]:
    """Regenerate context fields based on user feedback.

    Args:
        topic: Consultation topic.
        questions: List of consultation questions.
        feedback: User's feedback on what to change.
        n_fields: Target number of context fields.
        callbacks: LangChain callbacks for tracing.

    Returns:
        List of regenerated DemographicField objects.
    """
    llm = _get_context_llm(callbacks)
    structured_llm = with_structured_output(llm, ContextFieldSet)

    questions_text = "\n".join(f"{q.number}. {q.text}" for q in questions)

    human_prompt = f"""Regenerate policy-specific context questions for this UK government consultation,
taking into account the user's feedback.

## Consultation Topic
{topic}

## Consultation Questions
{questions_text}

## User Feedback
{feedback}

## Your Task
Generate {n_fields} improved context questions that address the feedback.
Focus on characteristics directly relevant to this policy."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    # Retry loop
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
                        f"Retryable error ({error_type}) in context regeneration, "
                        f"attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
            raise

    if result is None:
        raise last_error  # type: ignore[misc]

    # Convert to DemographicField objects
    demographic_fields = []
    for gen_field in result.fields:
        total_pct = sum(opt.distribution_percent for opt in gen_field.options)
        distribution = [
            opt.distribution_percent / total_pct for opt in gen_field.options
        ]

        demographic_fields.append(
            DemographicField(
                name=gen_field.name,
                display_name=gen_field.question_text,
                values=[opt.value for opt in gen_field.options],
                distribution=distribution,
                enabled=True,
                is_policy_context=True,
                stance_modifiers=[opt.stance_influence for opt in gen_field.options],
            )
        )

    return demographic_fields
