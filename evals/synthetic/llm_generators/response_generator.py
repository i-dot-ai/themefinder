"""Response generation using LLM for synthetic consultation datasets."""

import random
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from synthetic.config import NoiseLevel, ResponseSpec


class GeneratedResponse(BaseModel):
    """LLM-generated consultation response."""

    response: str = Field(description="The consultation response text")
    sentiment: str = Field(description="AGREE, DISAGREE, or UNCLEAR")
    evidence_rich: bool = Field(
        description="True if response contains specific evidence, examples, or personal experience"
    )


SYSTEM_PROMPT_TEMPLATE = """You are simulating a member of the UK public responding to a government consultation.

Persona:
{persona_desc}

Generate a realistic consultation response with these characteristics:
- Length: approximately {min_words}-{max_words} words
- Response type: {response_type}
- Write naturally as this persona would, including their likely vocabulary and concerns

Guidelines by response type:
- agree: Express clear support for the proposal with reasons
- disagree: Express clear opposition with reasons
- nuanced: Show conditional support with specific concerns or caveats
- off_topic: Drift to tangentially related issues, miss the main question
- low_quality: Be vague, very brief, or unclear

Generate authentic-sounding responses. Vary sentence structure and vocabulary."""


async def generate_response_batch(
    llm: AzureChatOpenAI,
    question: str,
    themes: list[dict],
    specs: list[ResponseSpec],
    noise_level: NoiseLevel,
    callbacks: list | None = None,
    on_response_complete: Callable[[], None] | None = None,
) -> list[dict]:
    """Generate a batch of responses from specifications (parallelised).

    Uses asyncio.as_completed for real-time progress tracking as each
    response finishes, rather than waiting for the entire batch.

    Args:
        llm: Azure OpenAI LLM instance.
        question: Consultation question text.
        themes: Full theme list for reference.
        specs: List of response specifications.
        noise_level: Noise injection intensity.
        callbacks: LangChain callbacks for tracing.
        on_response_complete: Callback invoked when each response finishes.

    Returns:
        List of response dicts with all required fields.
    """
    import asyncio

    structured_llm = llm.with_structured_output(GeneratedResponse)
    config = {"callbacks": callbacks} if callbacks else {}

    async def generate_single(spec: ResponseSpec) -> dict:
        """Generate a single response."""
        theme_context = _build_theme_context(spec, themes)
        system_prompt = _build_system_prompt(spec)
        human_prompt = _build_human_prompt(question, theme_context, spec)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = await structured_llm.ainvoke(messages, config=config)

        final_text = response.response
        if spec.apply_noise and spec.noise_type:
            final_text = _apply_noise(final_text, spec.noise_type, noise_level)

        return {
            "response_id": spec.response_id,
            "response": final_text,
            "position": response.sentiment,
            "evidence_rich": "YES" if response.evidence_rich else "NO",
            "labels": spec.themes,
            "stances": spec.stances,
        }

    # Create tasks for all specs
    tasks = [asyncio.create_task(generate_single(spec)) for spec in specs]

    # Use as_completed to track progress as each response finishes
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if on_response_complete:
            on_response_complete()

    return results


def _build_theme_context(spec: ResponseSpec, themes: list[dict]) -> list[str]:
    """Build theme context lines for prompt.

    Args:
        spec: Response specification.
        themes: Full theme list.

    Returns:
        List of formatted theme+stance strings.
    """
    theme_context = []
    for tid, stance in zip(spec.themes, spec.stances):
        theme = next((t for t in themes if t["topic_id"] == tid), None)
        if theme:
            theme_context.append(f"- {theme['topic_label']} (stance: {stance})")
    return theme_context


def _build_system_prompt(spec: ResponseSpec) -> str:
    """Build system prompt for response generation.

    Args:
        spec: Response specification.

    Returns:
        Formatted system prompt.
    """
    length_range = spec.length.value
    persona_desc = ", ".join(f"{k}: {v}" for k, v in spec.persona.items())

    return SYSTEM_PROMPT_TEMPLATE.format(
        persona_desc=persona_desc,
        min_words=length_range[0],
        max_words=length_range[1],
        response_type=spec.response_type.value,
    )


def _build_human_prompt(
    question: str,
    theme_context: list[str],
    spec: ResponseSpec,
) -> str:
    """Build human prompt for specific response.

    Args:
        question: Consultation question.
        theme_context: Formatted theme+stance lines.
        spec: Response specification.

    Returns:
        Formatted human prompt.
    """
    themes_str = "\n".join(theme_context)

    evidence_instruction = ""
    if spec.length.value[0] >= 51:  # Medium or long responses
        evidence_instruction = "\nInclude specific examples, personal experience, or evidence where appropriate."

    return f"""Consultation Question: {question}

Your response should address these themes with the indicated stance:
{themes_str}

Write your consultation response now.{evidence_instruction}"""


def _apply_noise(text: str, noise_type: str, level: NoiseLevel) -> str:
    """Apply noise injection to response text.

    Args:
        text: Original response text.
        noise_type: Type of noise to apply.
        level: Noise intensity level.

    Returns:
        Text with noise applied.
    """
    intensity = {
        NoiseLevel.LOW: 0.3,
        NoiseLevel.MEDIUM: 0.5,
        NoiseLevel.HIGH: 0.8,
    }[level]

    if noise_type == "typo":
        return _inject_typos(text, intensity)
    elif noise_type == "grammar":
        return _inject_grammar_errors(text, intensity)
    elif noise_type == "caps":
        return text.upper()
    elif noise_type == "emotional":
        return _inject_emotional_language(text)
    elif noise_type == "sarcasm":
        return _add_sarcasm_markers(text)

    return text


def _inject_typos(text: str, intensity: float) -> str:
    """Inject realistic typos into text.

    Args:
        text: Original text.
        intensity: How aggressively to inject typos (0-1).

    Returns:
        Text with typos.
    """
    words = text.split()
    n_typos = max(1, int(len(words) * intensity * 0.1))

    typo_patterns = [
        lambda w: w[:-1] if len(w) > 3 else w,  # Missing last letter
        lambda w: w + w[-1] if len(w) > 2 else w,  # Doubled letter
        lambda w: w[:1] + w[1:].replace("e", "r", 1)
        if "e" in w[1:]
        else w,  # Adjacent key
        lambda w: w.replace("th", "hte", 1) if "th" in w else w,  # Common transposition
    ]

    for _ in range(n_typos):
        if words:
            idx = random.randint(0, len(words) - 1)
            pattern = random.choice(typo_patterns)
            words[idx] = pattern(words[idx])

    return " ".join(words)


def _inject_grammar_errors(text: str, intensity: float) -> str:
    """Inject grammar errors into text.

    Args:
        text: Original text.
        intensity: How aggressively to inject errors (0-1).

    Returns:
        Text with grammar errors.
    """
    if intensity > 0.5:
        # Remove some punctuation
        text = text.replace(",", "", 1)

    if intensity > 0.7:
        # Common grammar mistakes
        replacements = [
            ("there is", "theres"),
            ("they are", "their"),
            ("should have", "should of"),
            ("could have", "could of"),
        ]
        for old, new in replacements:
            if old in text.lower():
                text = text.replace(old, new, 1)
                break

    return text


def _inject_emotional_language(text: str) -> str:
    """Add emotional emphasis to text.

    Args:
        text: Original text.

    Returns:
        Text with emotional language.
    """
    emphatics = [
        "Absolutely ",
        "I really think ",
        "It's outrageous that ",
        "I strongly believe ",
    ]

    # Add emphatic opener
    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    prefix = random.choice(emphatics)
    text = prefix.lower() + text[0].lower() + text[1:]

    # Add exclamation
    if text.endswith("."):
        text = text[:-1] + "!"

    return text


def _add_sarcasm_markers(text: str) -> str:
    """Add sarcasm indicators to text.

    Args:
        text: Original text.

    Returns:
        Text with sarcasm markers.
    """
    sarcastic_additions = [
        " (as if that will ever happen)",
        " - what a surprise",
        " Obviously...",
    ]

    # Find a sentence to add sarcasm to
    sentences = text.split(". ")
    if len(sentences) > 1:
        idx = random.randint(0, len(sentences) - 2)
        sentences[idx] += random.choice(sarcastic_additions)
        text = ". ".join(sentences)

    return text
