"""SystemOne (TypeSafe jev) implementations of themefinder's classification stages.

The jev SystemOne model answers typed questions (noul = yes/no probability,
choice, score) about a piece of state. It is not generative, so only the
classification-shaped stages of the themefinder pipeline transfer:

- theme mapping: one noul question per theme, batched into a single call per
  response ("does this response express this topic?")
- detail detection: one noul question per response ("is this response
  evidence-rich?")

The generative stages (theme generation, condensation, refinement) remain on
the LLM; ``find_themes_hybrid`` combines both.

Every answer comes back with a calibrated probability, so callers can tune the
assignment threshold or route low-confidence answers to human review.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    stop_after_attempt,
    wait_random_exponential,
)

from themefinder.llm import LLM
from themefinder.prompts import CONSULTATION_SYSTEM_PROMPT
from themefinder.tasks import (
    theme_condensation,
    theme_generation,
    theme_refinement,
)
from themefinder.themefinder_logging import logger

DEFAULT_ASSIGNMENT_THRESHOLD = 0.5

# Retry policy for SystemOne calls, mirroring the LLM batch processor.
RETRY_ATTEMPTS = 6
RETRY_MIN_WAIT_SECONDS = 1
RETRY_MAX_WAIT_SECONDS = 20

THEME_QUESTION_PREFIX = "theme_"
GIVES_REASON_KEY = "gives_reason"
FALLBACK_OTHER = "Other"
FALLBACK_NO_REASON = "No Reason Given"

THEME_MAPPING_INSTRUCTIONS = (
    "The state contains a consultation question and one free-text response to it. "
    "Does the response express the following topic? The response does not need to "
    "use the same wording as the topic; it is a match if it expresses a similar "
    "sentiment or point of view. Topic: {topic}"
)

MAPPING_CHOICE_KEY = "themes"
CHOICE_MAPPING_INSTRUCTIONS = (
    "The state contains a consultation question and one free-text response to it. "
    "Which topic does the response most clearly express? A response matches a topic "
    "if it expresses a similar sentiment or point of view; exact wording is not "
    "required."
)
FALLBACK_CRITERIA = {
    "Other": (
        "The response gives a substantive opinion or reason, but none of the "
        "listed topics apply."
    ),
    "No Reason Given": (
        "The response gives no substantive reason or opinion in answer to the "
        "question (it is empty, off-topic or a refusal to answer)."
    ),
}

GIVES_REASON_INSTRUCTIONS = (
    "The state contains a consultation question and one free-text response to it. "
    "Does the response give any substantive opinion, reason or argument in answer "
    "to the question, as opposed to being empty, off-topic or a refusal to answer?"
)

EVIDENCE_RICH_INSTRUCTIONS = (
    "The state contains a consultation question and one free-text response to it. "
    "Is the response evidence-rich? A response is evidence-rich only if it clearly "
    "answers the question with insights beyond generic opinion (nuanced reasoning, "
    "contextual explanation or argumentation that could inform decision-making) "
    "AND it contains substantive evidence: specific verifiable facts or data "
    "(statistics, dates, named reports or studies), concrete illustrative examples "
    "that support a broader claim, or detailed personal or professional experiences "
    "with contextual information such as roles, locations or timelines."
)

EVIDENCE_RICH_CRITERIA = (
    "Yes means the response would provide useful input to someone drafting policy, "
    "beyond what is already commonly known or expected. No means the response uses "
    "vague language with no supporting detail, restates commonly known points, or "
    "shares anecdotes without sufficient context or a clear takeaway."
)


@dataclass
class SystemOneUsage:
    """Accumulated token usage across SystemOne calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def record(self, usage: Any) -> None:
        """Add the usage from a single SystemOne response."""
        self.requests += 1
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0


@runtime_checkable
class SystemOneTransport(Protocol):
    """Anything with an async ``system_one(state=..., questions=...)`` method.

    Satisfied by ``typesafe_sdk.AsyncTypeSafeClient`` and by test fakes.
    """

    async def system_one(self, state: Any, questions: dict[str, Any]) -> Any: ...


@dataclass
class SystemOne:
    """A SystemOne client wrapper that tracks token usage across calls.

    Construct with an explicit transport (e.g. a test fake), or use
    :meth:`from_env` to build one backed by the TypeSafe SDK, which reads
    ``TYPESAFE_API_KEY`` from the environment.
    """

    transport: SystemOneTransport
    usage: SystemOneUsage = field(default_factory=SystemOneUsage)

    @classmethod
    def from_env(cls, model: str | None = None, **client_kwargs) -> "SystemOne":
        """Create a SystemOne client backed by the TypeSafe SDK.

        Args:
            model: Optional model override (defaults to the SDK default, jev-latest).
            **client_kwargs: Passed through to ``AsyncTypeSafeClient``.
        """
        try:
            from typesafe_sdk import AsyncTypeSafeClient
        except ImportError as e:
            raise ImportError(
                "The 'typesafe-sdk' package is required for SystemOne stages. "
                "Install it with the 'systemone' extra: pip install 'themefinder[systemone]'"
            ) from e
        if model is not None:
            client_kwargs["model"] = model
        return cls(transport=AsyncTypeSafeClient(**client_kwargs))

    async def ask(self, state: Any, questions: dict[str, Any]) -> Any:
        """Send one SystemOne request and record its token usage."""
        result = await self.transport.system_one(state=state, questions=questions)
        self.usage.record(getattr(result, "usage", None))
        return result


def _theme_texts(refined_themes_df: pd.DataFrame) -> dict[str, str]:
    """Return {topic_id: topic text} from a refined themes DataFrame.

    Uses the combined ``topic`` column ("label: description") when present,
    otherwise combines ``topic_label`` and ``topic_description``.
    """
    if "topic" in refined_themes_df.columns:
        return dict(
            zip(refined_themes_df["topic_id"], refined_themes_df["topic"])
        )
    return {
        row["topic_id"]: f"{row['topic_label']}: {row['topic_description']}"
        for _, row in refined_themes_df.iterrows()
    }


def _noul(instructions: str, criteria: str | None = None) -> Any:
    """Build a noul (yes/no) SystemOne question."""
    try:
        from typesafe_sdk import Noul
    except ImportError as e:
        raise ImportError(
            "The 'typesafe-sdk' package is required for SystemOne stages. "
            "Install it with the 'systemone' extra: pip install 'themefinder[systemone]'"
        ) from e
    if criteria:
        return Noul(instructions=instructions, criteria=criteria)
    return Noul(instructions=instructions)


def _mapping_questions(theme_texts: dict[str, str]) -> dict[str, Any]:
    """Build the batched noul question set for mapping one response to themes."""
    questions = {
        f"{THEME_QUESTION_PREFIX}{topic_id}": _noul(
            THEME_MAPPING_INSTRUCTIONS.format(topic=topic_text)
        )
        for topic_id, topic_text in theme_texts.items()
    }
    questions[GIVES_REASON_KEY] = _noul(GIVES_REASON_INSTRUCTIONS)
    return questions


def _mapping_choice_question(theme_texts: dict[str, str]) -> dict[str, Any]:
    """Build a single choice question over all themes plus fallback options."""
    try:
        from typesafe_sdk import Choice
    except ImportError as e:
        raise ImportError(
            "The 'typesafe-sdk' package is required for SystemOne stages. "
            "Install it with the 'systemone' extra: pip install 'themefinder[systemone]'"
        ) from e
    return {
        MAPPING_CHOICE_KEY: Choice(
            instructions=CHOICE_MAPPING_INSTRUCTIONS,
            criteria={**theme_texts, **FALLBACK_CRITERIA},
        )
    }


async def _ask_with_retries(
    client: SystemOne, state: Any, questions: dict[str, Any]
) -> Any:
    """Ask SystemOne with the same retry policy as the LLM batch processor."""
    retrying = AsyncRetrying(
        wait=wait_random_exponential(
            min=RETRY_MIN_WAIT_SECONDS, max=RETRY_MAX_WAIT_SECONDS
        ),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    return await retrying(client.ask, state=state, questions=questions)


def _labels_from_nouls(
    result: Any, theme_texts: dict[str, str], threshold: float
) -> tuple[list[str], dict[str, float]]:
    """Extract labels and probabilities from a batched-noul mapping answer."""
    probabilities = {
        topic_id: result.nouls[f"{THEME_QUESTION_PREFIX}{topic_id}"].noul
        for topic_id in theme_texts
    }
    labels = [
        topic_id
        for topic_id, probability in probabilities.items()
        if probability >= threshold
    ]
    if not labels:
        gives_reason = result.nouls[GIVES_REASON_KEY].noul
        labels = [FALLBACK_OTHER if gives_reason >= threshold else FALLBACK_NO_REASON]
    return labels, probabilities


def _labels_from_choice(
    result: Any, theme_texts: dict[str, str], threshold: float
) -> tuple[list[str], dict[str, float]]:
    """Extract labels and probabilities from a single-choice mapping answer.

    The choice's probability distribution spans all themes plus the fallback
    options and sums to one, so a response expressing several themes splits its
    probability mass between them. Every theme at or above the threshold is
    assigned; when none reaches it, the model's top choice is used (which may
    be a fallback option).
    """
    answer = result.choices[MAPPING_CHOICE_KEY]
    probabilities = dict(answer.probabilities)
    labels = [
        topic_id
        for topic_id, probability in probabilities.items()
        if topic_id in theme_texts and probability >= threshold
    ]
    if not labels:
        labels = [answer.choice]
    return labels, probabilities


async def theme_mapping_systemone(
    responses_df: pd.DataFrame,
    client: SystemOne,
    question: str,
    refined_themes_df: pd.DataFrame,
    threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
    concurrency: int = 10,
    question_type: str = "noul",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map survey responses to refined themes using SystemOne questions.

    Two question strategies are supported, one SystemOne call per response
    either way:

    - "noul" (default): one yes/no question per theme plus a fallback
      question, batched into the call. Each theme gets an independent
      calibrated probability, so multi-theme responses are handled naturally.
      Themes at or above the threshold are assigned; when none is, the
      response is labelled "Other" (or "No Reason Given" when it does not
      appear to answer the question at all).
    - "choice": a single choice question whose options are the themes plus
      the fallback options. The answer is a probability distribution over the
      options; themes at or above the threshold are assigned, falling back to
      the model's top choice. Because the distribution sums to one, a lower
      threshold (e.g. 0.25) is appropriate in this mode.

    Args:
        responses_df: DataFrame with 'response_id' and 'response' columns.
        client: SystemOne client wrapper.
        question: The survey question.
        refined_themes_df: DataFrame of refined themes with 'topic_id' and
            'topic' (or 'topic_label' + 'topic_description') columns.
        threshold: Minimum probability for a theme to be assigned.
        concurrency: Maximum number of simultaneous SystemOne calls.
        question_type: "noul" or "choice" (see above).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows).
        The results include a 'labels' column (list of topic_ids, matching the
        LLM stage's output) and a 'theme_probabilities' column with the full
        probability per theme.
    """
    if question_type not in ("noul", "choice"):
        raise ValueError(
            f"question_type must be 'noul' or 'choice', got '{question_type}'"
        )
    logger.info(
        f"Running SystemOne theme mapping ({question_type}) on "
        f"{len(responses_df)} responses using {len(refined_themes_df)} themes"
    )
    theme_texts = _theme_texts(refined_themes_df)
    if question_type == "choice":
        questions = _mapping_choice_question(theme_texts)
        extract_labels = _labels_from_choice
    else:
        questions = _mapping_questions(theme_texts)
        extract_labels = _labels_from_nouls
    semaphore = asyncio.Semaphore(concurrency)

    async def map_one(row: dict) -> dict | None:
        state = {"question": question, "response": row["response"]}
        async with semaphore:
            try:
                result = await _ask_with_retries(client, state, questions)
            except Exception as e:
                logger.warning(
                    f"SystemOne mapping failed for response {row['response_id']}: {e}"
                )
                return None

        labels, probabilities = extract_labels(result, theme_texts, threshold)
        return {
            "response_id": row["response_id"],
            "labels": labels,
            "theme_probabilities": probabilities,
        }

    rows = responses_df.to_dict(orient="records")
    results = await asyncio.gather(*[map_one(row) for row in rows])

    return _merge_results(responses_df, rows, results)


async def detail_detection_systemone(
    responses_df: pd.DataFrame,
    client: SystemOne,
    question: str,
    threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify evidence-rich responses using a SystemOne noul question.

    Args:
        responses_df: DataFrame with 'response_id' and 'response' columns.
        client: SystemOne client wrapper.
        question: The survey question.
        threshold: Minimum probability to classify a response as evidence-rich.
        concurrency: Maximum number of simultaneous SystemOne calls.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows).
        The results include an 'evidence_rich' column ("YES"/"NO", matching the
        LLM stage's output) and an 'evidence_probability' column.
    """
    logger.info(f"Running SystemOne detail detection on {len(responses_df)} responses")
    questions = {
        "evidence_rich": _noul(EVIDENCE_RICH_INSTRUCTIONS, EVIDENCE_RICH_CRITERIA)
    }
    semaphore = asyncio.Semaphore(concurrency)

    async def detect_one(row: dict) -> dict | None:
        state = {"question": question, "response": row["response"]}
        async with semaphore:
            try:
                result = await _ask_with_retries(client, state, questions)
            except Exception as e:
                logger.warning(
                    f"SystemOne detail detection failed for response "
                    f"{row['response_id']}: {e}"
                )
                return None

        probability = result.nouls["evidence_rich"].noul
        return {
            "response_id": row["response_id"],
            "evidence_rich": "YES" if probability >= threshold else "NO",
            "evidence_probability": probability,
        }

    rows = responses_df.to_dict(orient="records")
    results = await asyncio.gather(*[detect_one(row) for row in rows])

    return _merge_results(responses_df, rows, results)


def _merge_results(
    responses_df: pd.DataFrame,
    rows: list[dict],
    results: list[dict | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge per-response results back onto the input, splitting out failures."""
    processed = [result for result in results if result is not None]
    failed_ids = [
        row["response_id"]
        for row, result in zip(rows, results)
        if result is None
    ]
    unprocessable_df = responses_df[
        responses_df["response_id"].isin(failed_ids)
    ].reset_index(drop=True)

    if not processed:
        return pd.DataFrame(), unprocessable_df

    processed_df = responses_df.merge(
        pd.DataFrame(processed), how="inner", on="response_id"
    )
    return processed_df, unprocessable_df


async def find_themes_hybrid(
    responses_df: pd.DataFrame,
    llm: LLM,
    systemone_client: SystemOne,
    question: str,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    verbose: bool = True,
    concurrency: int = 10,
    threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
    mapping_question_type: str = "noul",
) -> dict[str, str | pd.DataFrame]:
    """Run the theme analysis pipeline with SystemOne classification stages.

    The generative stages (theme generation, condensation, refinement) run on
    the LLM exactly as in :func:`themefinder.find_themes`; the classification
    stages (theme mapping, detail detection) run on SystemOne.

    Args:
        responses_df: DataFrame containing survey responses.
        llm: LLM instance for the generative stages.
        systemone_client: SystemOne client for the classification stages.
        question: The survey question.
        system_prompt: System prompt guiding the LLM's behaviour.
        verbose: Whether to show information messages during processing.
        concurrency: Number of concurrent API calls to make.
        threshold: Probability threshold for SystemOne assignments.
        mapping_question_type: "noul" or "choice" mapping strategy (see
            :func:`theme_mapping_systemone`).

    Returns:
        Dictionary with the same shape as :func:`themefinder.find_themes`.
    """
    logger.setLevel(logging.INFO if verbose else logging.CRITICAL)

    theme_df, _ = await theme_generation(
        responses_df,
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    condensed_theme_df, _ = await theme_condensation(
        theme_df,
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    refined_theme_df, _ = await theme_refinement(
        condensed_theme_df,
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )

    mapping_df, mapping_unprocessables = await theme_mapping_systemone(
        responses_df[["response_id", "response"]],
        systemone_client,
        question=question,
        refined_themes_df=refined_theme_df,
        threshold=threshold,
        concurrency=concurrency,
        question_type=mapping_question_type,
    )
    detailed_df, _ = await detail_detection_systemone(
        responses_df[["response_id", "response"]],
        systemone_client,
        question=question,
        threshold=threshold,
        concurrency=concurrency,
    )

    logger.info("Finished finding themes (hybrid SystemOne pipeline)")
    return {
        "question": question,
        "themes": refined_theme_df,
        "mapping": mapping_df,
        "detailed_responses": detailed_df,
        "unprocessables": mapping_unprocessables,
    }
