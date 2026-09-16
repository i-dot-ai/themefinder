"""SystemOne (TypeSafe jev) implementation of themefinder's classification stages.

The jev SystemOne model answers typed yes/no ("noul") questions about a piece
of state, returning calibrated probabilities. It is not generative, so only
the classification-shaped stages of the themefinder pipeline transfer: theme
mapping ("does this response express this topic?") and detail detection ("is
this response evidence-rich?"). The generative stages (theme generation,
condensation, refinement) remain on the LLM; :func:`find_themes_hybrid`
combines both.

Requests are shaped for efficiency, following TypeSafe's guidance that the
state dominates each request and that questions in one request are evaluated
independently and in parallel:

- both stages' questions are asked in the same request, and
- a batch of responses shares each request: the state carries the question
  and a list of responses, and every question names the response_id it is
  about.

Benchmarked against the local gambling_XS ground truth
(evals/compare_systemone.py), this shape matched or beat the per-response and
choice-question alternatives on F1 while making a twentieth of the requests,
so the alternatives were removed.

Thresholds: theme probabilities are well spread, so the default assignment
threshold is 0.5. Evidence-rich probabilities cluster near zero (the rubric
is strict) while ranking responses accurately, so the default detail
threshold is 0.05 — tuned on one 100-response question part; sanity-check it
per consultation rather than trusting it universally.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
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
DEFAULT_DETAIL_THRESHOLD = 0.05

# SystemOne calls are small and fast; the model is built for high-throughput
# parallel questioning, so a much higher concurrency than an LLM's is safe.
DEFAULT_CONCURRENCY = 50

# Responses per request. Each response in the batch carries its own question
# set, so requests grow linearly with this; 20 mirrors the LLM stages.
DEFAULT_BATCH_SIZE = 20

# Retry policy for SystemOne calls, mirroring the LLM batch processor.
RETRY_ATTEMPTS = 6
RETRY_MIN_WAIT_SECONDS = 1
RETRY_MAX_WAIT_SECONDS = 20

THEME_QUESTION_PREFIX = "theme_"
GIVES_REASON_KEY = "gives_reason"
EVIDENCE_KEY = "evidence_rich"
FALLBACK_OTHER = "Other"
FALLBACK_NO_REASON = "No Reason Given"

# Every question opens with this preamble so its judgement targets exactly
# one response within the shared state.
RESPONSE_PREAMBLE = (
    "The state contains a consultation question and a list of responses, each "
    "with a response_id. Consider only the response whose response_id is "
    "{response_id}, ignoring all other responses. "
)

THEME_MAPPING_BODY = (
    "Does the response express the following topic? The response does not need to "
    "use the same wording as the topic; it is a match if it expresses a similar "
    "sentiment or point of view. Topic: {topic}"
)

GIVES_REASON_BODY = (
    "Does the response give any substantive opinion, reason or argument in answer "
    "to the question, as opposed to being empty, off-topic or a refusal to answer?"
)

EVIDENCE_RICH_BODY = (
    "Is the response evidence-rich? A response is evidence-rich only if it clearly "
    "answers the question with insights beyond generic opinion (nuanced reasoning, "
    "contextual explanation or argumentation that could inform decision-making) "
    "AND it contains substantive evidence: specific verifiable facts or data "
    "(statistics, dates, named reports or studies), concrete illustrative examples "
    "that support a broader claim, or detailed personal or professional experiences "
    "with contextual information such as roles, locations or timelines."
)

EVIDENCE_RICH_TRUE_CRITERIA = (
    "The response would provide useful input to someone drafting policy, beyond "
    "what is already commonly known or expected."
)
EVIDENCE_RICH_FALSE_CRITERIA = (
    "The response uses vague language with no supporting detail, restates commonly "
    "known points, or shares anecdotes without sufficient context or a clear "
    "takeaway."
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
        return dict(zip(refined_themes_df["topic_id"], refined_themes_df["topic"]))
    return {
        row["topic_id"]: f"{row['topic_label']}: {row['topic_description']}"
        for _, row in refined_themes_df.iterrows()
    }


def _noul(
    instructions: str,
    true_criteria: str | None = None,
    false_criteria: str | None = None,
) -> Any:
    """Build a noul (yes/no) SystemOne question.

    The optional criteria describe the yes and no outcomes, in the
    ``NoulCriteria`` shape the API expects ({"true": ..., "false": ...}).
    """
    try:
        from typesafe_sdk import Noul
    except ImportError as e:
        raise ImportError(
            "The 'typesafe-sdk' package is required for SystemOne stages. "
            "Install it with the 'systemone' extra: pip install 'themefinder[systemone]'"
        ) from e
    if true_criteria or false_criteria:
        return Noul(
            instructions=instructions,
            criteria={"true": true_criteria, "false": false_criteria},
        )
    return Noul(instructions=instructions)


def _is_retryable(exception: BaseException) -> bool:
    """Deterministic client errors (4xx validation, auth) will never succeed."""
    try:
        from typesafe_sdk import (
            TypeSafeAuthenticationError,
            TypeSafeBadRequestError,
            TypeSafeNotFoundError,
            TypeSafePermissionDeniedError,
            TypeSafeUnprocessableEntityError,
        )
    except ImportError:
        return True
    return not isinstance(
        exception,
        (
            TypeSafeAuthenticationError,
            TypeSafeBadRequestError,
            TypeSafeNotFoundError,
            TypeSafePermissionDeniedError,
            TypeSafeUnprocessableEntityError,
        ),
    )


async def _ask_with_retries(
    client: SystemOne, state: Any, questions: dict[str, Any]
) -> Any:
    """Ask SystemOne with the same retry policy as the LLM batch processor."""
    retrying = AsyncRetrying(
        wait=wait_random_exponential(
            min=RETRY_MIN_WAIT_SECONDS, max=RETRY_MAX_WAIT_SECONDS
        ),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    return await retrying(client.ask, state=state, questions=questions)


def _response_questions(response_id: Any, theme_texts: dict[str, str]) -> dict[str, Any]:
    """Build one response's question set: per-theme nouls, fallback, evidence."""
    prefix = f"r{response_id}_"
    preamble = RESPONSE_PREAMBLE.format(response_id=response_id)
    questions = {
        f"{prefix}{THEME_QUESTION_PREFIX}{topic_id}": _noul(
            preamble + THEME_MAPPING_BODY.format(topic=topic_text)
        )
        for topic_id, topic_text in theme_texts.items()
    }
    questions[f"{prefix}{GIVES_REASON_KEY}"] = _noul(preamble + GIVES_REASON_BODY)
    questions[f"{prefix}{EVIDENCE_KEY}"] = _noul(
        preamble + EVIDENCE_RICH_BODY,
        true_criteria=EVIDENCE_RICH_TRUE_CRITERIA,
        false_criteria=EVIDENCE_RICH_FALSE_CRITERIA,
    )
    return questions


def _extract_response_output(
    result: Any,
    response_id: Any,
    theme_texts: dict[str, str],
    threshold: float,
    detail_threshold: float,
) -> dict:
    """Turn one response's SystemOne answers into an output row."""
    prefix = f"r{response_id}_"
    probabilities = {
        topic_id: result.nouls[f"{prefix}{THEME_QUESTION_PREFIX}{topic_id}"].noul
        for topic_id in theme_texts
    }
    labels = [
        topic_id
        for topic_id, probability in probabilities.items()
        if probability >= threshold
    ]
    if not labels:
        gives_reason = result.nouls[f"{prefix}{GIVES_REASON_KEY}"].noul
        labels = [FALLBACK_OTHER if gives_reason >= threshold else FALLBACK_NO_REASON]

    evidence_probability = result.nouls[f"{prefix}{EVIDENCE_KEY}"].noul
    return {
        "response_id": response_id,
        "labels": labels,
        "theme_probabilities": probabilities,
        "evidence_rich": "YES" if evidence_probability >= detail_threshold else "NO",
        "evidence_probability": evidence_probability,
    }


def _merge_results(
    responses_df: pd.DataFrame,
    rows: list[dict],
    results: list[dict | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge per-response results back onto the input, splitting out failures."""
    processed = [result for result in results if result is not None]
    failed_ids = [
        row["response_id"] for row, result in zip(rows, results) if result is None
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


async def classify_responses_systemone(
    responses_df: pd.DataFrame,
    client: SystemOne,
    question: str,
    refined_themes_df: pd.DataFrame,
    threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
    detail_threshold: float = DEFAULT_DETAIL_THRESHOLD,
    concurrency: int = DEFAULT_CONCURRENCY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map responses to themes and detect evidence-rich responses via SystemOne.

    Responses are processed in batches: each request's state carries the
    question and a list of responses, and every response gets its own set of
    per-theme yes/no questions plus a fallback question and an evidence-rich
    question, all answered in that one request. Themes whose probability meets
    the threshold are assigned; when none does, the response is labelled
    "Other" (or "No Reason Given" when it does not appear to answer the
    question at all).

    Args:
        responses_df: DataFrame with 'response_id' and 'response' columns.
        client: SystemOne client wrapper.
        question: The survey question.
        refined_themes_df: DataFrame of refined themes with 'topic_id' and
            'topic' (or 'topic_label' + 'topic_description') columns.
        threshold: Minimum probability for a theme to be assigned.
        detail_threshold: Minimum probability to classify a response as
            evidence-rich. Evidence probabilities cluster low, hence the
            separate, lower default.
        concurrency: Maximum number of simultaneous SystemOne calls.
        batch_size: Number of responses sharing each request.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows).
        Results carry 'labels' and 'evidence_rich' (matching the LLM stages'
        outputs) plus 'theme_probabilities' and 'evidence_probability'.
    """
    logger.info(
        f"Running SystemOne classification on {len(responses_df)} responses "
        f"using {len(refined_themes_df)} themes (batch size {batch_size})"
    )
    theme_texts = _theme_texts(refined_themes_df)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_chunk(
        chunk: pd.DataFrame,
    ) -> tuple[list[dict], list[dict | None]]:
        rows = chunk.to_dict(orient="records")
        state = {
            "question": question,
            "responses": [
                {"response_id": row["response_id"], "response": row["response"]}
                for row in rows
            ],
        }
        questions: dict[str, Any] = {}
        for row in rows:
            questions.update(_response_questions(row["response_id"], theme_texts))

        async with semaphore:
            try:
                result = await _ask_with_retries(client, state, questions)
            except Exception as e:
                logger.warning(
                    f"SystemOne classification failed for responses "
                    f"{[row['response_id'] for row in rows]}: {e}"
                )
                return rows, [None] * len(rows)

        outputs = [
            _extract_response_output(
                result, row["response_id"], theme_texts, threshold, detail_threshold
            )
            for row in rows
        ]
        return rows, outputs

    chunks = [
        responses_df.iloc[i : i + batch_size]
        for i in range(0, len(responses_df), batch_size)
    ]
    chunk_results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

    all_rows = [row for rows, _ in chunk_results for row in rows]
    all_outputs = [output for _, outputs in chunk_results for output in outputs]
    return _merge_results(responses_df, all_rows, all_outputs)


async def find_themes_hybrid(
    responses_df: pd.DataFrame,
    llm: LLM,
    systemone_client: SystemOne,
    question: str,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    verbose: bool = True,
    concurrency: int = 10,
    systemone_concurrency: int = DEFAULT_CONCURRENCY,
    threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
    detail_threshold: float = DEFAULT_DETAIL_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, str | pd.DataFrame]:
    """Run the theme analysis pipeline with SystemOne classification stages.

    The generative stages (theme generation, condensation, refinement) run on
    the LLM exactly as in :func:`themefinder.find_themes`; the classification
    stages (theme mapping, detail detection) run on SystemOne in one batched
    request per group of responses.

    Args:
        responses_df: DataFrame containing survey responses.
        llm: LLM instance for the generative stages.
        systemone_client: SystemOne client for the classification stages.
        question: The survey question.
        system_prompt: System prompt guiding the LLM's behaviour.
        verbose: Whether to show information messages during processing.
        concurrency: Number of concurrent LLM calls to make.
        systemone_concurrency: Number of concurrent SystemOne calls to make.
        threshold: Probability threshold for theme assignment.
        detail_threshold: Probability threshold for evidence-rich classification.
        batch_size: Number of responses sharing each SystemOne request.

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

    classified_df, unprocessables = await classify_responses_systemone(
        responses_df[["response_id", "response"]],
        systemone_client,
        question=question,
        refined_themes_df=refined_theme_df,
        threshold=threshold,
        detail_threshold=detail_threshold,
        concurrency=systemone_concurrency,
        batch_size=batch_size,
    )
    if classified_df.empty:
        mapping_df = detailed_df = classified_df
    else:
        mapping_df = classified_df[
            ["response_id", "response", "labels", "theme_probabilities"]
        ]
        detailed_df = classified_df[
            ["response_id", "response", "evidence_rich", "evidence_probability"]
        ]

    logger.info("Finished finding themes (hybrid SystemOne pipeline)")
    return {
        "question": question,
        "themes": refined_theme_df,
        "mapping": mapping_df,
        "detailed_responses": detailed_df,
        "unprocessables": unprocessables,
    }
