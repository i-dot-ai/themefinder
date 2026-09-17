"""Tests for the SystemOne (TypeSafe jev) classification stage."""

from dataclasses import dataclass, field

import pandas as pd
import pytest

from themefinder import systemone
from themefinder.systemone import (
    EVIDENCE_KEY,
    FALLBACK_NO_REASON,
    FALLBACK_OTHER,
    GIVES_REASON_KEY,
    THEME_QUESTION_PREFIX,
    SystemOne,
    classify_responses_systemone,
)


@dataclass
class FakeNoulAnswer:
    noul: float


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 10


@dataclass
class FakeResponse:
    nouls: dict[str, FakeNoulAnswer]
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeTransport:
    """Answers noul questions from a canned {question_key: probability} map.

    Question keys are prefixed per response, e.g. "r1_theme_A" or
    "r2_evidence_rich"; unlisted keys answer 0.0.
    """

    def __init__(self, probabilities_by_key: dict[str, float]):
        self.probabilities_by_key = probabilities_by_key
        self.calls = []

    async def system_one(self, state, questions):
        self.calls.append((state, questions))
        return FakeResponse(
            nouls={
                key: FakeNoulAnswer(noul=self.probabilities_by_key.get(key, 0.0))
                for key in questions
            }
        )


class FailingTransport:
    async def system_one(self, state, questions):
        raise RuntimeError("SystemOne unavailable")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(systemone, "RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(systemone, "RETRY_MIN_WAIT_SECONDS", 0)
    monkeypatch.setattr(systemone, "RETRY_MAX_WAIT_SECONDS", 0)


@pytest.fixture
def themes_df():
    return pd.DataFrame(
        {
            "topic_id": ["A", "B"],
            "topic": [
                "Ban support: Supports a complete ban.",
                "Economic concerns: Worries about sports funding.",
            ],
        }
    )


@pytest.fixture
def responses_df():
    return pd.DataFrame(
        {
            "response_id": [1, 2],
            "response": ["ban them all", "think of the funding"],
        }
    )


async def test_classifies_both_stages_in_one_batched_call(themes_df, responses_df):
    transport = FakeTransport(
        {
            f"r1_{THEME_QUESTION_PREFIX}A": 0.9,
            f"r1_{EVIDENCE_KEY}": 0.8,
            f"r2_{THEME_QUESTION_PREFIX}A": 0.6,
            f"r2_{THEME_QUESTION_PREFIX}B": 0.7,
            f"r2_{EVIDENCE_KEY}": 0.01,
        }
    )
    client = SystemOne(transport=transport)

    result, unprocessable = await classify_responses_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    assert unprocessable.empty
    # One request covered every response and both stages
    assert len(transport.calls) == 1
    state, questions = transport.calls[0]
    assert [r["response_id"] for r in state["responses"]] == [1, 2]
    assert set(questions) == {
        f"r{rid}_{suffix}"
        for rid in (1, 2)
        for suffix in (
            f"{THEME_QUESTION_PREFIX}A",
            f"{THEME_QUESTION_PREFIX}B",
            GIVES_REASON_KEY,
            EVIDENCE_KEY,
        )
    }
    labels = dict(zip(result["response_id"], result["labels"]))
    assert labels == {1: ["A"], 2: ["A", "B"]}
    classifications = dict(zip(result["response_id"], result["evidence_rich"]))
    assert classifications == {1: "YES", 2: "NO"}
    assert result["theme_probabilities"].iloc[0] == {"A": 0.9, "B": 0.0}
    assert result["evidence_probability"].tolist() == [0.8, 0.01]


async def test_each_question_names_its_response_and_topic(themes_df, responses_df):
    transport = FakeTransport({})
    client = SystemOne(transport=transport)

    await classify_responses_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    state, questions = transport.calls[0]
    # Questions are compact JSON pointers; the definitions live in the state
    theme_question = questions[f"r1_{THEME_QUESTION_PREFIX}A"].instructions
    assert theme_question["response_id"] == 1
    assert theme_question["topic_id"] == "A"
    assert questions[f"r2_{THEME_QUESTION_PREFIX}A"].instructions["response_id"] == 2
    assert state["topics"]["A"] == "Ban support: Supports a complete ban."
    assert "evidence_rich_if" in state["evidence_rich_definition"]


async def test_batch_size_splits_responses_across_requests(themes_df, responses_df):
    transport = FakeTransport({})
    client = SystemOne(transport=transport)

    await classify_responses_systemone(
        responses_df,
        client,
        question="Q?",
        refined_themes_df=themes_df,
        batch_size=1,
    )

    assert len(transport.calls) == 2
    for (state, _), expected_id in zip(transport.calls, [1, 2]):
        assert [r["response_id"] for r in state["responses"]] == [expected_id]


async def test_falls_back_to_other_when_no_theme_matches_but_reason_given(themes_df):
    responses = pd.DataFrame({"response_id": [1], "response": "unrelated opinion"})
    transport = FakeTransport({f"r1_{GIVES_REASON_KEY}": 0.9})
    client = SystemOne(transport=transport)

    result, _ = await classify_responses_systemone(
        responses, client, question="Q?", refined_themes_df=themes_df
    )

    assert result["labels"].iloc[0] == [FALLBACK_OTHER]


async def test_falls_back_to_no_reason_given_for_non_answers(themes_df):
    responses = pd.DataFrame({"response_id": [1], "response": "n/a"})
    transport = FakeTransport({f"r1_{GIVES_REASON_KEY}": 0.1})
    client = SystemOne(transport=transport)

    result, _ = await classify_responses_systemone(
        responses, client, question="Q?", refined_themes_df=themes_df
    )

    assert result["labels"].iloc[0] == [FALLBACK_NO_REASON]


async def test_detail_threshold_is_separate_from_mapping_threshold(
    themes_df, responses_df
):
    # Evidence probabilities cluster low; 0.20 is rich at the 0.16 default
    # even though it is far below the 0.5 mapping threshold.
    transport = FakeTransport(
        {
            f"r1_{THEME_QUESTION_PREFIX}A": 0.9,
            f"r1_{EVIDENCE_KEY}": 0.20,
            f"r2_{THEME_QUESTION_PREFIX}B": 0.9,
            f"r2_{EVIDENCE_KEY}": 0.10,
        }
    )
    client = SystemOne(transport=transport)

    result, _ = await classify_responses_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    classifications = dict(zip(result["response_id"], result["evidence_rich"]))
    assert classifications == {1: "YES", 2: "NO"}


async def test_failed_batches_are_returned_as_unprocessable(themes_df, responses_df):
    client = SystemOne(transport=FailingTransport())

    result, unprocessable = await classify_responses_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    assert result.empty
    assert list(unprocessable["response_id"]) == [1, 2]


async def test_rate_limit_sets_shared_cooldown_and_recovers(
    monkeypatch, themes_df, responses_df
):
    import httpx
    from typesafe_sdk import TypeSafeRateLimitError

    monkeypatch.setattr(systemone, "RATE_LIMIT_COOLDOWN_SECONDS", 0.01)

    class RateLimitedOnceTransport(FakeTransport):
        def __init__(self):
            super().__init__({})
            self.failures_remaining = 1

        async def system_one(self, state, questions):
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise TypeSafeRateLimitError(429, None, httpx.Headers())
            return await super().system_one(state, questions)

    transport = RateLimitedOnceTransport()
    client = SystemOne(transport=transport)

    result, unprocessable = await classify_responses_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    assert unprocessable.empty
    assert len(result) == len(responses_df)
    assert client.rate_limit_hits == 1
    assert client._cooldown_until > 0
    assert 0 < client.rate_limit_pause_seconds < 0.05


async def test_client_accumulates_token_usage(themes_df, responses_df):
    transport = FakeTransport({})
    client = SystemOne(transport=transport)

    await classify_responses_systemone(
        responses_df,
        client,
        question="Q?",
        refined_themes_df=themes_df,
        batch_size=1,
    )

    assert client.usage.requests == 2
    assert client.usage.input_tokens == 200
    assert client.usage.output_tokens == 20


async def test_uses_label_and_description_when_no_combined_topic_column():
    themes = pd.DataFrame(
        {
            "topic_id": ["A"],
            "topic_label": ["ban support"],
            "topic_description": ["Supports a ban."],
        }
    )
    responses = pd.DataFrame({"response_id": [1], "response": "ban them all"})
    transport = FakeTransport({f"r1_{THEME_QUESTION_PREFIX}A": 0.9})
    client = SystemOne(transport=transport)

    result, _ = await classify_responses_systemone(
        responses, client, question="Q?", refined_themes_df=themes
    )

    assert result["labels"].iloc[0] == ["A"]
    state, _ = transport.calls[0]
    assert state["topics"]["A"] == "ban support: Supports a ban."
