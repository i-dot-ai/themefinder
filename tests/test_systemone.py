"""Tests for the SystemOne (TypeSafe jev) pipeline stages."""

from dataclasses import dataclass, field

import pandas as pd
import pytest

from themefinder import systemone
from themefinder.systemone import (
    FALLBACK_NO_REASON,
    FALLBACK_OTHER,
    GIVES_REASON_KEY,
    MAPPING_CHOICE_KEY,
    THEME_QUESTION_PREFIX,
    SystemOne,
    classify_responses_systemone,
    detail_detection_systemone,
    theme_mapping_systemone,
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
    """Returns canned noul probabilities keyed on the response text."""

    def __init__(self, probabilities_by_response: dict[str, dict[str, float]]):
        self.probabilities_by_response = probabilities_by_response
        self.calls = []

    async def system_one(self, state, questions):
        self.calls.append((state, questions))
        probabilities = self.probabilities_by_response[state["response"]]
        return FakeResponse(
            nouls={
                key: FakeNoulAnswer(noul=probabilities.get(key, 0.0))
                for key in questions
            }
        )


class FailingTransport:
    async def system_one(self, state, questions):
        raise RuntimeError("SystemOne unavailable")


@dataclass
class FakeChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.9


@dataclass
class FakeChoiceResponse:
    choices: dict[str, FakeChoiceAnswer]
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeChoiceTransport:
    """Returns a canned choice distribution keyed on the response text."""

    def __init__(self, distributions_by_response: dict[str, dict[str, float]]):
        self.distributions_by_response = distributions_by_response
        self.calls = []

    async def system_one(self, state, questions):
        self.calls.append((state, questions))
        probabilities = self.distributions_by_response[state["response"]]
        top_choice = max(probabilities, key=probabilities.get)
        return FakeChoiceResponse(
            choices={
                MAPPING_CHOICE_KEY: FakeChoiceAnswer(
                    choice=top_choice, probabilities=probabilities
                )
            }
        )


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


async def test_maps_responses_to_themes_above_threshold(themes_df, responses_df):
    transport = FakeTransport(
        {
            "ban them all": {f"{THEME_QUESTION_PREFIX}A": 0.9, f"{THEME_QUESTION_PREFIX}B": 0.2},
            "think of the funding": {
                f"{THEME_QUESTION_PREFIX}A": 0.6,
                f"{THEME_QUESTION_PREFIX}B": 0.8,
            },
        }
    )
    client = SystemOne(transport=transport)

    result, unprocessable = await theme_mapping_systemone(
        responses_df, client, question="Should ads be banned?", refined_themes_df=themes_df
    )

    assert unprocessable.empty
    labels = dict(zip(result["response_id"], result["labels"]))
    assert labels[1] == ["A"]
    assert labels[2] == ["A", "B"]
    assert result["theme_probabilities"].iloc[0] == {"A": 0.9, "B": 0.2}


async def test_batches_all_theme_questions_into_one_call_per_response(
    themes_df, responses_df
):
    transport = FakeTransport(
        {
            "ban them all": {f"{THEME_QUESTION_PREFIX}A": 0.9},
            "think of the funding": {f"{THEME_QUESTION_PREFIX}B": 0.9},
        }
    )
    client = SystemOne(transport=transport)

    await theme_mapping_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    assert len(transport.calls) == len(responses_df)
    _, questions = transport.calls[0]
    assert set(questions) == {
        f"{THEME_QUESTION_PREFIX}A",
        f"{THEME_QUESTION_PREFIX}B",
        GIVES_REASON_KEY,
    }


async def test_falls_back_to_other_when_no_theme_matches_but_reason_given(
    themes_df,
):
    responses = pd.DataFrame({"response_id": [1], "response": "unrelated opinion"})
    transport = FakeTransport(
        {"unrelated opinion": {GIVES_REASON_KEY: 0.9}},
    )
    client = SystemOne(transport=transport)

    result, _ = await theme_mapping_systemone(
        responses, client, question="Q?", refined_themes_df=themes_df
    )

    assert result["labels"].iloc[0] == [FALLBACK_OTHER]


async def test_falls_back_to_no_reason_given_for_non_answers(themes_df):
    responses = pd.DataFrame({"response_id": [1], "response": "n/a"})
    transport = FakeTransport({"n/a": {GIVES_REASON_KEY: 0.1}})
    client = SystemOne(transport=transport)

    result, _ = await theme_mapping_systemone(
        responses, client, question="Q?", refined_themes_df=themes_df
    )

    assert result["labels"].iloc[0] == [FALLBACK_NO_REASON]


async def test_failed_responses_are_returned_as_unprocessable(themes_df, responses_df):
    client = SystemOne(transport=FailingTransport())

    result, unprocessable = await theme_mapping_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    assert result.empty
    assert list(unprocessable["response_id"]) == [1, 2]


async def test_choice_mapping_assigns_themes_from_distribution(
    themes_df, responses_df
):
    transport = FakeChoiceTransport(
        {
            "ban them all": {"A": 0.7, "B": 0.1, "Other": 0.1, "No Reason Given": 0.1},
            "think of the funding": {
                "A": 0.35,
                "B": 0.45,
                "Other": 0.1,
                "No Reason Given": 0.1,
            },
        }
    )
    client = SystemOne(transport=transport)

    result, unprocessable = await theme_mapping_systemone(
        responses_df,
        client,
        question="Q?",
        refined_themes_df=themes_df,
        threshold=0.25,
        question_type="choice",
    )

    assert unprocessable.empty
    labels = dict(zip(result["response_id"], result["labels"]))
    assert labels[1] == ["A"]
    assert labels[2] == ["A", "B"]
    _, questions = transport.calls[0]
    criteria = questions[MAPPING_CHOICE_KEY].criteria
    assert set(criteria) == {"A", "B", "Other", "No Reason Given"}


async def test_choice_mapping_falls_back_to_top_choice(themes_df):
    responses = pd.DataFrame({"response_id": [1], "response": "n/a"})
    transport = FakeChoiceTransport(
        {"n/a": {"A": 0.1, "B": 0.1, "Other": 0.2, "No Reason Given": 0.6}}
    )
    client = SystemOne(transport=transport)

    result, _ = await theme_mapping_systemone(
        responses,
        client,
        question="Q?",
        refined_themes_df=themes_df,
        threshold=0.25,
        question_type="choice",
    )

    assert result["labels"].iloc[0] == [FALLBACK_NO_REASON]


async def test_rejects_unknown_mapping_question_type(themes_df, responses_df):
    client = SystemOne(transport=FailingTransport())

    with pytest.raises(ValueError, match="question_type"):
        await theme_mapping_systemone(
            responses_df,
            client,
            question="Q?",
            refined_themes_df=themes_df,
            question_type="score",
        )


async def test_detail_detection_classifies_by_threshold(responses_df):
    transport = FakeTransport(
        {
            "ban them all": {"evidence_rich": 0.8},
            "think of the funding": {"evidence_rich": 0.3},
        }
    )
    client = SystemOne(transport=transport)

    result, unprocessable = await detail_detection_systemone(
        responses_df, client, question="Q?"
    )

    assert unprocessable.empty
    classifications = dict(zip(result["response_id"], result["evidence_rich"]))
    assert classifications == {1: "YES", 2: "NO"}
    assert result["evidence_probability"].tolist() == [0.8, 0.3]


async def test_combined_classification_uses_one_call_per_response(
    themes_df, responses_df
):
    transport = FakeTransport(
        {
            "ban them all": {
                f"{THEME_QUESTION_PREFIX}A": 0.9,
                "evidence_rich": 0.8,
            },
            "think of the funding": {
                f"{THEME_QUESTION_PREFIX}B": 0.7,
                "evidence_rich": 0.2,
            },
        }
    )
    client = SystemOne(transport=transport)

    result, unprocessable = await classify_responses_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
    )

    assert unprocessable.empty
    # Both stages answered from a single call per response
    assert len(transport.calls) == len(responses_df)
    _, questions = transport.calls[0]
    assert set(questions) == {
        f"{THEME_QUESTION_PREFIX}A",
        f"{THEME_QUESTION_PREFIX}B",
        GIVES_REASON_KEY,
        "evidence_rich",
    }
    labels = dict(zip(result["response_id"], result["labels"]))
    assert labels == {1: ["A"], 2: ["B"]}
    classifications = dict(zip(result["response_id"], result["evidence_rich"]))
    assert classifications == {1: "YES", 2: "NO"}


async def test_client_accumulates_token_usage(themes_df, responses_df):
    transport = FakeTransport(
        {
            "ban them all": {f"{THEME_QUESTION_PREFIX}A": 0.9},
            "think of the funding": {f"{THEME_QUESTION_PREFIX}B": 0.9},
        }
    )
    client = SystemOne(transport=transport)

    await theme_mapping_systemone(
        responses_df, client, question="Q?", refined_themes_df=themes_df
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
    transport = FakeTransport(
        {"ban them all": {f"{THEME_QUESTION_PREFIX}A": 0.9}}
    )
    client = SystemOne(transport=transport)

    result, _ = await theme_mapping_systemone(
        responses, client, question="Q?", refined_themes_df=themes
    )

    assert result["labels"].iloc[0] == ["A"]
    _, questions = transport.calls[0]
    assert "ban support: Supports a ban." in questions[
        f"{THEME_QUESTION_PREFIX}A"
    ].instructions
