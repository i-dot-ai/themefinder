from unittest.mock import Mock, patch

import openai
import pytest

from themefinder import (
    rule_1_total_theme_number_less_than_70,
    rule_2_themes_must_have_a_non_negligible_number_of_responses,
    rule_3_semantic_similarity_must_be_less_than_90pc,
    rule_4_themes_should_not_overlap,
)
from themefinder.models import ThemeNode


@pytest.mark.parametrize("n, expected", [(69, 69), (70, 70), (71, None)])
def test_rule_1_total_theme_number_less_than_70(n, expected):
    result = rule_1_total_theme_number_less_than_70([{} for _ in range(n)])
    assert result is expected


def test_rule_2_themes_must_have_a_non_negligible_number_of_responses_large():
    """this rule should detect that:
    theme 'a' appears more than 0.001% of the time -> is not flagged
    theme 'b' appears less than 0.001% of the time -> is flagged
    """
    mapping = (
        [{"theme_keys": []} for _ in range(10_000)]
        + [{"theme_keys": ["a"]} for _ in range(11)]
        + [{"theme_keys": ["b"]}]
    )
    result = rule_2_themes_must_have_a_non_negligible_number_of_responses(mapping)
    assert result == [("b", 1, 10_012)]


def test_rule_2_themes_must_have_a_non_negligible_number_of_responses_small():
    """this rule should detect that:
    although the number of responses is so low that no theme can appear less than 0.001% of the time,
    theme 'a' appears more than 5 times -> is not flagged
    theme 'b' appears less than 5 times -> is flagged
    """
    mapping = [
        {"theme_keys": ["a"]},
        {"theme_keys": ["a"]},
        {"theme_keys": ["a"]},
        {"theme_keys": ["a", "b"]},
        {"theme_keys": ["a", "b"]},
        {"theme_keys": ["a", "b"]},
    ]
    result = rule_2_themes_must_have_a_non_negligible_number_of_responses(mapping)
    assert result == [("b", 3, 6)]


@patch("openai.resources.embeddings.Embeddings.create")
def test_rule_3_semantic_similarity_must_be_less_than_90pc(mock_create):
    clustered_themes = [
        ThemeNode(
            topic_label="A",
            topic_description="courgette",
            topic_id="1",
            source_topic_count=1,
        ),
        ThemeNode(
            topic_label="B",
            topic_description="zucchini",
            topic_id="2",
            source_topic_count=1,
        ),
        ThemeNode(
            topic_label="C",
            topic_description="louis XIV king of france",
            topic_id="3",
            source_topic_count=1,
        ),
    ]

    mock_embedding_for_a_and_b = Mock()
    mock_embedding_for_a_and_b.embedding = [1.0, 0.0, 0.0]
    mock_embedding_for_c = Mock()
    mock_embedding_for_c.embedding = [0.0, 0.0, 1.0]

    mock_response = Mock()
    mock_response.data = [
        mock_embedding_for_a_and_b,
        mock_embedding_for_a_and_b,
        mock_embedding_for_c,
    ]

    mock_create.return_value = mock_response

    client = openai.OpenAI(api_key="dummy")

    result = rule_3_semantic_similarity_must_be_less_than_90pc(clustered_themes, client)
    assert result == [("A", "B", 1.0)]


def test_rule_4_themes_should_not_overlap():
    """this rule should detect that themes a & b have more than 70% in common
    but that the other paris, b & c and a & c do not"""

    mapping = [
        {"themefinder_id": 1, "theme_keys": ["a", "b"]},
        {"themefinder_id": 2, "theme_keys": ["a", "b", "c"]},
        {"themefinder_id": 3, "theme_keys": ["a", "b", "c"]},
        {"themefinder_id": 4, "theme_keys": ["a", "b", "c"]},
        {"themefinder_id": 5, "theme_keys": ["a", "c"]},
        {"themefinder_id": 6, "theme_keys": ["c"]},
    ]
    result = rule_4_themes_should_not_overlap(mapping)
    assert result == [("a", "b", 0.8)]
