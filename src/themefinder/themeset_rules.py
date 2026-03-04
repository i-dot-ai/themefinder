import collections
from itertools import pairwise

import numpy as np
from openai import OpenAI

from themefinder.models import ThemeNode


def rule_1_total_theme_number_less_than_70(clustered_themes: list) -> int | None:
    """
    The number of child themes should be no more than 70
    Rationale: Users typically want less themes than this, so we do not want Consultation owners to have to much work to do to reduce the theme-set to meet their expectations.
    """
    if len(clustered_themes) <= 70:
        return len(clustered_themes)
    return None


def rule_2_themes_must_have_a_non_negligible_number_of_responses(
    mapping: list[dict],
) -> list[tuple[str, int, int]]:
    """
    A child theme must be assigned to at least 0.1% or 5, whichever is the greater, of the responses
    Rationale: We want to remove anomalous results from the consultation.
    """

    counter: collections.defaultdict[str, int] = collections.defaultdict(int)

    for line in mapping:
        for theme_key in line["theme_keys"]:
            counter[theme_key] += 1

    return [
        (theme_key, count, len(mapping))
        for theme_key, count in counter.items()
        if count / len(mapping) < 0.001 or count < 5
    ]


def rule_3_semantic_similarity_must_be_less_than_90pc(
    clustered_themes: list[ThemeNode], client: OpenAI
) -> list[tuple[str, str, float]]:
    """
    The semantic similarity between theme titles and descriptions must be less than 90%
    Rationale: We do not want multiple instances of very similar themes, i.e. "environment" and "environmental"
    """

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    response = client.embeddings.create(
        input=[x.topic_label + ": " + x.topic_description for x in clustered_themes],
        model="text-embedding-3-large",
    )

    labeled_embeddings = zip(clustered_themes, response.data)

    results = []
    for (label_1, embedding_1), (label_2, embedding_2) in pairwise(labeled_embeddings):
        similarity = cosine_similarity(embedding_1.embedding, embedding_2.embedding)
        if similarity > 0.9:
            results.append((label_1.topic_label, label_2.topic_label, similarity))

    return results


def rule_4_themes_should_not_overlap(
    mapping: list[dict],
) -> list[tuple[str, str, float]]:
    """
    The size of intersection of any two themes representative responses divined by the size of the union of its representative responses should be less than 70%
    Rationale: We do not want 2 themes, even if semantically distinct, to be mapped to the same response-set, in this case they can be merged.
    """
    counter = collections.defaultdict(set)

    for line in mapping:
        for theme_key in line["theme_keys"]:
            counter[theme_key].add(line["themefinder_id"])

    theme_keys = list(counter)

    results = []
    for i, theme_key_a in enumerate(theme_keys):
        for theme_key_b in theme_keys[i + 1 :]:
            response_a, response_b = counter[theme_key_a], counter[theme_key_b]
            intersection = len(set.intersection(response_a, response_b))
            union = len(set.union(response_a, response_b))
            if intersection / union > 0.7:
                results.append((theme_key_a, theme_key_b, intersection / union))
    return results
