"""Prompt templates for themefinder evaluations."""

from typing import Any

CONDENSATION_EVAL = """You are an expert in natural language processing and topic modelling. Your task is to evaluate how well a model condenses similar topics while maintaining information. You will be shown:

1. The ORIGINAL list of topics (before condensation)
2. The CONDENSED list of topics (after condensation)

Evaluate the condensed topics on two criteria, each scored from 0 to 5:

## Compression Quality

How well are similar topics merged while keeping distinct topics separate?

- 5: All similar topics properly combined, no distinct topics incorrectly merged
- 4: Most similar topics combined appropriately, minimal errors
- 3: Some appropriate combinations, but missed opportunities or minor errors
- 2: Several missed combinations or inappropriate mergers
- 1: Most similar topics not combined or distinct topics incorrectly merged
- 0: No meaningful compression or complete loss of topic distinctness

## Information Retention

How well is the key information from the original topics preserved?

- 5: All key information preserved in condensed topics
- 4: Most important information preserved, only minor details lost
- 3: Core concepts preserved but significant details lost
- 2: Important information missing from condensed topics
- 1: Most key information lost in condensation
- 0: Essential information completely lost

## Calibration Examples

**High quality condensation** (compression_quality: 5, information_retention: 4):
Original topics included "Lack of affordable options", "High cost of entry-level products", and "Price barriers for low-income groups" as separate topics. The condensed output merged these into a single topic "Affordability barriers: High costs and limited options create barriers for low-income groups seeking entry-level products." All three related concepts are captured in one well-worded topic. A minor detail about specific product types was lost, but the core message is preserved.

**Poor condensation** (compression_quality: 1, information_retention: 2):
Original topics included "Environmental impact of packaging", "Carbon emissions from delivery", and "Staff working conditions" as separate topics. The condensed output merged all three into "Sustainability concerns: Various environmental and social issues." This incorrectly merges the distinct environmental and social/labour concepts, and the vague description loses almost all specific information from the originals.

## Instructions

1. First, reason about how the condensation performed on each criterion.
2. Then provide your scores in strict JSON format.

Return only the following JSON, no other text:

{{
  "compression_quality": <score 0-5>,
  "compression_quality_reasoning": "1-2 sentence explanation",
  "information_retention": <score 0-5>,
  "information_retention_reasoning": "1-2 sentence explanation"
}}

ORIGINAL TOPICS:
{original_topics}

CONDENSED TOPICS:
{condensed_topics}
"""

GENERATION_EVAL = """You are an expert in natural language processing and topic modelling. Your task is to evaluate how well topics in LIST 1 are captured by topics in LIST 2.

For each topic in LIST 1, find the best matching topic in LIST 2 and assign one of three decisions:

- **STRONG**: The topics represent the same core concept, even if worded differently.
- **PARTIAL**: The topics are related but differ in scope, specificity, or emphasis.
- **NO**: No meaningful match exists in LIST 2.

For each topic, provide:
1. The label of the best matching topic in LIST 2 (or "none" if NO match)
2. Your decision (STRONG, PARTIAL, or NO)
3. A brief reasoning (1-2 sentences) explaining your decision

## Calibration Examples

These examples illustrate the expected standard for each decision:

**STRONG** — "Humiliating current system" ↔ "Inhumane current assessments"
Reasoning: Both topics describe the degrading nature of the existing assessment process. Different wording, same core concept.

**PARTIAL** — "Speeding up the process" ↔ "Dignified assessments"
Reasoning: Both relate to improving the assessment experience, but one focuses on speed while the other focuses on dignity. Related but different emphasis.

**NO** — "Better job matching" ↔ "Reduction in administrative burden"
Reasoning: Job matching and administrative burden are entirely different concerns with no conceptual overlap.

## Output Format

Return strict JSON in this format:
{{
    "evaluations": {{
        "topic_label_from_list_1": {{
            "matched_to": "topic_label_from_list_2 or none",
            "decision": "STRONG or PARTIAL or NO",
            "reasoning": "Brief explanation"
        }}
    }}
}}

LIST 1:
{topic_list_1}

LIST 2:
{topic_list_2}"""

REFINEMENT_EVAL = """You are an expert in natural language processing and topic modelling. Your task is to evaluate how well a model refines topic labels and descriptions. You will be shown:

1. ORIGINAL TOPICS: The topics before refinement.
2. NEW TOPICS: The topics after a refinement step that improves labels and descriptions.

Evaluate the NEW TOPICS based on the ORIGINAL TOPICS on four criteria, each scored from 0 to 5:

## Information Retention

How well do the NEW TOPICS preserve key information from the ORIGINAL TOPICS?

- 5: All important details retained, nothing meaningful lost
- 4: Most important information preserved, only minor details lost
- 3: Core concepts preserved but significant details lost
- 2: Important information missing from refined topics
- 1: Most key information lost during refinement
- 0: Essential information completely lost

## Response References

Do the NEW TOPICS avoid language that references survey responses directly? Topics should describe the content itself, not how respondents expressed it.

- 5: No topics use language that refers to responses (e.g. "respondents said", "the majority of responses")
- 4: At most one topic contains a minor response reference
- 3: A few topics reference responses but most are content-focused
- 2: Many topics refer to responses rather than describing content directly
- 1: Most topics are framed around what respondents said
- 0: Nearly all topics reference responses instead of describing content

## Distinctiveness

How well do the NEW TOPICS represent distinct concepts without overlap?

- 5: All topics are clearly distinct with no conceptual overlap
- 4: Nearly all distinct, one or two minor overlaps
- 3: Most distinct but some noticeable overlap between topics
- 2: Several topics overlap significantly
- 1: Most topics overlap or are poorly differentiated
- 0: Topics are largely redundant or indistinguishable

## Fluency

Are the NEW TOPICS well structured with clear labels and informative descriptions?

- 5: All topics have concise, descriptive labels followed by detailed descriptions that expand on (not repeat) the label
- 4: Most topics are well structured with clear labels and good descriptions
- 3: Topics are generally readable but some labels are vague or descriptions merely repeat the label
- 2: Several topics have poor labels or uninformative descriptions
- 1: Most topics are poorly worded or structured
- 0: Topics are unreadable or incomprehensible

## Calibration Examples

**High quality refinement** (information_retention: 5, response_references: 5, distinctiveness: 4, fluency: 5):
Original topic: "People saying they want better access to services". Refined to: "Service accessibility gaps: Limited availability of essential services in rural and underserved areas creates barriers to healthcare, education, and social support." The refined version removes the response reference ("people saying"), adds a specific label, and expands the description with concrete detail. One minor overlap exists with another topic about healthcare specifically.

**Poor refinement** (information_retention: 2, response_references: 1, distinctiveness: 3, fluency: 2):
Original topic: "Concerns about environmental regulations being too strict". Refined to: "Many respondents feel regulations are problematic: The majority of responses indicate that current rules are seen as excessive by participants." The refined version loses the environmental specificity, introduces multiple response references, and the description merely restates the label.

## Instructions

1. First, reason about how the refinement performed on each criterion.
2. Then provide your scores in strict JSON format.

Return only the following JSON, no other text:

{{
  "information_retention": <score 0-5>,
  "information_retention_reasoning": "1-2 sentence explanation",
  "response_references": <score 0-5>,
  "response_references_reasoning": "1-2 sentence explanation",
  "distinctiveness": <score 0-5>,
  "distinctiveness_reasoning": "1-2 sentence explanation",
  "fluency": <score 0-5>,
  "fluency_reasoning": "1-2 sentence explanation"
}}

ORIGINAL TOPICS:
{original_topics}

NEW TOPICS:
{new_topics}
"""

TITLE_SPECIFICITY_EVAL = """You are an expert in evaluating the quality of topic labels from a topic modelling system. Your task is to assess whether each theme title is specific enough to convey meaningful information, or whether it is too vague to be useful.

For each theme title, assign one of two decisions:

- **SPECIFIC**: The title conveys a clear, concrete concept that distinguishes it from other themes. A reader could understand the topic's focus from the title alone.
- **VAGUE**: The title is generic, ambiguous, or could apply to many different topics. It fails to communicate what makes this theme distinct.

## Calibration Examples

**VAGUE**: "Concerns about the process"
Reasoning: Too generic — almost any negative theme could be described as "concerns about the process."

**SPECIFIC**: "Delays in planning permission approvals"
Reasoning: Clearly identifies both the issue (delays) and the context (planning permission approvals).

**VAGUE**: "General support"
Reasoning: Does not specify what is being supported or why.

**SPECIFIC**: "Mandatory disability awareness training for assessors"
Reasoning: Identifies a concrete policy recommendation with clear scope.

## Output Format

Return strict JSON in this format:
{{
    "evaluations": {{
        "theme_title_1": {{
            "decision": "SPECIFIC or VAGUE",
            "reasoning": "Brief explanation"
        }}
    }}
}}

THEME TITLES:
{theme_titles}"""


# Typed prompt functions


def condensation_eval_prompt(
    original_topics: Any,
    condensed_topics: Any,
) -> str:
    """Generate prompt for condensation evaluation.

    Args:
        original_topics: Original topics before condensation
        condensed_topics: Topics after condensation

    Returns:
        Formatted prompt string
    """
    return CONDENSATION_EVAL.format(
        original_topics=original_topics,
        condensed_topics=condensed_topics,
    )


def generation_eval_prompt(
    topic_list_1: Any,
    topic_list_2: Any,
) -> str:
    """Generate prompt for generation evaluation (comparing two topic lists).

    Args:
        topic_list_1: First list of topics
        topic_list_2: Second list of topics

    Returns:
        Formatted prompt string
    """
    return GENERATION_EVAL.format(
        topic_list_1=topic_list_1,
        topic_list_2=topic_list_2,
    )


def refinement_eval_prompt(
    original_topics: Any,
    new_topics: Any,
) -> str:
    """Generate prompt for refinement evaluation.

    Args:
        original_topics: Original topics before refinement
        new_topics: Topics after refinement

    Returns:
        Formatted prompt string
    """
    return REFINEMENT_EVAL.format(
        original_topics=original_topics,
        new_topics=new_topics,
    )


def title_specificity_eval_prompt(
    theme_titles: Any,
) -> str:
    """Generate prompt for title specificity evaluation.

    Args:
        theme_titles: List of theme titles to evaluate

    Returns:
        Formatted prompt string
    """
    return TITLE_SPECIFICITY_EVAL.format(
        theme_titles=theme_titles,
    )
