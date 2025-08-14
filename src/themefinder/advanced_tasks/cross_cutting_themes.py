"""Cross-cutting themes analysis module.

This module provides a 2-step approach to identifying cross-cutting themes:
1. Pass all themes to LLM to create initial cross-cutting theme groups
2. Review unused themes and assign them to existing groups if appropriate
"""

from typing import Dict, List, Set, Tuple

import pandas as pd
from langchain.schema.runnable import RunnableWithFallbacks

from themefinder.models import (
    CrossCuttingThemesResponse,
    CrossCuttingThemeReviewResponse,
)
from themefinder.llm_batch_processor import load_prompt_from_file
from themefinder.themefinder_logging import logger


def analyze_cross_cutting_themes(
    questions_themes: Dict[int, pd.DataFrame],
    llm: RunnableWithFallbacks,
    system_prompt: str,
    min_themes: int = 3,
) -> List[Dict]:
    """Analyze themes to identify cross-cutting patterns using a 2-step approach.

    Args:
        questions_themes: Dictionary mapping question numbers to theme DataFrames
        llm: Language model instance configured for structured output
        system_prompt: System prompt to guide LLM behavior
        min_themes: Minimum number of themes required for valid cross-cutting theme

    Returns:
        List of cross-cutting theme dictionaries with name, description, and themes
    """
    logger.info("Starting 2-step cross-cutting themes analysis")

    # Step 1: Format themes data and get initial cross-cutting theme groups
    themes_data_str = _format_themes_for_prompt(questions_themes)
    initial_groups = _step1_identify_cross_cutting_themes(
        themes_data_str, llm, system_prompt
    )

    if not initial_groups:
        logger.info("No initial cross-cutting themes identified")
        return []

    logger.info(
        f"Step 1 complete: {len(initial_groups)} initial cross-cutting themes identified"
    )

    # Step 2: Review unused themes and add to existing groups
    used_themes = _get_used_themes(initial_groups)
    unused_themes = _get_unused_themes(questions_themes, used_themes)

    if unused_themes:
        logger.info(f"Step 2: Reviewing {len(unused_themes)} unused themes")

        final_groups = _step2_review_unused_themes(
            initial_groups, unused_themes, llm, system_prompt
        )
    else:
        logger.info("No unused themes to review")
        final_groups = initial_groups

    # Filter groups by minimum theme count
    filtered_groups = [
        group for group in final_groups if len(group["themes"]) >= min_themes
    ]

    logger.info(f"Analysis complete: {len(filtered_groups)} final cross-cutting themes")
    return filtered_groups


def _format_themes_for_prompt(questions_themes: Dict[int, pd.DataFrame]) -> str:
    """Format themes data as text for LLM prompt."""
    theme_lines = []

    for question_num, themes_df in questions_themes.items():
        for _, row in themes_df.iterrows():
            # Handle both formats: "topic" column or separate "topic_label" + "topic_description"
            if "topic" in themes_df.columns:
                # Split topic on first colon to get label and description
                topic_parts = row["topic"].split(":", 1)
                topic_label = topic_parts[0].strip()
                topic_description = (
                    topic_parts[1].strip() if len(topic_parts) > 1 else topic_label
                )
            else:
                topic_label = row.get("topic_label", "")
                topic_description = row.get("topic_description", "")

            theme_line = f"Question {question_num}, Theme {row['topic_id']}: {topic_label} - {topic_description}"
            theme_lines.append(theme_line)

    return "\n".join(theme_lines)


def _step1_identify_cross_cutting_themes(
    themes_data_str: str, llm: RunnableWithFallbacks, system_prompt: str
) -> List[Dict]:
    """Step 1: Identify initial cross-cutting theme groups."""
    prompt_template = load_prompt_from_file("cross_cutting_themes")
    prompt = prompt_template.format(
        system_prompt=system_prompt, themes_data=themes_data_str
    )

    # Use structured output to get CrossCuttingThemesResponse
    structured_llm = llm.with_structured_output(CrossCuttingThemesResponse)
    result = structured_llm.invoke(prompt)

    if isinstance(result, dict):
        result = CrossCuttingThemesResponse(**result)

    # Convert to our expected format
    groups = []
    for cc_theme in result.cross_cutting_themes:
        group = {
            "name": cc_theme.name,
            "description": cc_theme.description,
            "themes": [
                {"question_number": theme.question_number, "theme_key": theme.theme_key}
                for theme in cc_theme.themes
            ],
        }
        groups.append(group)

    return groups


def _step2_review_unused_themes(
    initial_groups: List[Dict],
    unused_themes: List[Dict],
    llm: RunnableWithFallbacks,
    system_prompt: str,
) -> List[Dict]:
    """Step 2: Review unused themes and add to existing groups if appropriate."""
    if not unused_themes:
        return initial_groups

    # Format existing groups for prompt
    existing_themes_str = "\n".join(
        [
            f"- {group['name']}: {group['description']}\n  Current themes: "
            + ", ".join(
                [f"Q{t['question_number']}-{t['theme_key']}" for t in group["themes"]]
            )
            for group in initial_groups
        ]
    )

    # Format unused themes for prompt
    unused_themes_str = "\n".join(
        [
            f"Q{theme['question_number']}-{theme['theme_key']}: {theme['label']} - {theme['description']}"
            for theme in unused_themes
        ]
    )

    prompt_template = load_prompt_from_file("cross_cutting_themes_review")
    prompt = prompt_template.format(
        system_prompt=system_prompt,
        existing_themes=existing_themes_str,
        unused_themes=unused_themes_str,
    )

    # Use structured output to get review response
    structured_llm = llm.with_structured_output(CrossCuttingThemeReviewResponse)
    review_result = structured_llm.invoke(prompt)

    if isinstance(review_result, dict):
        review_result = CrossCuttingThemeReviewResponse(**review_result)

    # Apply additions to groups
    final_groups = initial_groups.copy()
    for addition in review_result.additions:
        # Find the target group
        for group in final_groups:
            if group["name"] == addition.cross_cutting_theme_name:
                # Check if this question is already represented in the group
                existing_questions = {t["question_number"] for t in group["themes"]}
                if addition.question_number not in existing_questions:
                    group["themes"].append(
                        {
                            "question_number": addition.question_number,
                            "theme_key": addition.theme_key,
                        }
                    )
                    logger.info(
                        f"Added Q{addition.question_number}-{addition.theme_key} "
                        f"to '{addition.cross_cutting_theme_name}'"
                    )
                break

    return final_groups


def _get_used_themes(groups: List[Dict]) -> Set[Tuple[int, str]]:
    """Get set of (question_number, theme_key) tuples that are used in groups."""
    used = set()
    for group in groups:
        for theme in group["themes"]:
            used.add((theme["question_number"], theme["theme_key"]))
    return used


def _get_unused_themes(
    questions_themes: Dict[int, pd.DataFrame], used_themes: Set[Tuple[int, str]]
) -> List[Dict]:
    """Get list of unused theme dictionaries."""
    unused = []

    for question_num, themes_df in questions_themes.items():
        for _, row in themes_df.iterrows():
            theme_key = (question_num, row["topic_id"])
            if theme_key not in used_themes:
                # Handle both topic formats
                if "topic" in themes_df.columns:
                    topic_parts = row["topic"].split(":", 1)
                    topic_label = topic_parts[0].strip()
                    topic_description = (
                        topic_parts[1].strip() if len(topic_parts) > 1 else topic_label
                    )
                else:
                    topic_label = row.get("topic_label", "")
                    topic_description = row.get("topic_description", "")

                unused.append(
                    {
                        "question_number": question_num,
                        "theme_key": row["topic_id"],
                        "label": topic_label,
                        "description": topic_description,
                    }
                )

    return unused
