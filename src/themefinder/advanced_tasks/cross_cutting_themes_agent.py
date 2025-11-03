"""Cross-cutting themes analysis agent for theme analysis.

This module provides the CrossCuttingThemesAgent class for identifying
high-level cross-cutting themes across multiple questions using a language model.
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
from langchain_core.runnables import Runnable
from tenacity import (
    before,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from themefinder.models import (
    CrossCuttingThemeIdentificationResponse,
    CrossCuttingThemeMappingResponse,
)
from themefinder.llm_batch_processor import load_prompt_from_file
from themefinder.themefinder_logging import logger


class CrossCuttingThemesAgent:
    """Agent for identifying cross-cutting themes across multiple questions.

    This class manages the process of identifying high-level themes that
    span across different questions, mapping individual themes to those cross-cutting themes, and
    refining cross-cutting theme descriptions based on assigned themes.

    Attributes:
        llm: Language model instance for cross-cutting theme identification and refinement
        questions_themes: Dictionary mapping question numbers to theme DataFrames
        question_strings: Dictionary mapping question IDs to question text
        n_concepts: Number of high-level cross-cutting themes to identify
        concepts: List of identified cross-cutting themes with names and descriptions
        concept_assignments: Dictionary mapping cross-cutting theme names to assigned themes
        concept_descriptions: Enhanced descriptions for each cross-cutting theme
        total_themes: Total number of themes across all questions
    """

    def __init__(
        self,
        llm: Runnable,
        questions_themes: Dict[int, pd.DataFrame],
        question_strings: Optional[Dict[str, str]] = None,
        n_concepts: int = 5,
    ) -> None:
        """Initialize the cross-cutting themes agent.

        Args:
            llm: Language model instance for text generation
            questions_themes: Dictionary mapping question numbers to theme DataFrames
            question_strings: Optional dictionary mapping question IDs to question text
            n_concepts: Number of high-level cross-cutting themes to identify

        Raises:
            ValueError: If questions_themes is empty
        """
        self.llm = llm
        self.questions_themes = questions_themes
        self.question_strings = question_strings or {}
        self.n_concepts = n_concepts
        self.concepts: List[Dict[str, str]] = []
        self.concept_assignments: Dict[str, List[Dict[str, Any]]] = {}
        self.concept_descriptions: Dict[str, str] = {}

        # Validate input
        if not questions_themes:
            raise ValueError("questions_themes cannot be empty")

        # Count total themes for statistics
        self.total_themes = sum(len(df) for df in questions_themes.values())

        logger.info(
            f"Initialized CrossCuttingThemesAgent with {len(questions_themes)} questions, "
            f"{self.total_themes} total themes"
        )

    def _format_questions_and_themes(self) -> str:
        """Format all questions and themes for cross-cutting theme identification.

        Returns:
            Formatted string with all questions and their themes
        """
        formatted_lines = []

        for q_id, themes_df in self.questions_themes.items():
            # Get question text if available
            question_text = self.question_strings.get(str(q_id), f"Question {q_id}")

            # Get theme list
            theme_list = themes_df["topic"].to_list()

            # Format as single line
            formatted_lines.append(
                f"Question: {question_text}, theme list: {theme_list}"
            )

        return "\\n".join(formatted_lines)

    @retry(
        wait=wait_random_exponential(min=1, max=2),
        stop=stop_after_attempt(3),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    def identify_concepts(self) -> List[Dict[str, str]]:
        """Identify high-level cross-cutting themes across all questions.

        Uses a single LLM call to identify cross-cutting themes that unify individual themes
        across multiple questions in the consultation.

        Returns:
            List of cross-cutting theme dictionaries with 'name' and 'description' keys
        """
        logger.info(f"Identifying {self.n_concepts} high-level cross-cutting themes")

        # Format all questions and themes
        questions_and_themes = self._format_questions_and_themes()

        # Load prompt template
        prompt_template = load_prompt_from_file("cross_cutting_identification")

        # Create the prompt
        prompt = prompt_template.format(
            n_concepts=self.n_concepts, questions_and_themes=questions_and_themes
        )

        # Use structured output to get concepts
        structured_llm = self.llm.with_structured_output(
            CrossCuttingThemeIdentificationResponse
        )
        result = structured_llm.invoke(prompt)

        if isinstance(result, dict):
            result = CrossCuttingThemeIdentificationResponse(**result)

        # Convert to our expected format
        concepts = []
        for theme in result.themes:
            concepts.append({"name": theme.name, "description": theme.description})

        self.concepts = concepts
        logger.info(f"Identified {len(concepts)} cross-cutting themes")
        return concepts

    @retry(
        wait=wait_random_exponential(min=1, max=2),
        stop=stop_after_attempt(3),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    def map_themes_to_concepts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Map themes to identified cross-cutting themes using semantic similarity.

        This uses individual LLM calls per question to classify which themes
        belong to which cross-cutting themes.

        Returns:
            Dictionary mapping cross-cutting theme names to lists of assigned themes

        Raises:
            ValueError: If cross-cutting themes have not been identified yet
        """
        if not self.concepts:
            raise ValueError("Must call identify_concepts() first")

        logger.info("Mapping themes to cross-cutting themes for all questions")

        all_assignments = {}
        for concept in self.concepts:
            all_assignments[concept["name"]] = []

        # Process each question
        for q_id, themes_df in self.questions_themes.items():
            logger.info(f"Processing question {q_id}")

            # Get question text
            question_text = self.question_strings.get(str(q_id), f"Question {q_id}")

            # Create theme dictionary
            theme_dict = dict(zip(themes_df["topic_id"], themes_df["topic"]))

            # Format question input
            question_input = (
                f"Question {q_id}: {question_text}, theme dictionary: {theme_dict}"
            )

            # Format cross-cutting themes
            concepts_text = "\\n".join(
                [
                    f"{concept['name']}: {concept['description']}"
                    for concept in self.concepts
                ]
            )

            # Load prompt template
            prompt_template = load_prompt_from_file("cross_cutting_mapping")

            # Create mapping prompt
            prompt = prompt_template.format(
                question_input=question_input, concepts_text=concepts_text
            )

            # Use structured output to get mappings
            structured_llm = self.llm.with_structured_output(
                CrossCuttingThemeMappingResponse
            )
            result = structured_llm.invoke(prompt)

            if isinstance(result, dict):
                result = CrossCuttingThemeMappingResponse(**result)

            # Convert to our expected format
            question_assignments = {}
            for mapping in result.mappings:
                if mapping.theme_name in all_assignments:
                    question_assignments[mapping.theme_name] = mapping.theme_ids

            # Add to overall assignments
            for concept_name, theme_ids in question_assignments.items():
                if concept_name in all_assignments:
                    for theme_id in theme_ids:
                        all_assignments[concept_name].append(
                            {
                                "question_id": q_id,
                                "theme_id": theme_id,
                                "theme_text": theme_dict.get(theme_id, ""),
                            }
                        )

        self.concept_assignments = all_assignments
        logger.info(
            f"Completed theme mapping for {len(self.questions_themes)} questions"
        )

        return all_assignments

    @retry(
        wait=wait_random_exponential(min=1, max=2),
        stop=stop_after_attempt(3),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    def refine_concept_descriptions(self) -> Dict[str, str]:
        """Refine cross-cutting theme descriptions based on their assigned themes.

        Creates enhanced descriptions that capture insights and details
        from the themes actually assigned to each cross-cutting theme.

        Returns:
            Dictionary mapping cross-cutting theme names to refined descriptions

        Raises:
            ValueError: If theme mapping has not been performed yet
        """
        if not self.concept_assignments:
            raise ValueError("Must call map_themes_to_concepts() first")

        logger.info(
            "Refining cross-cutting theme descriptions based on assigned themes"
        )

        refined_descriptions = {}

        for concept_name, assignments in self.concept_assignments.items():
            if not assignments:
                # Keep original description if no themes assigned
                original = next(
                    (
                        c["description"]
                        for c in self.concepts
                        if c["name"] == concept_name
                    ),
                    "",
                )
                refined_descriptions[concept_name] = original
                continue

            # Format assigned themes for the prompt
            theme_lines = []
            for assignment in assignments:
                theme_lines.append(
                    f"Question {assignment['question_id']}, "
                    f"Theme {assignment['theme_id']}: {assignment['theme_text']}"
                )

            # Load prompt template
            prompt_template = load_prompt_from_file("cross_cutting_refinement")

            # Create refinement prompt
            prompt = prompt_template.format(
                concept_name=concept_name, theme_lines=chr(10).join(theme_lines)
            )

            # Get refined description
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            refined_descriptions[concept_name] = content.strip()
            logger.info(f"Refined description for '{concept_name}'")

        self.concept_descriptions = refined_descriptions
        return refined_descriptions

    def analyze(self) -> Dict[str, Any]:
        """Run the cross-cutting theme identification and mapping process.

        This orchestrates the analysis workflow:
        1. Identify high-level cross-cutting themes across all questions
        2. Map individual themes to the identified cross-cutting themes

        Returns:
            Dictionary with analysis results including cross-cutting themes and assignments
        """

        concepts = self.identify_concepts()
        assignments = self.map_themes_to_concepts()

        return {"concepts": concepts, "assignments": assignments}

    def get_results_as_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame format for compatibility.

        Returns:
            DataFrame with concepts and their assigned themes, compatible
            with other themefinder output formats
        """
        if not self.concepts or not self.concept_assignments:
            return pd.DataFrame()

        df_data = []

        for concept in self.concepts:
            concept_name = concept["name"]

            # Group themes by question
            themes_by_question = {}
            if concept_name in self.concept_assignments:
                for assignment in self.concept_assignments[concept_name]:
                    q_id = assignment["question_id"]
                    theme_id = assignment["theme_id"]

                    if q_id not in themes_by_question:
                        themes_by_question[q_id] = []
                    themes_by_question[q_id].append(theme_id)

            # Use refined description if available, otherwise original
            description = self.concept_descriptions.get(
                concept_name, concept["description"]
            )

            df_data.append(
                {
                    "name": concept_name,
                    "description": description,
                    "themes": themes_by_question,
                    "n_themes": sum(
                        len(themes) for themes in themes_by_question.values()
                    ),
                    "n_questions": len(themes_by_question),
                }
            )

        return pd.DataFrame(df_data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the concept identification analysis.

        Returns:
            Dictionary with analysis statistics including theme utilization,
            concept coverage, and processing metrics
        """
        # Count assigned themes
        assigned_themes = set()
        for assignments in self.concept_assignments.values():
            for assignment in assignments:
                assigned_themes.add((assignment["question_id"], assignment["theme_id"]))

        used_count = len(assigned_themes)

        return {
            "total_themes": self.total_themes,
            "used_themes": used_count,
            "unused_themes": self.total_themes - used_count,
            "utilization_rate": used_count / self.total_themes
            if self.total_themes > 0
            else 0,
            "n_concepts": len(self.concepts),
            "n_questions": len(self.questions_themes),
            "concepts_with_themes": sum(
                1 for assignments in self.concept_assignments.values() if assignments
            ),
        }
