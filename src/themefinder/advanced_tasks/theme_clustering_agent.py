"""Theme clustering agent for hierarchical topic organization.

This module provides the ThemeClusteringAgent class for performing iterative
hierarchical clustering of topics using a language model.
"""

import json
import logging
from typing import Dict, List, Any

import pandas as pd
from langchain_core.runnables import Runnable
from tenacity import (
    before,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from themefinder.models import ThemeNode
from themefinder.llm_batch_processor import load_prompt_from_file
from themefinder.themefinder_logging import logger

CONSULTATION_SYSTEM_PROMPT = load_prompt_from_file("consultation_system_prompt")


class ThemeClusteringAgent:
    """Agent for performing hierarchical clustering of topics using language models.

    This class manages the iterative process of merging similar topics into a
    hierarchical structure using an LLM to identify semantic relationships and
    create meaningful parent-child topic relationships.

    Attributes:
        llm: Language model configured with structured output for clustering
        themes: Dictionary mapping topic IDs to ThemeNode objects
        active_themes: Set of topic IDs that are currently active for clustering
        current_iteration: Current iteration number in the clustering process
    """

    def __init__(
        self,
        llm: Runnable,
        themes: List[ThemeNode],
        system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
        target_themes: int = 10,
    ) -> None:
        """Initialize the clustering agent with an LLM and initial themes.

        Args:
            llm: Language model instance configured with structured output
                for HierarchicalClusteringResponse
            themes: List of ThemeNode objects to be clustered
            system_prompt: System prompt to guide the LLM's behavior
            target_themes: Target number of themes to cluster down to (default 10)
        """
        self.llm = llm
        self.themes: Dict[str, ThemeNode] = {}
        for theme in themes:
            self.themes[theme.topic_id] = theme
        self.active_themes = set(self.themes.keys())
        self.current_iteration = 0
        self.system_prompt = system_prompt
        self.target_themes = target_themes

    def _format_prompt(self) -> str:
        """Format the clustering prompt with current active themes.

        Creates a JSON representation of all currently active themes and
        formats them into the clustering prompt template.

        Returns:
            str: Formatted prompt string ready for LLM processing
        """
        themes_for_prompt = []
        for active_id in self.active_themes:
            theme_dict = {
                "topic_id": self.themes[active_id].topic_id,
                "topic_label": self.themes[active_id].topic_label,
                "topic_description": self.themes[active_id].topic_description,
            }
            themes_for_prompt.append(theme_dict)
        themes_json = json.dumps(themes_for_prompt, indent=2)

        # Load the clustering prompt template
        prompt_template = load_prompt_from_file("agentic_theme_clustering")
        return prompt_template.format(
            themes_json=themes_json,
            iteration=self.current_iteration,
            system_prompt=self.system_prompt,
            target_themes=self.target_themes,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=2),
        stop=stop_after_attempt(3),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    def cluster_iteration(self) -> None:
        """Perform one iteration of hierarchical theme clustering.

        Uses the configured LLM to identify semantically similar themes
        and merge them into parent themes. Updates the theme hierarchy
        and active theme set based on the clustering results.

        The method includes retry logic to handle transient API failures
        and will automatically retry up to 3 times with exponential backoff.

        Side Effects:
            - Creates new parent ThemeNode objects in self.themes
            - Updates parent_id relationships for child themes
            - Modifies self.active_themes set
            - Increments self.current_iteration
        """
        prompt = self._format_prompt()
        response = self.llm.invoke(prompt)
        for i, parent in enumerate(response.parent_themes):

            def to_alpha(idx: int) -> str:
                """Convert 0-based integer to Excel-style column name (A, B, ..., Z, AA, AB, ...) without divmod."""
                idx += 1  # 1-based for Excel logic
                result = []
                while idx > 0:
                    rem = (idx - 1) % 26
                    result.append(chr(65 + rem))
                    idx = (idx - 1) // 26
                return "".join(reversed(result))

            new_theme_id = f"{to_alpha(i)}_{self.current_iteration}"
            children = [c for c in parent.children if c in self.active_themes]
            for child in children:
                self.themes[child].parent_id = new_theme_id
            total_source_count = sum(
                self.themes[child_id].source_topic_count for child_id in children
            )
            new_theme = ThemeNode(
                topic_id=new_theme_id,
                topic_label=parent.topic_label,
                topic_description=parent.topic_description,
                source_topic_count=total_source_count,
                children=children,
            )
            self.themes[new_theme_id] = new_theme
            self.active_themes.add(new_theme_id)
            for child in children:
                self.active_themes.remove(child)
        self.current_iteration += 1

    def cluster_themes(
        self, max_iterations: int = 5, target_themes: int = 5
    ) -> pd.DataFrame:
        """Perform hierarchical clustering to reduce themes to target number.

        Iteratively merges similar themes using the clustering agent until
        either the maximum iterations is reached or the target number of
        themes is achieved. Creates a root node to represent the complete
        hierarchy.

        Args:
            max_iterations: Maximum number of clustering iterations to perform
            target_themes: Target number of themes to cluster down to

        Returns:
            pd.DataFrame: DataFrame containing all theme nodes (excluding root)
                with their hierarchical relationships and metadata
        """
        logger.info(f"Starting clustering with {len(self.active_themes)} active themes")
        while (
            self.current_iteration <= max_iterations
            and len(self.active_themes) > target_themes
        ):
            self.cluster_iteration()
            logger.info(
                f"After {self.current_iteration} iterations {len(self.active_themes)} active themes remaining"
            )
        root_node = ThemeNode(
            topic_id="0",
            topic_label="All Topics",
            topic_description="",
            source_topic_count=sum(
                self.themes[theme_id].source_topic_count
                for theme_id in self.active_themes
            ),
            children=list(self.active_themes),
        )
        self.themes["0"] = root_node
        for theme in self.active_themes:
            self.themes[theme].parent_id = "0"

        # Convert all themes (except root) to DataFrame
        theme_nodes_dicts = [
            node.model_dump() for node in self.themes.values() if node.topic_id != "0"
        ]
        return pd.DataFrame(theme_nodes_dicts)

    def convert_themes_to_tree_json(self) -> str:
        """Convert themes into a hierarchical JSON structure for visualization.

        Creates a nested JSON structure starting from the root node (ID '0')
        that represents the complete theme hierarchy. Each node includes
        metadata and references to its children.

        Returns:
            str: JSON string representing the hierarchical tree structure
                suitable for JavaScript tree visualization libraries
        """

        def build_tree(node: ThemeNode) -> Dict[str, Any]:
            return {
                "id": node.topic_id,
                "name": node.topic_label,
                "description": node.topic_description,
                "value": node.source_topic_count,
                "children": [
                    build_tree(self.themes[child_id])
                    for child_id in node.children
                    if child_id in self.themes
                ],
            }

        tree_data = build_tree(self.themes["0"])
        return json.dumps(tree_data, indent=2)

    def select_significant_themes(
        self, significance_threshold: int, total_responses: int
    ) -> Dict[str, Any]:
        """Select significant themes using depth-first traversal.

        Performs a depth-first search on the theme hierarchy to identify
        themes that meet the significance threshold. Prioritizes leaf nodes
        when possible, but selects parent nodes when children don't meet
        the threshold.

        Args:
            significance_threshold: Minimum source_topic_count for significance
            total_responses: Total number of responses across all themes

        Returns:
            Dict containing selected theme nodes and metadata
        """
        # Track selected nodes
        selected_nodes: List[Dict[str, Any]] = []

        # Perform the DFS selection
        self._traverse_tree(self.themes["0"], selected_nodes, significance_threshold)

        # Format the final result
        result = {"selected_nodes": selected_nodes, "total_responses": total_responses}

        return result

    def _traverse_tree(
        self,
        node: ThemeNode,
        selected_nodes: List[Dict[str, Any]],
        significance_threshold: int,
    ) -> bool:
        """Recursively traverse theme tree to select significant nodes.

        Implements depth-first traversal logic for theme selection:
        1. For leaf nodes: always select
        2. For parent nodes: select if no significant children exist
        3. For significant children: recursively process them

        Args:
            node: Current ThemeNode being processed
            selected_nodes: List to accumulate selected theme dictionaries
            significance_threshold: Minimum source_topic_count for significance

        Returns:
            bool: True if this node or descendants were selected, False otherwise
        """
        # Base case: if node has no children (leaf node)
        if not node.children:
            selected_nodes.append(
                {
                    "id": node.topic_id,
                    "name": node.topic_label,
                    "value": node.source_topic_count,
                }
            )
            return True

        # Check if any children are significant
        has_significant_children = any(
            self.themes[child_id].source_topic_count >= significance_threshold
            for child_id in node.children
            if child_id in self.themes
        )

        # If no significant children, select this node
        if not has_significant_children:
            selected_nodes.append(
                {
                    "id": node.topic_id,
                    "name": node.topic_label,
                    "value": node.source_topic_count,
                }
            )
            return True

        # If significant children exist, recursively process them
        any_selected = False
        for child_id in node.children:
            if child_id in self.themes:
                if self._traverse_tree(
                    self.themes[child_id], selected_nodes, significance_threshold
                ):
                    any_selected = True

        # If none of the children were selected, select this node
        if not any_selected:
            selected_nodes.append(
                {
                    "id": node.topic_id,
                    "name": node.topic_label,
                    "value": node.source_topic_count,
                }
            )
            return True

        return any_selected

    def select_themes(self, significance_percentage: float) -> pd.DataFrame:
        """Select themes that meet the significance threshold.

        Calculates the significance threshold based on the percentage of total
        responses and returns only themes that meet or exceed this threshold.
        Excludes the root node from results.

        Args:
            significance_percentage: Percentage (0-100) of total responses
                required for a theme to be considered significant

        Returns:
            pd.DataFrame: DataFrame containing significant theme data,
                excluding the root node (topic_id='0')
        """
        total_responses = self.themes["0"].source_topic_count
        # Convert percentage to absolute threshold
        significance_threshold = int(total_responses * (significance_percentage / 100))

        # Filter themes that meet the significance threshold
        significant_themes = [
            theme_node
            for theme_node in self.themes.values()
            if theme_node.source_topic_count >= significance_threshold
        ]
        # Convert significant themes to DataFrame, excluding root node
        theme_nodes_dicts = [
            node.model_dump() for node in significant_themes if node.topic_id != "0"
        ]
        return pd.DataFrame(theme_nodes_dicts)
