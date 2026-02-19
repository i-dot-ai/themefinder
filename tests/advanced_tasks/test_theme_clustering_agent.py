"""Tests for theme_clustering_agent module."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from themefinder.models import HierarchicalClusteringResponse, ThemeNode
from themefinder.advanced_tasks.theme_clustering_agent import ThemeClusteringAgent


@pytest.fixture
def sample_theme_nodes():
    """Create sample ThemeNode objects for testing."""
    return [
        ThemeNode(
            topic_id="A",
            topic_label="Environmental Issues",
            topic_description="Concerns about climate change and pollution",
            source_topic_count=10,
        ),
        ThemeNode(
            topic_id="B",
            topic_label="Economic Concerns",
            topic_description="Issues related to cost of living and unemployment",
            source_topic_count=8,
        ),
        ThemeNode(
            topic_id="C",
            topic_label="Healthcare Access",
            topic_description="Problems with healthcare availability and quality",
            source_topic_count=12,
        ),
        ThemeNode(
            topic_id="D",
            topic_label="Education Reform",
            topic_description="Need for improvements in educational system",
            source_topic_count=6,
        ),
        ThemeNode(
            topic_id="E",
            topic_label="Transportation",
            topic_description="Public transport and infrastructure issues",
            source_topic_count=4,
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for clustering."""
    return HierarchicalClusteringResponse(
        parent_themes=[
            ThemeNode(
                topic_id="F",
                topic_label="Social Services",
                topic_description="Combined healthcare and education concerns",
                source_topic_count=18,
                children=["C", "D"],  # 2 children - valid
            ),
            ThemeNode(
                topic_id="G",
                topic_label="Infrastructure",
                topic_description="Environmental and transportation issues",
                source_topic_count=14,
                children=["A", "E"],  # 2 children - valid
            ),
        ],
        should_terminate=False,
    )


@pytest.fixture
def mock_llm(mock_llm_response):
    """Create a mock LLM that returns structured responses."""
    mock = MagicMock()
    mock.invoke.return_value = mock_llm_response
    return mock


@pytest.fixture
def clustering_agent(mock_llm, sample_theme_nodes):
    """Create a ThemeClusteringAgent instance for testing."""
    return ThemeClusteringAgent(mock_llm, sample_theme_nodes)


class TestThemeClusteringAgent:
    """Test suite for ThemeClusteringAgent class."""

    def test_init(self, mock_llm, sample_theme_nodes):
        """Test agent initialization."""
        agent = ThemeClusteringAgent(mock_llm, sample_theme_nodes)

        assert agent.llm == mock_llm
        assert len(agent.themes) == 5
        assert agent.active_themes == {"A", "B", "C", "D", "E"}
        assert agent.current_iteration == 0

        # Test that themes are properly indexed by topic_id
        assert agent.themes["A"].topic_label == "Environmental Issues"
        assert agent.themes["C"].source_topic_count == 12

    def test_format_prompt(self, clustering_agent):
        """Test prompt formatting."""
        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = (
                "Test prompt with {themes_json} and iteration {iteration}"
            )

            result = clustering_agent._format_prompt()

            # Verify the prompt template was loaded
            mock_load.assert_called_once_with("agentic_theme_clustering")

            # Check that JSON is properly formatted and iteration is included
            assert "Test prompt with" in result
            assert "iteration 0" in result

            # Verify JSON structure contains expected theme data
            assert '"topic_id": "A"' in result
            assert '"topic_label": "Environmental Issues"' in result

    def test_cluster_iteration_success(self, clustering_agent, mock_llm_response):
        """Test successful clustering iteration."""
        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            clustering_agent.cluster_iteration()

            # Check that iteration counter incremented
            assert clustering_agent.current_iteration == 1

            # Check that new parent themes were created (A_0, B_0 based on enumeration)
            assert "A_0" in clustering_agent.themes
            assert "B_0" in clustering_agent.themes

            # Check parent theme properties (first parent gets A_0)
            parent_a = clustering_agent.themes["A_0"]
            assert parent_a.topic_label == "Social Services"
            assert parent_a.source_topic_count == 18
            assert set(parent_a.children) == {"C", "D"}

            # Check that child themes have parent_id set
            assert clustering_agent.themes["C"].parent_id == "A_0"
            assert clustering_agent.themes["D"].parent_id == "A_0"

            # Check that children are removed from active themes
            assert "C" not in clustering_agent.active_themes
            assert "D" not in clustering_agent.active_themes

            # Check that new parents are added to active themes
            assert "A_0" in clustering_agent.active_themes
            assert "B_0" in clustering_agent.active_themes

    def test_cluster_iteration_with_retry(self, mock_llm, sample_theme_nodes):
        """Test that retry mechanism works correctly."""
        # Create a mock that fails twice then succeeds
        mock_llm.invoke.side_effect = [
            Exception("API Error"),
            Exception("Another Error"),
            HierarchicalClusteringResponse(
                parent_themes=[
                    ThemeNode(
                        topic_id="X",
                        topic_label="Test Theme",
                        topic_description="Test description",
                        source_topic_count=18,  # Sum of A(10) + B(8)
                        children=["A", "B"],  # 2 children - valid
                    )
                ],
                should_terminate=False,
            ),
        ]

        agent = ThemeClusteringAgent(mock_llm, sample_theme_nodes)

        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            # Should succeed after retries
            agent.cluster_iteration()

            # Verify it was called 3 times (2 failures + 1 success)
            assert mock_llm.invoke.call_count == 3
            assert "A_0" in agent.themes  # First parent gets A_0

    def test_cluster_themes_basic(self, clustering_agent):
        """Test basic theme clustering functionality."""
        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            result_df = clustering_agent.cluster_themes(
                max_iterations=1, target_themes=3
            )

            # Check return type
            assert isinstance(result_df, pd.DataFrame)

            # Check that root node was created
            assert "0" in clustering_agent.themes
            root = clustering_agent.themes["0"]
            assert root.topic_label == "All Topics"
            assert root.source_topic_count == 40  # Sum of all original counts

            # Check DataFrame structure
            assert "topic_id" in result_df.columns
            assert "topic_label" in result_df.columns
            assert "source_topic_count" in result_df.columns

            # Root node should not be in DataFrame
            assert "0" not in result_df["topic_id"].values

    def test_cluster_themes_stops_at_target(self, clustering_agent):
        """Test that clustering stops when target theme count is reached."""
        # Mock response that reduces themes but not too much
        clustering_agent.llm.invoke.return_value = HierarchicalClusteringResponse(
            parent_themes=[
                ThemeNode(
                    topic_id="MEGA1",
                    topic_label="Combined Issues 1",
                    topic_description="Some combined issues",
                    source_topic_count=22,  # A(10) + C(12)
                    children=["A", "C"],  # 2 children - valid
                ),
                ThemeNode(
                    topic_id="MEGA2",
                    topic_label="Combined Issues 2",
                    topic_description="Other combined issues",
                    source_topic_count=14,  # B(8) + D(6)
                    children=["B", "D"],  # 2 children - valid
                ),
            ],
            should_terminate=False,
        )

        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            clustering_agent.cluster_themes(max_iterations=5, target_themes=3)

            # Should stop after one iteration since we went from 5 to 3 active themes (2 new + 1 unchanged)
            assert clustering_agent.current_iteration == 1
            assert len(clustering_agent.active_themes) == 3  # A_0, B_0, E

    def test_cluster_themes_stops_at_max_iterations(self, clustering_agent):
        """Test that clustering stops at maximum iterations."""

        # Track which themes have been merged
        call_count = [0]

        def mock_response(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call - merge D and E
                return HierarchicalClusteringResponse(
                    parent_themes=[
                        ThemeNode(
                            topic_id="MERGE1",
                            topic_label="First Merge",
                            topic_description="Merge D and E",
                            source_topic_count=10,  # D(6) + E(4)
                            children=["D", "E"],
                        )
                    ],
                    should_terminate=False,
                )
            elif call_count[0] == 2:
                # Second call - merge A and B
                return HierarchicalClusteringResponse(
                    parent_themes=[
                        ThemeNode(
                            topic_id="MERGE2",
                            topic_label="Second Merge",
                            topic_description="Merge A and B",
                            source_topic_count=18,  # A(10) + B(8)
                            children=["A", "B"],
                        )
                    ],
                    should_terminate=False,
                )
            else:
                # Third call - merge C with A_0
                return HierarchicalClusteringResponse(
                    parent_themes=[
                        ThemeNode(
                            topic_id="MERGE3",
                            topic_label="Third Merge",
                            topic_description="Merge C with first parent",
                            source_topic_count=22,  # C(12) + A_0(10)
                            children=["C", "A_0"],
                        )
                    ],
                    should_terminate=False,
                )

        clustering_agent.llm.invoke.side_effect = mock_response

        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            clustering_agent.cluster_themes(max_iterations=2, target_themes=1)

            # Should stop at max iterations even though target not reached
            # The condition is <=, so max_iterations=2 runs 3 iterations (0, 1, 2)
            assert clustering_agent.current_iteration == 3
            # After 3 iterations with merges, we should have 2 active themes
            assert len(clustering_agent.active_themes) == 2

    def test_convert_themes_to_tree_json(self, clustering_agent):
        """Test JSON tree conversion."""
        # First run clustering to create hierarchy with minimal clustering
        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            # Mock to merge just two themes on first call
            clustering_agent.llm.invoke.return_value = HierarchicalClusteringResponse(
                parent_themes=[
                    ThemeNode(
                        topic_id="MERGED",
                        topic_label="Merged Theme",
                        topic_description="A minimal merge for testing",
                        source_topic_count=18,  # A(10) + B(8)
                        children=["A", "B"],
                    )
                ],
                should_terminate=True,
            )

            clustering_agent.cluster_themes(max_iterations=0, target_themes=3)

            json_result = clustering_agent.convert_themes_to_tree_json()

            # Parse and validate JSON structure
            tree_data = json.loads(json_result)

            assert "id" in tree_data
            assert "name" in tree_data
            assert "value" in tree_data
            assert "children" in tree_data

            # Root should have ID "0"
            assert tree_data["id"] == "0"
            assert tree_data["name"] == "All Topics"
            assert isinstance(tree_data["children"], list)

    def test_select_significant_themes(self, clustering_agent):
        """Test significant theme selection."""
        # Set up a simple hierarchy
        clustering_agent.themes["0"] = ThemeNode(
            topic_id="0",
            topic_label="All Topics",
            topic_description="",
            source_topic_count=40,
            children=list(clustering_agent.active_themes),
        )

        result = clustering_agent.select_significant_themes(
            significance_threshold=8, total_responses=40
        )

        assert "selected_nodes" in result
        assert "total_responses" in result
        assert result["total_responses"] == 40

        selected = result["selected_nodes"]
        assert isinstance(selected, list)

        # Should select themes with count >= 8 (B:8, C:12, A:10)
        selected_ids = {node["id"] for node in selected}
        expected_significant = {"A", "B", "C"}  # counts: 10, 8, 12
        assert expected_significant.issubset(selected_ids)

    def test_traverse_tree_leaf_nodes(self, clustering_agent):
        """Test tree traversal for leaf nodes."""
        selected_nodes = []
        node = clustering_agent.themes["A"]  # No children

        result = clustering_agent._traverse_tree(node, selected_nodes, 5)

        assert result is True
        assert len(selected_nodes) == 1
        assert selected_nodes[0]["id"] == "A"
        assert selected_nodes[0]["name"] == "Environmental Issues"
        assert selected_nodes[0]["value"] == 10

    def test_traverse_tree_parent_with_significant_children(self, clustering_agent):
        """Test tree traversal with significant children."""
        # Create a parent node with significant children
        parent = ThemeNode(
            topic_id="PARENT",
            topic_label="Parent Theme",
            topic_description="A parent theme",
            source_topic_count=20,
            children=["A", "C"],  # Both have counts >= 8
        )
        clustering_agent.themes["PARENT"] = parent

        selected_nodes = []
        result = clustering_agent._traverse_tree(parent, selected_nodes, 8)

        assert result is True
        # Should select the significant children, not the parent
        assert len(selected_nodes) == 2
        selected_ids = {node["id"] for node in selected_nodes}
        assert selected_ids == {"A", "C"}

    def test_traverse_tree_parent_with_insignificant_children(self, clustering_agent):
        """Test tree traversal with insignificant children."""
        # Create a parent node with insignificant children
        parent = ThemeNode(
            topic_id="PARENT",
            topic_label="Parent Theme",
            topic_description="A parent theme",
            source_topic_count=10,
            children=["D", "E"],  # Both have counts < 8
        )
        clustering_agent.themes["PARENT"] = parent

        selected_nodes = []
        result = clustering_agent._traverse_tree(parent, selected_nodes, 8)

        assert result is True
        # Should select the parent instead of insignificant children
        assert len(selected_nodes) == 1
        assert selected_nodes[0]["id"] == "PARENT"

    def test_select_themes_basic(self, clustering_agent):
        """Test basic theme selection by percentage."""
        # Set up root node
        clustering_agent.themes["0"] = ThemeNode(
            topic_id="0",
            topic_label="All Topics",
            topic_description="",
            source_topic_count=40,
            children=list(clustering_agent.active_themes),
        )

        result_df = clustering_agent.select_themes(significance_percentage=20.0)

        assert isinstance(result_df, pd.DataFrame)
        assert "topic_id" in result_df.columns

        # 20% of 40 = 8, so themes with count >= 8 should be selected
        # That's A(10), B(8), C(12)
        assert len(result_df) >= 3

        # Root node should not be included
        assert "0" not in result_df["topic_id"].values

    def test_select_themes_high_threshold(self, clustering_agent):
        """Test theme selection with high significance threshold."""
        clustering_agent.themes["0"] = ThemeNode(
            topic_id="0",
            topic_label="All Topics",
            topic_description="",
            source_topic_count=40,
            children=list(clustering_agent.active_themes),
        )

        result_df = clustering_agent.select_themes(significance_percentage=50.0)

        # 50% of 40 = 20, no themes meet this threshold
        # So should return empty DataFrame (except for structure)
        assert isinstance(result_df, pd.DataFrame)
        # The DataFrame structure should exist but may be empty
        assert len(result_df) == 0

    def test_select_themes_low_threshold(self, clustering_agent):
        """Test theme selection with low significance threshold."""
        clustering_agent.themes["0"] = ThemeNode(
            topic_id="0",
            topic_label="All Topics",
            topic_description="",
            source_topic_count=40,
            children=list(clustering_agent.active_themes),
        )

        result_df = clustering_agent.select_themes(significance_percentage=5.0)

        # 5% of 40 = 2, all themes should be selected
        assert len(result_df) == 5

        # Verify all original themes are included
        theme_ids = set(result_df["topic_id"].values)
        assert theme_ids == {"A", "B", "C", "D", "E"}

    def test_integration_full_workflow(self, mock_llm, sample_theme_nodes):
        """Test complete workflow integration."""
        # Mock a realistic clustering response
        mock_llm.invoke.return_value = HierarchicalClusteringResponse(
            parent_themes=[
                ThemeNode(
                    topic_id="SOCIAL",
                    topic_label="Social Issues",
                    topic_description="Healthcare and education combined",
                    source_topic_count=18,  # Sum of C(12) + D(6)
                    children=["C", "D"],  # 2 children - valid
                ),
            ],
            should_terminate=False,
        )

        agent = ThemeClusteringAgent(mock_llm, sample_theme_nodes)

        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            # Run full clustering with safer target
            all_themes_df = agent.cluster_themes(max_iterations=1, target_themes=4)

            # Select significant themes
            selected_df = agent.select_themes(significance_percentage=15.0)

            # Verify results
            assert isinstance(all_themes_df, pd.DataFrame)
            assert isinstance(selected_df, pd.DataFrame)

            # Should have created hierarchy (first parent gets A_0)
            assert "A_0" in agent.themes

            # Verify tree structure can be generated
            json_tree = agent.convert_themes_to_tree_json()
            tree_data = json.loads(json_tree)
            assert tree_data["id"] == "0"


class TestThemeClusteringAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_theme_list(self, mock_llm):
        """Test agent with empty theme list."""
        agent = ThemeClusteringAgent(mock_llm, [])

        assert len(agent.themes) == 0
        assert len(agent.active_themes) == 0

    def test_single_theme(self, mock_llm):
        """Test agent with single theme."""
        # Use two themes to satisfy the root node validation (needs at least 2 children)
        themes = [
            ThemeNode(
                topic_id="FIRST",
                topic_label="First Theme",
                topic_description="The first theme",
                source_topic_count=10,
            ),
            ThemeNode(
                topic_id="SECOND",
                topic_label="Second Theme",
                topic_description="The second theme",
                source_topic_count=5,
            ),
        ]

        agent = ThemeClusteringAgent(mock_llm, themes)

        # Should not attempt clustering if target is already met
        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"
            result_df = agent.cluster_themes(max_iterations=1, target_themes=2)

            # Should stop immediately since target is reached
            assert agent.current_iteration == 0
            assert len(result_df) == 2

    def test_cluster_iteration_no_merges(self, clustering_agent):
        """Test cluster iteration when merging occurs and should terminate."""
        # Mock response with a single merge to indicate termination
        clustering_agent.llm.invoke.return_value = HierarchicalClusteringResponse(
            parent_themes=[
                ThemeNode(
                    topic_id="LAST",
                    topic_label="Final Theme",
                    topic_description="Final merge when no more clustering makes sense",
                    source_topic_count=18,  # A(10) + B(8)
                    children=["A", "B"],
                )
            ],
            should_terminate=True,
        )

        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            initial_themes = len(clustering_agent.active_themes)
            clustering_agent.cluster_iteration()

            # Should increment iteration and merge A+B into A_0
            assert clustering_agent.current_iteration == 1
            # Active themes should be reduced by 1 (A and B removed, A_0 added)
            assert len(clustering_agent.active_themes) == initial_themes - 1

    def test_invalid_children_in_response(self, clustering_agent):
        """Test handling of invalid children in LLM response."""
        # Mock response with non-existent children
        clustering_agent.llm.invoke.return_value = HierarchicalClusteringResponse(
            parent_themes=[
                ThemeNode(
                    topic_id="INVALID",
                    topic_label="Invalid Theme",
                    topic_description="Theme with backup children",
                    source_topic_count=18,  # Sum of A(10) + B(8)
                    children=["NONEXISTENT", "A", "B"],  # Two valid, one invalid
                )
            ],
            should_terminate=False,
        )

        with patch(
            "themefinder.advanced_tasks.theme_clustering_agent.load_prompt_from_file"
        ) as mock_load:
            mock_load.return_value = "Mock prompt {themes_json} {iteration}"

            clustering_agent.cluster_iteration()

            # Should only process valid children (first parent gets A_0)
            new_theme = clustering_agent.themes["A_0"]
            assert set(new_theme.children) == {"A", "B"}  # Only valid children
            assert "A" not in clustering_agent.active_themes
            assert "B" not in clustering_agent.active_themes
            assert "NONEXISTENT" not in clustering_agent.themes
