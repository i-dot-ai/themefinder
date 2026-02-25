import logging
from pathlib import Path

import pandas as pd

from themefinder.advanced_tasks.cross_cutting_themes_agent import (
    CrossCuttingThemesAgent,
)
from themefinder.advanced_tasks.theme_clustering_agent import ThemeClusteringAgent
from themefinder.llm import LLM
from themefinder.llm_batch_processor import batch_and_run, load_prompt_from_file
from themefinder.models import (
    DetailDetectionResponses,
    SentimentAnalysisResponses,
    ThemeCondensationResponses,
    ThemeGenerationResponses,
    ThemeMappingResponses,
    ThemeNode,
    ThemeRefinementResponses,
)
from themefinder.themefinder_logging import logger

CONSULTATION_SYSTEM_PROMPT = load_prompt_from_file("consultation_system_prompt")


async def find_themes(
    responses_df: pd.DataFrame,
    llm: LLM,
    question: str,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    verbose: bool = True,
    concurrency: int = 10,
) -> dict[str, str | pd.DataFrame]:
    """Process survey responses through a multi-stage theme analysis pipeline.

    This pipeline performs sequential analysis steps:
    1. Sentiment analysis of responses
    2. Initial theme generation
    3. Theme condensation (combining similar themes)
    4. Theme refinement
    5. Mapping responses to refined themes
    6. Detail detection

    Args:
        responses_df: DataFrame containing survey responses
        llm: LLM instance for text analysis
        question: The survey question
        system_prompt: System prompt to guide the LLM's behaviour.
        verbose: Whether to show information messages during processing.
        concurrency: Number of concurrent API calls to make.

    Returns:
        Dictionary containing results from each pipeline stage:
            - question: The survey question string
            - sentiment: DataFrame with sentiment analysis results
            - themes: DataFrame with the final themes output
            - mapping: DataFrame mapping responses to final themes
            - detailed_responses: DataFrame with detail detection results
            - unprocessables: DataFrame containing inputs that could not be processed
    """
    logger.setLevel(logging.INFO if verbose else logging.CRITICAL)

    sentiment_df, sentiment_unprocessables = await sentiment_analysis(
        responses_df,
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    theme_df, _ = await theme_generation(
        sentiment_df,
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

    mapping_df, mapping_unprocessables = await theme_mapping(
        sentiment_df[["response_id", "response"]],
        llm,
        question=question,
        refined_themes_df=refined_theme_df,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    detailed_df, _ = await detail_detection(
        responses_df[["response_id", "response"]],
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )

    logger.info("Finished finding themes")
    logger.info("Provide feedback or report bugs: packages@cabinetoffice.gov.uk")
    return {
        "question": question,
        "sentiment": sentiment_df,
        "themes": refined_theme_df,
        "mapping": mapping_df,
        "detailed_responses": detailed_df,
        "unprocessables": pd.concat([sentiment_unprocessables, mapping_unprocessables]),
    }


async def sentiment_analysis(
    responses_df: pd.DataFrame,
    llm: LLM,
    question: str,
    batch_size: int = 20,
    prompt_template: str | Path = "sentiment_analysis",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform sentiment analysis on survey responses using an LLM.

    Args:
        responses_df: DataFrame containing survey responses to analyze.
        llm: LLM instance to use for sentiment analysis.
        question: The survey question.
        batch_size: Number of responses to process in each batch.
        prompt_template: Name of a prompt file or Path object.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows)
    """
    logger.info(f"Running sentiment analysis on {len(responses_df)} responses")
    return await batch_and_run(
        responses_df,
        prompt_template,
        llm,
        output_model=SentimentAnalysisResponses,
        batch_size=batch_size,
        question=question,
        integrity_check=True,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )


async def theme_generation(
    responses_df: pd.DataFrame,
    llm: LLM,
    question: str,
    batch_size: int = 50,
    partition_key: str | None = None,
    prompt_template: str | Path = "theme_generation",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate themes from survey responses using an LLM.

    Args:
        responses_df: DataFrame containing survey responses.
        llm: LLM instance to use for theme generation.
        question: The survey question.
        batch_size: Number of responses to process in each batch.
        partition_key: Column name to use for batching related responses together.
        prompt_template: Name of a prompt file or Path object.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows)
    """
    logger.info(f"Running theme generation on {len(responses_df)} responses")
    return await batch_and_run(
        responses_df,
        prompt_template,
        llm,
        output_model=ThemeGenerationResponses,
        batch_size=batch_size,
        partition_key=partition_key,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )


async def theme_condensation(
    themes_df: pd.DataFrame,
    llm: LLM,
    question: str,
    batch_size: int = 75,
    prompt_template: str | Path = "theme_condensation",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condense and combine similar themes identified from survey responses.

    When the theme count exceeds the batch size, a first pass condenses within
    each batch independently, then a second pass merges across batches.

    Args:
        themes_df: DataFrame containing the initial themes.
        llm: LLM instance to use for theme condensation.
        question: The survey question.
        batch_size: Number of themes to process in each batch.
        prompt_template: Name of a prompt file or Path object.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows)
    """
    logger.info(f"Running theme condensation on {len(themes_df)} themes")
    themes_df["response_id"] = themes_df.index + 1

    target = 30
    retry = 0
    while len(themes_df) > target:
        original_theme_count = len(themes_df)
        logger.info(
            f"{len(themes_df)} larger than {target}, using recursive theme condensation"
        )
        themes_df, _ = await batch_and_run(
            themes_df,
            prompt_template,
            llm,
            output_model=ThemeCondensationResponses,
            batch_size=batch_size,
            question=question,
            system_prompt=system_prompt,
            concurrency=concurrency,
            **kwargs,
        )
        themes_df = themes_df.sample(frac=1).reset_index(drop=True)
        themes_df["response_id"] = themes_df.index + 1

        if len(themes_df) == original_theme_count:
            retry += 1
            if retry > 1:
                logger.warning("failed to reduce the number of themes after 1 retry")
                break
        else:
            retry = 0

    themes_df, _ = await batch_and_run(
        themes_df,
        prompt_template,
        llm,
        output_model=ThemeCondensationResponses,
        batch_size=batch_size,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
        **kwargs,
    )

    logger.info(f"Final number of condensed themes: {themes_df.shape[0]}")
    return themes_df, _


async def theme_clustering(
    themes_df: pd.DataFrame,
    llm: LLM,
    max_iterations: int = 5,
    target_themes: int = 10,
    significance_percentage: float = 10.0,
    return_all_themes: bool = False,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform hierarchical clustering of themes using an agentic approach.

    Args:
        themes_df: DataFrame containing themes.
        llm: LLM instance for clustering.
        max_iterations: Maximum number of clustering iterations.
        target_themes: Target number of themes to cluster down to.
        significance_percentage: Percentage threshold for selecting significant themes.
        return_all_themes: If True, returns all clustered themes.
        system_prompt: System prompt to guide the LLM's behavior.

    Returns:
        Tuple of (clustered themes DataFrame, empty DataFrame).
    """
    logger.info(f"Starting hierarchical clustering of {len(themes_df)} themes")

    # Convert DataFrame to ThemeNode objects
    initial_themes = [
        ThemeNode(
            topic_id=row["topic_id"],
            topic_label=row["topic_label"],
            topic_description=row["topic_description"],
            source_topic_count=row["source_topic_count"],
        )
        for _, row in themes_df.iterrows()
    ]

    # Initialise clustering agent
    agent = ThemeClusteringAgent(
        llm,
        initial_themes,
        system_prompt,
        target_themes,
    )

    # Perform clustering
    logger.info(
        f"Clustering themes with max_iterations={max_iterations}, target_themes={target_themes}"
    )
    all_themes_df = await agent.cluster_themes(
        max_iterations=max_iterations, target_themes=target_themes
    )

    # Return appropriate themes based on parameter
    if return_all_themes:
        logger.info(
            f"Clustering complete: returning all {len(all_themes_df)} clustered themes"
        )
        return all_themes_df, pd.DataFrame()
    else:
        # Select significant themes
        logger.info(
            f"Selecting themes with significance_percentage={significance_percentage}%"
        )
        selected_themes_df = agent.select_themes(significance_percentage)
        logger.info(
            f"Clustering complete: returning {len(selected_themes_df)} significant themes"
        )
        return selected_themes_df, pd.DataFrame()


async def theme_refinement(
    condensed_themes_df: pd.DataFrame,
    llm: LLM,
    question: str,
    batch_size: int = 10000,
    prompt_template: str | Path = "theme_refinement",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refine and standardise condensed themes using an LLM.

    Args:
        condensed_themes_df: DataFrame containing the condensed themes.
        llm: LLM instance to use for theme refinement.
        question: The survey question.
        batch_size: Number of themes to process in each batch.
        prompt_template: Name of a prompt file or Path object.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows)
    """
    logger.info(f"Running theme refinement on {len(condensed_themes_df)} responses")
    condensed_themes_df["response_id"] = condensed_themes_df.index + 1

    refined_themes, _ = await batch_and_run(
        condensed_themes_df,
        prompt_template,
        llm,
        output_model=ThemeRefinementResponses,
        batch_size=batch_size,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )

    def assign_sequential_topic_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Assigns sequential alphabetic topic_ids (A, B, ..., Z, AA, AB, ...) to the DataFrame."""

        def alpha_ids(n: int) -> list[str]:
            ids = []
            for i in range(n):
                s = ""
                x = i
                while True:
                    x, r = divmod(x, 26)
                    s = chr(65 + r) + s
                    if x == 0:
                        break
                    x -= 1
                ids.append(s)
            return ids

        if not df.empty:
            df["topic_id"] = alpha_ids(len(df))
        return df

    refined_themes = assign_sequential_topic_ids(refined_themes)

    return refined_themes, _


async def theme_mapping(
    responses_df: pd.DataFrame,
    llm: LLM,
    question: str,
    refined_themes_df: pd.DataFrame,
    batch_size: int = 20,
    prompt_template: str | Path = "theme_mapping",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map survey responses to refined themes using an LLM.

    Args:
        responses_df: DataFrame containing survey responses.
        llm: LLM instance to use for theme mapping.
        question: The survey question.
        refined_themes_df: DataFrame of refined themes.
        batch_size: Number of responses to process in each batch.
        prompt_template: Name of a prompt file or Path object.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows)
    """
    logger.info(
        f"Running theme mapping on {len(responses_df)} responses using {len(refined_themes_df)} themes"
    )

    def transpose_refined_themes(refined_themes: pd.DataFrame):
        """Transpose topics for increased legibility."""
        transposed_df = pd.DataFrame(
            [refined_themes["topic"].to_numpy()], columns=refined_themes["topic_id"]
        )
        return transposed_df

    return await batch_and_run(
        responses_df,
        prompt_template,
        llm,
        output_model=ThemeMappingResponses,
        batch_size=batch_size,
        question=question,
        refined_themes=transpose_refined_themes(refined_themes_df).to_dict(
            orient="records"
        ),
        integrity_check=True,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )


async def detail_detection(
    responses_df: pd.DataFrame,
    llm: LLM,
    question: str,
    batch_size: int = 20,
    prompt_template: str | Path = "detail_detection",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify responses that provide high-value detailed evidence.

    Args:
        responses_df: DataFrame containing survey responses to analyze.
        llm: LLM instance to use for detail detection.
        question: The survey question.
        batch_size: Number of responses to process in each batch.
        prompt_template: Name of a prompt file or Path object.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed results, unprocessable rows)
    """
    logger.info(f"Running detail detection on {len(responses_df)} responses")
    return await batch_and_run(
        responses_df,
        prompt_template,
        llm,
        output_model=DetailDetectionResponses,
        batch_size=batch_size,
        question=question,
        integrity_check=True,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )


async def cross_cutting_themes(
    questions_themes: dict[int, pd.DataFrame],
    llm: LLM,
    n_concepts: int = 5,
    min_themes: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify cross-cutting themes using a single-pass agent approach.

    Args:
        questions_themes: Dictionary mapping question numbers to their refined themes DataFrames.
        llm: LLM instance for theme analysis.
        n_concepts: The target number of cross-cutting themes to generate.
        min_themes: Minimum number of themes required for a valid cross-cutting theme group.

    Returns:
        Tuple of (cross-cutting themes DataFrame, empty DataFrame).
    """
    # Validate input
    if not questions_themes:
        raise ValueError("questions_themes cannot be empty")

    # Use the CrossCuttingThemesAgent with external prompt files
    agent = CrossCuttingThemesAgent(
        llm=llm, questions_themes=questions_themes, n_concepts=n_concepts
    )

    # Run the analysis
    await agent.analyze()

    # Get results as DataFrame using the agent's method
    df_results = agent.get_results_as_dataframe()

    # Apply minimum themes filter
    if min_themes > 0:
        df_results = df_results[df_results["n_themes"] >= min_themes]

    # Create and return DataFrame with empty unprocessed data for consistency
    return df_results.reset_index(drop=True), pd.DataFrame()
