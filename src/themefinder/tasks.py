import logging
from pathlib import Path

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithFallbacks

from themefinder.advanced_tasks.cross_cutting_themes_agent import (
    CrossCuttingThemesAgent,
)
from themefinder.advanced_tasks.theme_clustering_agent import ThemeClusteringAgent
from themefinder.llm_batch_processor import batch_and_run, load_prompt_from_file
from themefinder.models import (
    DetailDetectionResponses,
    HierarchicalClusteringResponse,
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
    llm: RunnableWithFallbacks,
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
    5. Theme target alignment (optional, if target_n_themes is specified)
    6. Mapping responses to refined themes

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses
        llm (RunnableWithFallbacks): Language model instance for text analysis
        question (str): The survey question
        target_n_themes (int | None, optional): Target number of themes to consolidate to.
            If None, skip theme target alignment step. Defaults to None.
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        verbose (bool): Whether to show information messages during processing.
            Defaults to True.
        concurrency (int): Number of concurrent API calls to make. Defaults to 10.

    Returns:
        dict[str, str | pd.DataFrame]: Dictionary containing results from each pipeline stage:
            - question: The survey question string
            - sentiment: DataFrame with sentiment analysis results
            - themes: DataFrame with the final themes output
            - mapping: DataFrame mapping responses to final themes
            - unprocessables: Dataframe containing the inputs that could not be processed by the LLM
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
    llm: RunnableWithFallbacks,
    question: str,
    batch_size: int = 20,
    prompt_template: str | Path | PromptTemplate = "sentiment_analysis",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform sentiment analysis on survey responses using an LLM.

    This function processes survey responses in batches to analyze their sentiment
    using a language model. It maintains response integrity by checking response IDs.

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses to analyze.
            Must contain 'response_id' and 'response' columns.
        llm (RunnableWithFallbacks): Language model instance to use for sentiment analysis.
        question (str): The survey question.
        batch_size (int, optional): Number of responses to process in each batch.
            Defaults to 20.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "sentiment_analysis".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        concurrency (int): Number of concurrent API calls to make. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing two DataFrames:
                - The first DataFrame contains the rows that were successfully processed by the LLM
                - The second DataFrame contains the rows that could not be processed by the LLM

    Note:
        The function uses integrity_check to ensure responses maintain
        their original order and association after processing.
    """
    logger.info(f"Running sentiment analysis on {len(responses_df)} responses")
    sentiment, unprocessable = await batch_and_run(
        responses_df,
        prompt_template,
        llm.with_structured_output(SentimentAnalysisResponses),
        batch_size=batch_size,
        question=question,
        integrity_check=True,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )

    return sentiment, unprocessable


async def theme_generation(
    responses_df: pd.DataFrame,
    llm: RunnableWithFallbacks,
    question: str,
    batch_size: int = 50,
    partition_key: str | None = None,
    prompt_template: str | Path | PromptTemplate = "theme_generation",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate themes from survey responses using an LLM.

    This function processes batches of survey responses to identify common themes or topics.

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses.
            Must include 'response_id' and 'response' columns.
        llm (RunnableWithFallbacks): Language model instance to use for theme generation.
        question (str): The survey question.
        batch_size (int, optional): Number of responses to process in each batch.
            Defaults to 50.
        partition_key (str | None, optional): Column name to use for batching related
            responses together. Defaults to "position" for sentiment-enriched responses,
            but can be set to None for sequential batching or another column name for
            different grouping strategies.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "theme_generation".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        concurrency (int): Number of concurrent API calls to make. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing two DataFrames:
                - The first DataFrame contains the rows that were successfully processed by the LLM
                - The second DataFrame contains the rows that could not be processed by the LLM

    """
    logger.info(f"Running theme generation on {len(responses_df)} responses")
    generated_themes, _ = await batch_and_run(
        responses_df,
        prompt_template,
        llm.with_structured_output(ThemeGenerationResponses),
        batch_size=batch_size,
        partition_key=partition_key,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    return generated_themes, _


async def theme_condensation(
    themes_df: pd.DataFrame,
    llm: RunnableWithFallbacks,
    question: str,
    batch_size: int = 75,
    prompt_template: str | Path | PromptTemplate = "theme_condensation",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condense and combine similar themes identified from survey responses.

    This function processes the initially identified themes to combine similar or
    overlapping topics into more cohesive, broader categories using an LLM.

    Args:
        themes_df (pd.DataFrame): DataFrame containing the initial themes identified
            from survey responses.
        llm (RunnableWithFallbacks): Language model instance to use for theme condensation.
        question (str): The survey question.
        batch_size (int, optional): Number of themes to process in each batch.
            Defaults to 100.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "theme_condensation".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        concurrency (int): Number of concurrent API calls to make. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing two DataFrames:
                - The first DataFrame contains the rows that were successfully processed by the LLM
                - The second DataFrame contains the rows that could not be processed by the LLM

    """
    logger.info(f"Running theme condensation on {len(themes_df)} themes")
    themes_df["response_id"] = themes_df.index + 1

    n_themes = themes_df.shape[0]
    while n_themes > batch_size:
        logger.info(
            f"{n_themes} larger than batch size, using recursive theme condensation"
        )
        themes_df, _ = await batch_and_run(
            themes_df,
            prompt_template,
            llm.with_structured_output(ThemeCondensationResponses),
            batch_size=batch_size,
            question=question,
            system_prompt=system_prompt,
            concurrency=concurrency,
            **kwargs,
        )
        themes_df = themes_df.sample(frac=1).reset_index(drop=True)
        themes_df["response_id"] = themes_df.index + 1
        if len(themes_df) == n_themes:
            logger.info("Themes no longer being condensed")
            break
        n_themes = themes_df.shape[0]

    themes_df, _ = await batch_and_run(
        themes_df,
        prompt_template,
        llm.with_structured_output(ThemeCondensationResponses),
        batch_size=batch_size,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
        **kwargs,
    )

    logger.info(f"Final number of condensed themes: {themes_df.shape[0]}")
    return themes_df, _


def theme_clustering(
    themes_df: pd.DataFrame,
    llm: RunnableWithFallbacks,
    max_iterations: int = 5,
    target_themes: int = 10,
    significance_percentage: float = 10.0,
    return_all_themes: bool = False,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform hierarchical clustering of themes using an agentic approach.

    This function takes a DataFrame of themes and uses the ThemeClusteringAgent
    to iteratively merge similar themes into a hierarchical structure, then
    selects the most significant themes based on a threshold.

    Args:
        themes_df (pd.DataFrame): DataFrame containing themes with columns:
            - topic_id: Unique identifier for each theme
            - topic_label: Short descriptive label for the theme
            - topic_description: Detailed description of the theme
            - source_topic_count: Number of source responses for this theme
        llm (RunnableWithFallbacks): Language model instance configured with
            structured output for HierarchicalClusteringResponse
        max_iterations (int, optional): Maximum number of clustering iterations.
            Defaults to 5.
        target_themes (int, optional): Target number of themes to cluster down to.
            Defaults to 10.
        significance_percentage (float, optional): Percentage threshold for
            selecting significant themes. Defaults to 10.0.
        return_all_themes (bool, optional): If True, returns all clustered themes.
            If False, returns only significant themes. Defaults to False.
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing:
                - DataFrame of clustered themes (all or significant based on return_all_themes)
                - Empty DataFrame (for consistency with other functions)
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

    # Initialize clustering agent with structured output LLM
    agent = ThemeClusteringAgent(
        llm.with_structured_output(HierarchicalClusteringResponse),
        initial_themes,
        system_prompt,
        target_themes,
    )

    # Perform clustering
    logger.info(
        f"Clustering themes with max_iterations={max_iterations}, target_themes={target_themes}"
    )
    all_themes_df = agent.cluster_themes(
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
    llm: RunnableWithFallbacks,
    question: str,
    batch_size: int = 10000,
    prompt_template: str | Path | PromptTemplate = "theme_refinement",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refine and standardize condensed themes using an LLM.

    This function processes previously condensed themes to create clear, standardized
    theme descriptions. It also transforms the output format for improved readability
    by transposing the results into a single-row DataFrame where columns represent
    individual themes.

    Args:
        condensed_themes (pd.DataFrame): DataFrame containing the condensed themes
            from the previous pipeline stage.
        llm (RunnableWithFallbacks): Language model instance to use for theme refinement.
        question (str): The survey question.
        batch_size (int, optional): Number of themes to process in each batch.
            Defaults to 10000.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "theme_refinement".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        concurrency (int): Number of concurrent API calls to make. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing two DataFrames:
                - The first DataFrame contains the rows that were successfully processed by the LLM
                - The second DataFrame contains the rows that could not be processed by the LLM

    Note:
        The function adds sequential response_ids to the input DataFrame and
        transposes the output for improved readability and easier downstream
        processing.
    """
    logger.info(f"Running theme refinement on {len(condensed_themes_df)} responses")
    condensed_themes_df["response_id"] = condensed_themes_df.index + 1

    refined_themes, _ = await batch_and_run(
        condensed_themes_df,
        prompt_template,
        llm.with_structured_output(ThemeRefinementResponses),
        batch_size=batch_size,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )

    def assign_sequential_topic_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns sequential alphabetic topic_ids (A, B, ..., Z, AA, AB, ...) to the DataFrame.
        """

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
    llm: RunnableWithFallbacks,
    question: str,
    refined_themes_df: pd.DataFrame,
    batch_size: int = 20,
    prompt_template: str | Path | PromptTemplate = "theme_mapping",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map survey responses to refined themes using an LLM.

    This function analyzes each survey response and determines which of the refined
    themes best matches its content. Multiple themes can be assigned to a single response.

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses.
            Must include 'response_id' and 'response' columns.
        llm (RunnableWithFallbacks): Language model instance to use for theme mapping.
        question (str): The survey question.
        refined_themes_df (pd.DataFrame): Single-row DataFrame where each column
            represents a theme (from theme_refinement stage).
        batch_size (int, optional): Number of responses to process in each batch.
            Defaults to 20.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "theme_mapping".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        concurrency (int): Number of concurrent API calls to make. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing two DataFrames:
                - The first DataFrame contains the rows that were successfully processed by the LLM
                - The second DataFrame contains the rows that could not be processed by the LLM

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

    mapping, unprocessable = await batch_and_run(
        responses_df,
        prompt_template,
        llm.with_structured_output(ThemeMappingResponses),
        batch_size=batch_size,
        question=question,
        refined_themes=transpose_refined_themes(refined_themes_df).to_dict(
            orient="records"
        ),
        integrity_check=True,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    return mapping, unprocessable


async def detail_detection(
    responses_df: pd.DataFrame,
    llm: RunnableWithFallbacks,
    question: str,
    batch_size: int = 20,
    prompt_template: str | Path | PromptTemplate = "detail_detection",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify responses that provide high-value detailed evidence.

    This function processes survey responses in batches to analyze their level of detail
    and evidence using a language model. It identifies responses that contain specific
    examples, data, or detailed reasoning that provide strong supporting evidence.

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses to analyze.
            Must contain 'response_id' and 'response' columns.
        llm (RunnableWithFallbacks): Language model instance to use for detail detection.
        question (str): The survey question.
        batch_size (int, optional): Number of responses to process in each batch.
            Defaults to 20.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "detail_detection".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        concurrency (int): Number of concurrent API calls to make. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing two DataFrames:
                - The first DataFrame contains the rows that were successfully processed by the LLM
                - The second DataFrame contains the rows that could not be processed by the LLM

    Note:
        The function uses response_id_integrity_check to ensure responses maintain
        their original order and association after processing.
    """
    logger.info(f"Running detail detection on {len(responses_df)} responses")
    detailed, _ = await batch_and_run(
        responses_df,
        prompt_template,
        llm.with_structured_output(DetailDetectionResponses),
        batch_size=batch_size,
        question=question,
        integrity_check=True,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    return detailed, _


def cross_cutting_themes(
    questions_themes: dict[int, pd.DataFrame],
    llm: RunnableWithFallbacks,
    n_concepts: int = 5,
    min_themes: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify cross-cutting themes using a single-pass agent approach.

    This function analyzes refined themes from multiple questions to identify semantic
    patterns that span across different questions, creating cross-cutting theme
    categories that represent common concerns or policy areas.

    The analysis uses a single-pass process:
    1. Identify high-level cross-cutting themes across all questions
    2. Map individual themes to the identified cross-cutting themes
    3. Refine descriptions based on assigned themes

    Args:
        questions_themes (dict[int, pd.DataFrame]): Dictionary mapping question numbers
            to their refined themes DataFrames. Each DataFrame should have columns:
            - topic_id: Theme identifier (e.g., 'A', 'B', 'C')
            - topic: String in format "topic_name: topic_description"
        llm (RunnableWithFallbacks): Language model instance configured for
            structured output
        n_concepts (int): The target number of cross-cutting themes to generate
        min_themes (int): Minimum number of themes required for a valid
            cross-cutting theme group. Groups with fewer themes will be discarded.
            Defaults to 5.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - DataFrame with cross-cutting themes with columns:
                - name: Name of the cross-cutting theme
                - description: Description of what this theme represents
                - themes: Dictionary mapping question_number to list of theme_keys
                  e.g., {1: ["A", "B"], 3: ["C"]}
            - Empty DataFrame (for consistency with other core functions)

    Raises:
        ValueError: If questions_themes is empty or contains invalid data
        KeyError: If required columns are missing from themes DataFrames
    """
    # Validate input
    if not questions_themes:
        raise ValueError("questions_themes cannot be empty")

    # Use the CrossCuttingThemesAgent with external prompt files
    agent = CrossCuttingThemesAgent(
        llm=llm, questions_themes=questions_themes, n_concepts=n_concepts
    )

    # Run the analysis
    agent.analyze()

    # Get results as DataFrame using the agent's method
    df_results = agent.get_results_as_dataframe()

    # Apply minimum themes filter
    if min_themes > 0:
        df_results = df_results[df_results["n_themes"] >= min_themes]

    # Create and return DataFrame with empty unprocessed data for consistency
    return df_results.reset_index(drop=True), pd.DataFrame()
