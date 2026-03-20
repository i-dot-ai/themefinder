import logging

import pandas as pd

from themefinder.advanced_tasks.theme_clustering_agent import ThemeClusteringAgent
from themefinder.llm import LLM
from themefinder.llm_batch_processor import batch_and_run
from themefinder.models import (
    DetailDetectionResponses,
    TaskResult,
    ThemeCondensationResponses,
    ThemeGenerationResponses,
    ThemeMappingResponses,
    ThemeNode,
    ThemeRefinementResponses,
)
from themefinder.prompts import (
    CONSULTATION_SYSTEM_PROMPT,
    DETAIL_DETECTION,
    THEME_CONDENSATION,
    THEME_GENERATION,
    THEME_MAPPING,
    THEME_REFINEMENT,
)
from themefinder.themefinder_logging import logger


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
    1. Initial theme generation
    2. Theme condensation (combining similar themes)
    3. Theme refinement
    4. Mapping responses to refined themes
    5. Detail detection

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
            - themes: DataFrame with the final themes output
            - mapping: DataFrame mapping responses to final themes
            - detailed_responses: DataFrame with detail detection results
            - unprocessables: DataFrame containing inputs that could not be processed
    """
    logger.setLevel(logging.INFO if verbose else logging.CRITICAL)

    theme_result = await theme_generation(
        responses_df,
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    condensed_result = await theme_condensation(
        theme_result.output,
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    refined_result = await theme_refinement(
        condensed_result.output,
        llm,
        question=question,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )

    mapping_result = await theme_mapping(
        responses_df[["response_id", "response"]],
        llm,
        question=question,
        refined_themes_df=refined_result.output,
        system_prompt=system_prompt,
        concurrency=concurrency,
    )
    detail_result = await detail_detection(
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
        "themes": refined_result.output,
        "mapping": mapping_result.output,
        "detailed_responses": detail_result.output,
        "unprocessables": mapping_result.failures,
    }


async def theme_generation(
    responses_df: pd.DataFrame,
    llm: LLM,
    question: str,
    batch_size: int = 50,
    partition_key: str | None = None,
    prompt_template: str = THEME_GENERATION,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> TaskResult:
    """Generate themes from survey responses using an LLM.

    Args:
        responses_df: DataFrame containing survey responses.
        llm: LLM instance to use for theme generation.
        question: The survey question.
        batch_size: Number of responses to process in each batch.
        partition_key: Column name to use for batching related responses together.
        prompt_template: Prompt template string.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        TaskResult with output and failures DataFrames.
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
    prompt_template: str = THEME_CONDENSATION,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
    **kwargs,
) -> TaskResult:
    """Condense and combine similar themes identified from survey responses.

    When the theme count exceeds the batch size, a first pass condenses within
    each batch independently, then a second pass merges across batches.

    Args:
        themes_df: DataFrame containing the initial themes.
        llm: LLM instance to use for theme condensation.
        question: The survey question.
        batch_size: Number of themes to process in each batch.
        prompt_template: Prompt template string.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        TaskResult with output and failures DataFrames.
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
        condense_result = await batch_and_run(
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
        themes_df = condense_result.output.sample(frac=1).reset_index(drop=True)
        themes_df["response_id"] = themes_df.index + 1

        if len(themes_df) == original_theme_count:
            retry += 1
            if retry > 1:
                logger.warning("failed to reduce the number of themes after 1 retry")
                break
        else:
            retry = 0

    final_result = await batch_and_run(
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

    logger.info(f"Final number of condensed themes: {final_result.output.shape[0]}")
    return final_result


async def theme_clustering(
    themes_df: pd.DataFrame,
    llm: LLM,
    max_iterations: int = 5,
    target_themes: int = 10,
    significance_percentage: float = 10.0,
    return_all_themes: bool = False,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> TaskResult:
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
        TaskResult with output and failures DataFrames.
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
        return TaskResult(output=all_themes_df, failures=pd.DataFrame())
    else:
        # Select significant themes
        logger.info(
            f"Selecting themes with significance_percentage={significance_percentage}%"
        )
        selected_themes_df = agent.select_themes(significance_percentage)
        logger.info(
            f"Clustering complete: returning {len(selected_themes_df)} significant themes"
        )
        return TaskResult(output=selected_themes_df, failures=pd.DataFrame())


async def theme_refinement(
    condensed_themes_df: pd.DataFrame,
    llm: LLM,
    question: str,
    batch_size: int = 10000,
    prompt_template: str = THEME_REFINEMENT,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> TaskResult:
    """Refine and standardise condensed themes using an LLM.

    Args:
        condensed_themes_df: DataFrame containing the condensed themes.
        llm: LLM instance to use for theme refinement.
        question: The survey question.
        batch_size: Number of themes to process in each batch.
        prompt_template: Prompt template string.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        TaskResult with output and failures DataFrames.
    """
    logger.info(f"Running theme refinement on {len(condensed_themes_df)} responses")
    condensed_themes_df["response_id"] = condensed_themes_df.index + 1

    refine_result = await batch_and_run(
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

    refined_themes = assign_sequential_topic_ids(refine_result.output)

    return TaskResult(output=refined_themes, failures=refine_result.failures)


async def theme_mapping(
    responses_df: pd.DataFrame,
    llm: LLM,
    question: str,
    refined_themes_df: pd.DataFrame,
    batch_size: int = 20,
    prompt_template: str = THEME_MAPPING,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> TaskResult:
    """Map survey responses to refined themes using an LLM.

    Args:
        responses_df: DataFrame containing survey responses.
        llm: LLM instance to use for theme mapping.
        question: The survey question.
        refined_themes_df: DataFrame of refined themes.
        batch_size: Number of responses to process in each batch.
        prompt_template: Prompt template string.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        TaskResult with output and failures DataFrames.
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
    prompt_template: str = DETAIL_DETECTION,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> TaskResult:
    """Identify responses that provide high-value detailed evidence.

    Args:
        responses_df: DataFrame containing survey responses to analyze.
        llm: LLM instance to use for detail detection.
        question: The survey question.
        batch_size: Number of responses to process in each batch.
        prompt_template: Prompt template string.
        system_prompt: System prompt to guide the LLM's behavior.
        concurrency: Number of concurrent API calls to make.

    Returns:
        TaskResult with output and failures DataFrames.
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
