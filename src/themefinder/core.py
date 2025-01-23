import logging
from pathlib import Path

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from .llm_batch_processor import batch_and_run, load_prompt_from_file
from .themefinder_logging import logger


CONSULTATION_SYSTEM_PROMPT = load_prompt_from_file("consultation_system_prompt")


async def find_themes(
    responses_df: pd.DataFrame,
    llm: Runnable,
    question: str,
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Process survey responses through a multi-stage theme analysis pipeline.

    This pipeline performs sequential analysis steps:
    1. Sentiment analysis of responses
    2. Initial theme generation
    3. Theme condensation (combining similar themes)
    4. Theme refinement
    5. Mapping responses to refined themes

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses
        llm (Runnable): Language model instance for text analysis
        question (str): The survey question
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.
        verbose (bool): Whether to show information messages during processing.
            Defaults to False.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing results from each pipeline stage:
            - question: The survey question
            - sentiment: DataFrame with sentiment analysis results
            - topics: DataFrame with initial generated themes
            - condensed_topics: DataFrame with combined similar themes
            - refined_topics: DataFrame with refined theme definitions
            - mapping: DataFrame mapping responses to final themes
    """
    # Set logging level based on verbose flag
    logger.setLevel(logging.INFO if verbose else logging.CRITICAL)

    sentiment_df = await sentiment_analysis(
        responses_df,
        llm,
        question=question,
        system_prompt=system_prompt,
    )
    theme_df = await theme_generation(
        sentiment_df,
        llm,
        question=question,
        system_prompt=system_prompt,
    )
    condensed_theme_df = await theme_condensation(
        theme_df, llm, question=question, system_prompt=system_prompt
    )
    refined_theme_df = await theme_refinement(
        condensed_theme_df,
        llm,
        question=question,
        system_prompt=system_prompt,
    )
    mapping_df = await theme_mapping(
        sentiment_df,
        llm,
        question=question,
        refined_themes_df=refined_theme_df,
        system_prompt=system_prompt,
    )

    logger.info("Finished finding themes")
    logger.info(
        "Provide feedback or report bugs: https://forms.gle/85xUSMvxGzSSKQ499 or themefinder@cabinetoffice.gov.uk"
    )
    return {
        "question": question,
        "sentiment": sentiment_df,
        "topics": theme_df,
        "condensed_topics": condensed_theme_df,
        "refined_topics": refined_theme_df,
        "mapping": mapping_df,
    }


async def sentiment_analysis(
    responses_df: pd.DataFrame,
    llm: Runnable,
    question: str,
    batch_size: int = 10,
    prompt_template: str | Path | PromptTemplate = "sentiment_analysis",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> pd.DataFrame:
    """Perform sentiment analysis on survey responses using an LLM.

    This function processes survey responses in batches to analyze their sentiment
    using a language model. It maintains response integrity by checking response IDs.

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses to analyze.
            Must contain 'response_id' and 'response' columns.
        llm (Runnable): Language model instance to use for sentiment analysis.
        question (str): The survey question.
        batch_size (int, optional): Number of responses to process in each batch.
            Defaults to 10.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "sentiment_analysis".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.

    Returns:
        pd.DataFrame: DataFrame containing the original responses enriched with
            sentiment analysis results.

    Note:
        The function uses response_id_integrity_check to ensure responses maintain
        their original order and association after processing.
    """
    logger.info(f"Running sentiment analysis on {len(responses_df)} responses")
    return await batch_and_run(
        responses_df,
        prompt_template,
        llm,
        batch_size=batch_size,
        question=question,
        response_id_integrity_check=True,
        system_prompt=system_prompt,
    )


async def theme_generation(
    responses_df: pd.DataFrame,
    llm: Runnable,
    question: str,
    batch_size: int = 50,
    partition_key: str | None = "position",
    prompt_template: str | Path | PromptTemplate = "theme_generation",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> pd.DataFrame:
    """Generate themes from survey responses using an LLM.

    This function processes batches of survey responses to identify common themes or topics.

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses.
            Must include 'response_id' and 'response' columns.
        llm (Runnable): Language model instance to use for theme generation.
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

    Returns:
        pd.DataFrame: DataFrame containing identified themes and their associated metadata.
    """
    logger.info(f"Running theme generation on {len(responses_df)} responses")
    return await batch_and_run(
        responses_df,
        prompt_template,
        llm,
        batch_size=batch_size,
        partition_key=partition_key,
        question=question,
        system_prompt=system_prompt,
    )


async def theme_condensation(
    themes_df: pd.DataFrame,
    llm: Runnable,
    question: str,
    batch_size: int = 10000,
    prompt_template: str | Path | PromptTemplate = "theme_condensation",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> pd.DataFrame:
    """Condense and combine similar themes identified from survey responses.

    This function processes the initially identified themes to combine similar or
    overlapping topics into more cohesive, broader categories using an LLM.

    Args:
        themes_df (pd.DataFrame): DataFrame containing the initial themes identified
            from survey responses.
        llm (Runnable): Language model instance to use for theme condensation.
        question (str): The survey question.
        batch_size (int, optional): Number of themes to process in each batch.
            Defaults to 10000.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "theme_condensation".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.

    Returns:
        pd.DataFrame: DataFrame containing the condensed themes, where similar topics
            have been combined into broader categories.
    """
    logger.info(f"Running theme condensation on {len(themes_df)} topics")
    themes_df["response_id"] = range(len(themes_df))
    return await batch_and_run(
        themes_df,
        prompt_template,
        llm,
        batch_size=batch_size,
        question=question,
        system_prompt=system_prompt,
    )


async def theme_refinement(
    condensed_themes_df: pd.DataFrame,
    llm: Runnable,
    question: str,
    batch_size: int = 10000,
    prompt_template: str | Path | PromptTemplate = "theme_refinement",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> pd.DataFrame:
    """Refine and standardize condensed themes using an LLM.

    This function processes previously condensed themes to create clear, standardized
    theme descriptions. It also transforms the output format for improved readability
    by transposing the results into a single-row DataFrame where columns represent
    individual themes.

    Args:
        condensed_themes (pd.DataFrame): DataFrame containing the condensed themes
            from the previous pipeline stage.
        llm (Runnable): Language model instance to use for theme refinement.
        question (str): The survey question.
        batch_size (int, optional): Number of themes to process in each batch.
            Defaults to 10000.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "topic_refinement".
        system_prompt (str): System prompt to guide the LLM's behavior.
            Defaults to CONSULTATION_SYSTEM_PROMPT.

    Returns:
        pd.DataFrame: A single-row DataFrame where:
            - Each column represents a unique theme (identified by topic_id)
            - The values contain the refined theme descriptions
            - The format is optimized for subsequent theme mapping operations

    Note:
        The function adds sequential response_ids to the input DataFrame and
        transposes the output for improved readability and easier downstream
        processing.
    """
    logger.info(f"Running topic refinement on {len(condensed_themes_df)} responses")
    condensed_themes_df["response_id"] = range(len(condensed_themes_df))

    def transpose_refined_topics(refined_themes: pd.DataFrame):
        """Transpose topics for increased legibility."""
        transposed_df = pd.DataFrame(
            [refined_themes["topic"].to_numpy()], columns=refined_themes["topic_id"]
        )
        return transposed_df

    refined_themes = await batch_and_run(
        condensed_themes_df,
        prompt_template,
        llm,
        batch_size=batch_size,
        question=question,
        system_prompt=system_prompt,
    )
    return transpose_refined_topics(refined_themes)


async def theme_mapping(
    responses_df: pd.DataFrame,
    llm: Runnable,
    question: str,
    refined_themes_df: pd.DataFrame,
    batch_size: int = 20,
    prompt_template: str | Path | PromptTemplate = "theme_mapping",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
) -> pd.DataFrame:
    """Map survey responses to refined themes using an LLM.

    This function analyzes each survey response and determines which of the refined
    themes best matches its content. Multiple themes can be assigned to a single response.

    Args:
        responses_df (pd.DataFrame): DataFrame containing survey responses.
            Must include 'response_id' and 'response' columns.
        llm (Runnable): Language model instance to use for theme mapping.
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

    Returns:
        pd.DataFrame: DataFrame containing the original responses enriched with
            theme mapping results, ensuring all responses are mapped through ID integrity checks.
    """
    logger.info(
        f"Running theme mapping on {len(responses_df)} responses using {len(refined_themes_df.columns)} themes"
    )
    return await batch_and_run(
        responses_df,
        prompt_template,
        llm,
        batch_size=batch_size,
        question=question,
        refined_themes=refined_themes_df.to_dict(orient="records"),
        response_id_integrity_check=True,
        system_prompt=system_prompt,
    )
