import logging
from pathlib import Path

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableWithFallbacks

from .llm_batch_processor import batch_and_run, load_prompt_from_file
from .models import (
    SentimentAnalysisResponses,
    ThemeGenerationResponses,
    ThemeCondensationResponses,
    ThemeRefinementResponses,
    ThemeMappingResponses,
    DetailDetectionResponses,
)
from .themefinder_logging import logger

CONSULTATION_SYSTEM_PROMPT = load_prompt_from_file("consultation_system_prompt")


async def find_themes(
    responses_df: pd.DataFrame,
    llm: RunnableWithFallbacks,
    question: str,
    target_n_themes: int | None = None,
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
    if target_n_themes is not None:
        refined_theme_df, _ = await theme_target_alignment(
            refined_theme_df,
            llm,
            question=question,
            target_n_themes=target_n_themes,
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
    logger.info(
        "Provide feedback or report bugs: https://forms.gle/85xUSMvxGzSSKQ499 or packages@cabinetoffice.gov.uk"
    )
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
    partition_key: str | None = "position",
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
    return refined_themes, _


async def theme_target_alignment(
    refined_themes_df: pd.DataFrame,
    llm: RunnableWithFallbacks,
    question: str,
    target_n_themes: int = 10,
    batch_size: int = 10000,
    prompt_template: str | Path | PromptTemplate = "theme_target_alignment",
    system_prompt: str = CONSULTATION_SYSTEM_PROMPT,
    concurrency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align themes to target number using an LLM.

    This function processes refined themes to consolidate them into a target number of
    distinct categories while preserving all significant details and perspectives.
    It transforms the output format for improved readability by transposing the
    results into a single-row DataFrame where columns represent individual themes.

    Args:
        refined_themes_df (pd.DataFrame): DataFrame containing the refined themes
            from the previous pipeline stage.
        llm (RunnableWithFallbacks): Language model instance to use for theme alignment.
        question (str): The survey question.
        target_n_themes (int, optional): Target number of themes to consolidate to.
            Defaults to 10.
        batch_size (int, optional): Number of themes to process in each batch.
            Defaults to 10000.
        prompt_template (str | Path | PromptTemplate, optional): Template for structuring
            the prompt to the LLM. Can be a string identifier, path to template file,
            or PromptTemplate instance. Defaults to "theme_target_alignment".
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
    logger.info(
        f"Running theme target alignment on {len(refined_themes_df)} themes compressing to {target_n_themes} themes"
    )
    refined_themes_df["response_id"] = refined_themes_df.index + 1
    aligned_themes, _ = await batch_and_run(
        refined_themes_df,
        prompt_template,
        llm.with_structured_output(ThemeRefinementResponses),
        batch_size=batch_size,
        question=question,
        system_prompt=system_prompt,
        target_n_themes=target_n_themes,
        concurrency=concurrency,
    )
    return aligned_themes, _


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
