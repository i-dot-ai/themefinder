import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import openai
import pandas as pd
import tiktoken
from pydantic import BaseModel, ValidationError
from tenacity import (
    before,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from themefinder.llm import LLM, LLMResponse
from themefinder.themefinder_logging import logger


@dataclass
class BatchPrompt:
    prompt_string: str
    response_ids: list[int]


async def batch_and_run(
    input_df: pd.DataFrame,
    prompt_template: str | Path,
    llm: LLM,
    output_model: type[BaseModel],
    batch_size: int = 10,
    partition_key: str | None = None,
    integrity_check: bool = False,
    concurrency: int = 10,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a DataFrame of responses in batches using an LLM.

    Args:
        input_df: DataFrame containing input to be processed. Must include a 'response_id' column.
        prompt_template: Name of a prompt file (without .txt extension) or Path object.
        llm: LLM instance that will process the prompts.
        output_model: Pydantic model class for structured LLM output.
        batch_size: Number of input rows to process in each batch. Defaults to 10.
        partition_key: Optional column name to group input rows before batching.
        integrity_check: If True, verifies that all input response IDs are present in LLM output.
        concurrency: Maximum number of simultaneous LLM calls allowed. Defaults to 10.
        **kwargs: Additional keyword arguments to pass to the prompt template.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The first DataFrame contains rows successfully processed by the LLM
            - The second DataFrame contains rows that could not be processed
    """

    logger.info(f"Running batch and run with batch size {batch_size}")
    template_str = load_prompt_template(prompt_template)
    batch_prompts = generate_prompts(
        template_str,
        input_df,
        batch_size=batch_size,
        partition_key=partition_key,
        **kwargs,
    )
    processed_rows, failed_ids = await call_llm(
        batch_prompts=batch_prompts,
        llm=llm,
        output_model=output_model,
        integrity_check=integrity_check,
        concurrency=concurrency,
    )
    processed_results = process_llm_responses(processed_rows, input_df)

    if failed_ids:
        retry_df = input_df[input_df["response_id"].isin(failed_ids)]
        retry_prompts = generate_prompts(template_str, retry_df, batch_size=1, **kwargs)
        retry_results, unprocessable_ids = await call_llm(
            batch_prompts=retry_prompts,
            llm=llm,
            output_model=output_model,
            integrity_check=integrity_check,
            concurrency=concurrency,
        )
        retry_processed_results = process_llm_responses(retry_results, retry_df)
        unprocessable_df = retry_df.loc[retry_df["response_id"].isin(unprocessable_ids)]
        processed_results = pd.concat([processed_results, retry_processed_results])
    else:
        unprocessable_df = pd.DataFrame()
    return processed_results, unprocessable_df


def load_prompt_from_file(file_path: str | Path) -> str:
    """Load a prompt template from a text file in the prompts directory.

    Args:
        file_path: Name of the prompt file (without .txt extension) or Path object.

    Returns:
        Content of the prompt template file.
    """
    parent_dir = Path(__file__).parent
    with Path.open(parent_dir / "prompts" / f"{file_path}.txt") as file:
        return file.read()


def load_prompt_template(prompt_template: str | Path) -> str:
    """Load a prompt template string from a file.

    Args:
        prompt_template: Name of a prompt file (without .txt extension) or Path object.

    Returns:
        The prompt template string.

    Raises:
        TypeError: If prompt_template is not a str or Path.
    """
    if isinstance(prompt_template, str | Path):
        return load_prompt_from_file(prompt_template)
    msg = "Invalid prompt_template type. Expected str or Path."
    raise TypeError(msg)


def partition_dataframe(
    df: pd.DataFrame, partition_key: Optional[str]
) -> list[pd.DataFrame]:
    """Splits the DataFrame into partitions based on the partition_key if provided."""
    if partition_key:
        return [group.reset_index(drop=True) for _, group in df.groupby(partition_key)]
    return [df]


def split_overflowing_batch(
    batch: pd.DataFrame, allowed_tokens: int
) -> list[pd.DataFrame]:
    """
    Splits a DataFrame batch into smaller sub-batches such that each sub-batch's total token count
    does not exceed the allowed token limit.

    Args:
        batch: The input DataFrame to split.
        allowed_tokens: The maximum allowed number of tokens per sub-batch.

    Returns:
        A list of sub-batches, each within the token limit.
    """
    sub_batches = []
    current_indices = []
    current_token_sum = 0
    token_counts = batch.apply(
        lambda row: calculate_string_token_length(row.to_json()), axis=1
    ).tolist()

    for i, token_count in enumerate(token_counts):
        if token_count > allowed_tokens:
            logger.warning(
                f"Row at index {batch.index[i]} exceeds allowed token limit ({token_count} > {allowed_tokens}). Skipping row."
            )
            continue

        if current_token_sum + token_count > allowed_tokens:
            if current_indices:
                sub_batch = batch.iloc[current_indices].reset_index(drop=True)
                if not sub_batch.empty:
                    sub_batches.append(sub_batch)
            current_indices = [i]
            current_token_sum = token_count
        else:
            current_indices.append(i)
            current_token_sum += token_count

    if current_indices:
        sub_batch = batch.iloc[current_indices].reset_index(drop=True)
        if not sub_batch.empty:
            sub_batches.append(sub_batch)
    return sub_batches


def batch_task_input_df(
    df: pd.DataFrame,
    allowed_tokens: int,
    batch_size: int,
    partition_key: Optional[str] = None,
) -> list[pd.DataFrame]:
    """
    Partitions and batches a DataFrame according to a token limit and batch size,
    optionally using a partition key.

    Args:
        df: The input DataFrame to batch.
        allowed_tokens: Maximum allowed tokens per batch.
        batch_size: Maximum number of rows per batch before token filtering.
        partition_key: Column name to partition the DataFrame by.

    Returns:
        A list of batches, each within the specified token and size limits.
    """
    batches = []
    partitions = partition_dataframe(df, partition_key)

    for partition in partitions:
        partition_batches = [
            partition.iloc[i : i + batch_size].reset_index(drop=True)
            for i in range(0, len(partition), batch_size)
        ]
        for batch in partition_batches:
            batch_length = calculate_string_token_length(batch.to_json())
            if batch_length <= allowed_tokens:
                batches.append(batch)
            else:
                sub_batches = split_overflowing_batch(batch, allowed_tokens)
                batches.extend(sub_batches)
    return batches


def generate_prompts(
    template_str: str,
    input_data: pd.DataFrame,
    batch_size: int = 50,
    max_prompt_length: int = 50_000,
    partition_key: str | None = None,
    **kwargs,
) -> list[BatchPrompt]:
    """
    Generate a list of BatchPrompt objects by splitting the input DataFrame into batches
    and formatting each batch using a prompt template string.

    Args:
        template_str: The prompt template string with {variable} placeholders.
        input_data: A DataFrame containing the input responses, with at least a 'response_id' column.
        batch_size: Maximum number of rows to include in each batch. Defaults to 50.
        max_prompt_length: The maximum total token length allowed for the prompt. Defaults to 50,000.
        partition_key: Column name used to partition the DataFrame before batching.
        **kwargs: Additional keyword arguments to pass to the template's format method.

    Returns:
        A list of BatchPrompt objects.
    """
    prompt_token_length = calculate_string_token_length(template_str)
    allowed_tokens_for_data = max_prompt_length - prompt_token_length
    batches = batch_task_input_df(
        input_data, allowed_tokens_for_data, batch_size, partition_key
    )
    prompts = [build_prompt(template_str, batch, **kwargs) for batch in batches]
    return prompts


async def call_llm(
    batch_prompts: list[BatchPrompt],
    llm: LLM,
    output_model: type[BaseModel],
    concurrency: int = 10,
    integrity_check: bool = False,
) -> tuple[list[dict], list[int]]:
    """Process multiple batches of prompts concurrently through an LLM with retry logic.

    Returns:
        Tuple of (processed_rows, failed_ids).
    """
    semaphore = asyncio.Semaphore(concurrency)

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        reraise=True,
    )
    async def async_llm_call(batch_prompt) -> tuple[list[dict], list[int]]:
        async with semaphore:
            try:
                llm_response: LLMResponse = await llm.ainvoke(
                    batch_prompt.prompt_string, output_model=output_model
                )
                all_results = (
                    llm_response.parsed.model_dump()
                    if hasattr(llm_response.parsed, "model_dump")
                    else llm_response.parsed
                )
                responses = (
                    all_results["responses"]
                    if isinstance(all_results, dict)
                    else all_results.responses
                )
            except (openai.BadRequestError, ValueError) as e:
                logger.warning(e)
                return [], batch_prompt.response_ids
            except ValidationError as e:
                logger.warning(e)
                return [], batch_prompt.response_ids

            if integrity_check:
                failed_ids = get_missing_response_ids(
                    batch_prompt.response_ids, all_results
                )
                return responses, failed_ids
            else:
                return responses, []

    results = await asyncio.gather(
        *[async_llm_call(batch_prompt) for batch_prompt in batch_prompts]
    )
    valid_inputs = [row for result, _ in results for row in result]
    failed_response_ids = [
        failed_response_id
        for _, batch_failures in results
        for failed_response_id in batch_failures
    ]

    return valid_inputs, failed_response_ids


def get_missing_response_ids(
    input_response_ids: list[int], parsed_response: dict
) -> list[int]:
    """Identify which response IDs are missing from the LLM's parsed response.

    Args:
        input_response_ids: List of response IDs that were included in the original prompt.
        parsed_response: Parsed response from the LLM containing a 'responses' key.

    Returns:
        List of response IDs that are missing from the parsed response.
    """

    response_ids_set = {int(response_id) for response_id in input_response_ids}
    returned_ids_set = {
        int(element["response_id"])
        for element in parsed_response["responses"]
        if element.get("response_id", False)
    }

    missing_ids = list(response_ids_set - returned_ids_set)
    if missing_ids:
        logger.info(f"Missing response IDs from LLM output: {missing_ids}")
    return missing_ids


def process_llm_responses(
    llm_responses: list[dict[str, Any]], responses: pd.DataFrame
) -> pd.DataFrame:
    """Process and merge LLM responses with the original DataFrame.

    Args:
        llm_responses: List of LLM response dictionaries.
        responses: Original DataFrame containing the input responses.

    Returns:
        A merged DataFrame.
    """
    responses.loc[:, "response_id"] = responses["response_id"].astype(int)
    task_responses = pd.DataFrame(llm_responses)
    if "response_id" in task_responses.columns:
        task_responses["response_id"] = task_responses["response_id"].astype(int)
        return responses.merge(task_responses, how="inner", on="response_id")
    return task_responses


def calculate_string_token_length(input_text: str, model: str = None) -> int:
    """
    Calculates the number of tokens in a given string using the specified model's tokenizer.

    Args:
        input_text: The input string to tokenize.
        model: The model name used for tokenization. Defaults to MODEL_NAME env var or "gpt-4o".

    Returns:
        The number of tokens in the input string.
    """
    # Use the MODEL_NAME env var if no model is provided; otherwise default to "gpt-4o"
    model = model or os.environ.get("MODEL_NAME", "gpt-4o")
    tokenizer_encoding = tiktoken.encoding_for_model(model)
    number_of_tokens = len(tokenizer_encoding.encode(input_text))
    return number_of_tokens


def build_prompt(template_str: str, input_batch: pd.DataFrame, **kwargs) -> BatchPrompt:
    """
    Constructs a BatchPrompt by formatting a template string with a batch of responses.

    Args:
        template_str: The prompt template string with {variable} placeholders.
        input_batch: A DataFrame containing the batch of responses.
        **kwargs: Additional keyword arguments to pass to the format method.

    Returns:
        A BatchPrompt containing the formatted prompt string and response IDs.
    """
    prompt = template_str.format(
        responses=input_batch.to_dict(orient="records"), **kwargs
    )
    response_ids = input_batch["response_id"].astype(int).to_list()
    return BatchPrompt(prompt_string=prompt, response_ids=response_ids)
