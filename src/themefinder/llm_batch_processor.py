import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

import pandas as pd
import tiktoken
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError
from tenacity import before, retry, stop_after_attempt, wait_random_exponential

from .themefinder_logging import logger


@dataclass
class BatchPrompt:
    prompt_string: str
    response_ids: list[str]


async def batch_and_run(
    responses_df: pd.DataFrame,
    prompt_template: str | Path | PromptTemplate,
    llm: Runnable,
    batch_size: int = 10,
    partition_key: str | None = None,
    response_id_integrity_check: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Process a DataFrame of responses in batches using an LLM.

    Args:
        responses_df (pd.DataFrame): DataFrame containing responses to be processed.
            Must include a 'response_id' column.
        prompt_template (Union[str, Path, PromptTemplate]): Template for LLM prompts.
            Can be a string (file path), Path object, or PromptTemplate.
        llm (Runnable): LangChain Runnable instance that will process the prompts.
        batch_size (int, optional): Number of responses to process in each batch.
            Defaults to 10.
        partition_key (str | None, optional): Optional column name to group responses
            before batching. Defaults to None.
        response_id_integrity_check (bool, optional): If True, verifies that all input
            response IDs are present in LLM output and retries failed responses individually.
            If False, no integrity checking or retrying occurs. Defaults to False.
        **kwargs (Any): Additional keyword arguments to pass to the prompt template.

    Returns:
        pd.DataFrame: DataFrame containing the original responses merged with the
            LLM-processed results.
    """
    logger.info(f"Running batch and run with batch size {batch_size}")
    prompt_template = convert_to_prompt_template(prompt_template)
    batch_prompts = generate_prompts(
        prompt_template, responses_df, batch_size=batch_size, **kwargs
    )
    llm_responses, failed_ids = await call_llm(
        batch_prompts=batch_prompts,
        llm=llm,
        response_id_integrity_check=response_id_integrity_check,
    )
    processed_responses = process_llm_responses(llm_responses, responses_df)
    if failed_ids and response_id_integrity_check:
        new_df = responses_df[responses_df["response_id"].astype(str).isin(failed_ids)]
        processed_failed_responses = await batch_and_run(
            responses_df=new_df,
            prompt_template=prompt_template,
            llm=llm,
            batch_size=1,
            partition_key=partition_key,
            response_id_integrity_check=False,
            **kwargs,
        )
        return pd.concat(objs=[processed_failed_responses, processed_responses])
    return processed_responses


def load_prompt_from_file(file_path: str | Path) -> str:
    """Load a prompt template from a text file in the prompts directory.

    Args:
        file_path (str | Path): Name of the prompt file (without .txt extension)
            or Path object pointing to the file.

    Returns:
        str: Content of the prompt template file.
    """
    parent_dir = Path(__file__).parent
    with Path.open(parent_dir / "prompts" / f"{file_path}.txt") as file:
        return file.read()


def convert_to_prompt_template(prompt_template: str | Path | PromptTemplate):
    """Convert various input types to a LangChain PromptTemplate.

    Args:
        prompt_template (str | Path | PromptTemplate): Input template that can be either:
            - str: Name of a prompt file in the prompts directory (without .txt extension)
            - Path: Path object pointing to a prompt file
            - PromptTemplate: Already initialized LangChain PromptTemplate

    Returns:
        PromptTemplate: Initialized LangChain PromptTemplate object.

    Raises:
        TypeError: If prompt_template is not one of the expected types.
        FileNotFoundError: If using str/Path input and the prompt file doesn't exist.
    """
    if isinstance(prompt_template, str | Path):
        prompt_content = load_prompt_from_file(prompt_template)
        template = PromptTemplate.from_template(template=prompt_content)
    elif isinstance(prompt_template, PromptTemplate):
        template = prompt_template
    else:
        msg = "Invalid prompt_template type. Expected str, Path, or PromptTemplate."
        raise TypeError(msg)
    return template


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
    sub_batches = []
    current_indices = []
    current_token_sum = 0
    token_counts = batch.apply(
        lambda row: calculate_string_token_length(row.to_json()), axis=1
    ).tolist()

    for i, token_count in enumerate(token_counts):
        if token_count > allowed_tokens:
            logging.warning(
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
    prompt_template: PromptTemplate,
    input_data: pd.DataFrame,
    batch_size: int = 50,
    max_prompt_length: int = 50_000,
    partition_key: str | None = None,
    **kwargs,
) -> list[BatchPrompt]:
    """
    Generate a list of BatchPrompt objects by splitting the input DataFrame into batches
    and formatting each batch using a prompt template.

    The function first calculates the token length of the prompt template to determine
    the allowed tokens available for the input data. It then splits the input data into batches,
    optionally partitioning by a specified key. Each batch is then formatted into a prompt string
    using the provided prompt template, and a BatchPrompt is created containing the prompt string
    and a list of response IDs from the batch.

    Args:
        prompt_template (PromptTemplate): An object with a 'template' attribute and a 'format' method
            used to create a prompt string from a list of response dictionaries.
        input_data (pd.DataFrame): A DataFrame containing the input responses, with at least a
            'response_id' column.
        batch_size (int, optional): Maximum number of rows to include in each batch. Defaults to 50.
        max_prompt_length (int, optional): The maximum total token length allowed for the prompt,
            including both the prompt template and the input data. Defaults to 50,000.
        partition_key (str | None, optional): Column name used to partition the DataFrame before batching.
            If provided, the DataFrame will be grouped by this key so that rows with the same value
            remain in the same batch. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the prompt template's format method.

    Returns:
        list[BatchPrompt]: A list of BatchPrompt objects where each object contains:
            - prompt_string: The formatted prompt string for a batch.
            - response_ids: A list of response IDs corresponding to the rows in that batch.
    """
    prompt_token_length = calculate_string_token_length(prompt_template.template)
    allowed_tokens_for_data = max_prompt_length - prompt_token_length

    all_batches = batch_task_input_df(
        input_data, allowed_tokens_for_data, batch_size, partition_key
    )
    prompts = [build_prompt(prompt_template, batch, **kwargs) for batch in all_batches]

    return prompts


async def call_llm(
    batch_prompts: list[BatchPrompt],
    llm: Runnable,
    concurrency: int = 10,
    response_id_integrity_check: bool = False,
):
    """Process multiple batches of prompts concurrently through an LLM with retry logic.

    Args:
        batch_prompts (list[BatchPrompt]): List of BatchPrompt objects, each containing a
            prompt string and associated response IDs to be processed.
        llm (Runnable): LangChain Runnable instance that will process the prompts.
        concurrency (int, optional): Maximum number of simultaneous LLM calls allowed.
            Defaults to 10.
        response_id_integrity_check (bool, optional): If True, verifies that all input
            response IDs are present in the LLM output. Failed batches are discarded and
            their IDs are returned for retry. Defaults to False.

    Returns:
        tuple[list[dict[str, Any]], set[str]]: A tuple containing:
            - list of successful LLM responses as dictionaries
            - set of failed response IDs (empty if no failures or integrity check is False)

    Notes:
        - Uses exponential backoff retry strategy with up to 6 attempts per batch
        - Failed batches (when integrity check fails) return None and are filtered out
        - Concurrency is managed via asyncio.Semaphore to prevent overwhelming the LLM
    """
    semaphore = asyncio.Semaphore(concurrency)

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        reraise=True,
    )
    async def async_llm_call(batch_prompt):
        async with semaphore:
            response = await llm.ainvoke(batch_prompt.prompt_string)
            parsed_response = json.loads(response.content)
            failed_ids: set = set()

            if response_id_integrity_check and not check_response_integrity(
                batch_prompt.response_ids, parsed_response
            ):
                # discard this response but keep track of failed response ids
                failed_ids.update(batch_prompt.response_ids)
                return {"response": None, "failed_ids": failed_ids}

            return {"response": parsed_response, "failed_ids": failed_ids}

    results = await asyncio.gather(
        *[async_llm_call(batch_prompt) for batch_prompt in batch_prompts]
    )

    # Extract responses
    successful_responses = [
        r["response"] for r in results if r["response"] is not None
    ]  # ignore discarded responses

    # Extract failed ids
    failed_ids: set = set()
    for r in results:
        if r["response"] is None:
            failed_ids.update(r["failed_ids"])
    return (successful_responses, failed_ids)


def check_response_integrity(
    input_response_ids: set[str], parsed_response: dict
) -> bool:
    """Verify that all input response IDs are present in the LLM's parsed response.

    Args:
        input_response_ids (set[str]): Set of response IDs that were included in the
            original prompt sent to the LLM.
        parsed_response (dict): Parsed response from the LLM containing a 'responses' key
            with a list of dictionaries, each containing a 'response_id' field.

    Returns:
        bool: True if all input response IDs are present in the parsed response and
            no additional IDs are present, False otherwise.
    """
    response_ids_set = set(input_response_ids)

    returned_ids_set = {
        str(
            element["response_id"]
        )  # treat ids as strings to match response_ids_in_each_prompt
        for element in parsed_response["responses"]
        if element.get("response_id", False)
    }
    # assumes: all input ids ought to be present in output
    if returned_ids_set != response_ids_set:
        logger.info("Failed integrity check")
        logger.info(
            f"Present in original but not returned from LLM: {response_ids_set - returned_ids_set}. Returned in LLM but not present in original: {returned_ids_set - response_ids_set}"
        )
        return False
    return True


def process_llm_responses(
    llm_responses: list[dict[str, Any]], responses: pd.DataFrame
) -> pd.DataFrame:
    """Process and merge LLM responses with the original DataFrame.

    Args:
        llm_responses (list[dict[str, Any]]): List of LLM response dictionaries, where each
            dictionary contains a 'responses' key with a list of individual response objects.
        responses (pd.DataFrame): Original DataFrame containing the input responses, must
            include a 'response_id' column.

    Returns:
        pd.DataFrame: A merged DataFrame containing:
            - If response_id exists in LLM output: Original responses joined with LLM results
              on response_id (inner join)
            - If no response_id in LLM output: DataFrame containing only the LLM results
    """
    responses.loc[:, "response_id"] = responses["response_id"].astype(int)
    unpacked_responses = [
        response
        for batch_response in llm_responses
        for response in batch_response.get("responses", [])
    ]
    task_responses = pd.DataFrame(unpacked_responses)
    if "response_id" in task_responses.columns:
        task_responses["response_id"] = task_responses["response_id"].astype(int)
        return responses.merge(task_responses, how="inner", on="response_id")
    return task_responses


def calculate_string_token_length(input_text: str, model: str = None) -> int:
    # Use the MODEL_NAME env var if no model is provided; otherwise default to "gpt-4o"
    model = model or os.environ.get("MODEL_NAME", "gpt-4o")
    tokenizer_encoding = tiktoken.encoding_for_model(model)
    number_of_tokens = len(tokenizer_encoding.encode(input_text))
    return number_of_tokens


def build_prompt(
    prompt_template: PromptTemplate, input_batch: pd.DataFrame, **kwargs
) -> BatchPrompt:
    """
    Constructs a BatchPrompt by formatting a prompt template with a batch of responses.

    The function converts the input DataFrame batch into a list of dictionaries (one per row) and passes
    this list to the prompt template's format method under the key 'responses', along with any additional
    keyword arguments. It also extracts the 'response_id' column from the batch, converts the IDs to strings,
    and uses these to create the BatchPrompt.

    Args:
        prompt_template (PromptTemplate): An object with a 'template' attribute and a 'format' method that is used
            to generate the prompt string.
        input_batch (pd.DataFrame): A DataFrame containing the batch of responses, which must include a 'response_id'
            column.
        **kwargs: Additional keyword arguments to pass to the prompt template's format method.

    Returns:
        BatchPrompt: An object containing:
            - prompt_string: The formatted prompt string for the batch.
            - response_ids: A list of response IDs (as strings) corresponding to the responses in the batch.
    """
    prompt = prompt_template.format(
        responses=input_batch.to_dict(orient="records"), **kwargs
    )
    response_ids = input_batch["response_id"].astype(str).to_list()

    return BatchPrompt(prompt_string=prompt, response_ids=response_ids)


def validate_task_data(
    task_data: pd.DataFrame | list[dict], task_data_model: Type[BaseModel]
) -> tuple[list[dict], list[dict]]:
    """
    Validate each row in task_output against the provided Pydantic model.

    Returns:
        valid: a list of validated recors(dicts).
        invalid: a list of records (dicts) that failed validation.
    """

    records = (
        task_data.to_dict(orient="records")
        if isinstance(task_data, pd.DataFrame)
        else task_data
    )

    valid, invalid = [], []
    for record in records:
        try:
            task_data_model(**record)
            valid.append(record)
        except ValidationError:
            invalid.append(record)
            print(record)
    return valid, invalid
