from unittest.mock import MagicMock

import httpx
import openai
import pandas as pd
import pytest
import tiktoken

from themefinder import sentiment_analysis
from themefinder.llm_batch_processor import (
    BatchPrompt,
    batch_task_input_df,
    build_prompt,
    calculate_string_token_length,
    call_llm,
    generate_prompts,
    process_llm_responses,
    split_overflowing_batch,
    get_missing_response_ids,
    get_duplicate_response_ids,
)


async def test_process_llm_responses_with_clashing_types():
    """
    Test that process_llm_responses handles type mismatches between response IDs.
    Verifies that string response IDs from LLM responses are correctly matched
    with integer response IDs in the original DataFrame.
    """
    responses = pd.DataFrame({"response_id": [1], "text": ["response1"]})
    processed = process_llm_responses(
        # llm gives us a str response_id but the original response_id is an int
        [{"response_id": 1, "llm_contribution": "banana"}],
        responses,
    )
    assert list(processed["response_id"]) == [1]
    assert list(processed["llm_contribution"]) == ["banana"]


@pytest.mark.asyncio()
async def test_retries(mock_llm, sample_df):
    """
    Test the retry mechanism when LLM calls fail.
    Verifies that the system properly retries after an exception
    and successfully processes the responses on subsequent attempts.
    """
    exception = Exception("Rate limited!")
    mock_llm.ainvoke.side_effect = [
        exception,
        MagicMock(
            content='{"responses": [{"response_id": 1, "position": "AGREEMENT", "text": "response1"}, {"response_id": 2, "position": "DISAGREEMENT", "text": "response2"}]}'
        ),
    ]
    result, _ = await sentiment_analysis(sample_df, mock_llm, question="doesn't matter")
    # we got something back
    assert isinstance(result, pd.DataFrame)
    # we hit the llm twice
    assert mock_llm.ainvoke.call_count == 2


def test_empty_string(monkeypatch):
    fake_encoding = MagicMock()
    fake_encoding.encode.return_value = []
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: fake_encoding)

    token_length = calculate_string_token_length("", model="test-model")
    assert token_length == 0


def test_non_empty_string(monkeypatch):
    fake_encoding = MagicMock()
    fake_encoding.encode.side_effect = lambda text: text.split()
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: fake_encoding)

    token_length = calculate_string_token_length("hello world", model="test-model")

    assert token_length == 2


def test_calls_encoding_for_model(monkeypatch):
    fake_encoding = MagicMock()
    fake_encoding.encode.return_value = ["token1", "token2", "token3"]
    fake_encoding_for_model = MagicMock(return_value=fake_encoding)
    monkeypatch.setattr(tiktoken, "encoding_for_model", fake_encoding_for_model)

    token_length = calculate_string_token_length("any text", model="custom-model")
    fake_encoding_for_model.assert_called_once_with("custom-model")

    assert token_length == 3


class DummyPromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        # For testing, simply return a string that includes the number of responses and any extra info.
        responses = kwargs.get("responses", [])
        extra = kwargs.get("extra", "")
        return f"Prompt with {len(responses)} responses. {extra}"


def test_build_prompt():
    data = {"response_id": [1, 2], "text": ["response1", "response2"]}
    df = pd.DataFrame(data)
    prompt_template = DummyPromptTemplate("dummy template")

    extra_info = "Extra context"
    result = build_prompt(prompt_template, df, extra=extra_info)

    expected_prompt = (
        f"Prompt with {len(df.to_dict(orient='records'))} responses. {extra_info}"
    )
    assert result.prompt_string == expected_prompt
    assert result.response_ids == [1, 2]


def test_build_prompt_with_mock_template():
    data = {"response_id": [101, 202], "text": ["foo", "bar"]}
    df = pd.DataFrame(data)

    prompt_template = MagicMock()
    prompt_template.format.return_value = "formatted prompt"

    result = build_prompt(prompt_template, df)

    prompt_template.format.assert_called_once_with(
        responses=df.to_dict(orient="records")
    )

    assert result.prompt_string == "formatted prompt"
    assert result.response_ids == [101, 202]


def dummy_token_length_low(input_text: str, model="gpt-4o"):
    """
    Simulate a tokenizer that always returns a low token count.
    This forces the branch where the sample batch is small enough,
    so we simply split by batch size.
    """
    return 10


def dummy_token_length_iterative(input_text: str, model="gpt-4o"):
    """
    Simulate a tokenizer that returns a high token count for multi-row batches.
    We detect multiple rows by the presence of the sequence "},{"
    in the JSON string. For single-row JSON (which normally lacks "},{")
    return a low token count.
    """
    if "},{" in input_text:
        return 100  # High count to force iterative row-by-row splitting.
    return 10


def test_batch_task_input_df_row_based(monkeypatch):
    """
    When token counts are low, the function should simply split the DataFrame
    into batches based on the row count.
    """
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_low,
    )
    df = pd.DataFrame(
        {"response_id": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]}
    )
    batch_size = 2
    allowed_tokens = 100  # High enough so that each candidate batch is within limit.

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key=None)

    # Expect 3 batches: [rows 1,2], [rows 3,4], and [row 5].
    assert len(batches) == 3
    assert batches[0]["response_id"].tolist() == [1, 2]
    assert batches[1]["response_id"].tolist() == [3, 4]
    assert batches[2]["response_id"].tolist() == [5]


def test_batch_task_input_df_iterative(monkeypatch):
    """
    When a candidate batch's token count exceeds allowed_tokens, the function
    should split it into sub-batches by accumulating rows iteratively.
    """
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_iterative,
    )
    df = pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})
    batch_size = 2
    allowed_tokens = (
        50  # Force iterative splitting since a 2-row batch will appear too long.
    )

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key=None)

    # Expect two batches: first with rows [1,2] and second with row [3].
    assert len(batches) == 2
    assert batches[0]["response_id"].tolist() == [1, 2]
    assert batches[1]["response_id"].tolist() == [3]


def test_batch_task_input_df_with_partition(monkeypatch):
    """
    When a partition_key is provided, the DataFrame should be split into partitions,
    and each partition processed independently.
    """
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_low,
    )
    df = pd.DataFrame(
        {
            "response_id": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
            "group": ["x", "x", "y", "y"],
        }
    )
    batch_size = 2
    allowed_tokens = 100

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key="group")

    # Expect two batches, one per partition.
    assert len(batches) == 2
    # Verify that each batch contains rows from only one partition.
    for batch in batches:
        assert batch["group"].nunique() == 1


def test_batch_task_input_df_all_invalid(monkeypatch):
    """
    If every row in the DataFrame is individually invalid (token count > allowed_tokens),
    the function should return an empty list.
    """

    # Dummy function: return 60 for any input so that every row is considered invalid
    def dummy_token_length_all_exceed(text: str) -> int:
        return 60

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_all_exceed,
    )

    df = pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})
    batch_size = 2
    allowed_tokens = 50  # Every row returns 60 tokens, so each is individually invalid.

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key=None)

    # With all rows invalid, we expect an empty list.
    assert batches == []


def test_batch_task_input_df_else_branch(monkeypatch):
    """
    Test the else branch in the for loop, where rows accumulate normally within token and batch constraints.
    """
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_low,
    )

    # Create a DataFrame with 4 rows.
    df = pd.DataFrame({"response_id": [1, 2, 3, 4], "text": ["a", "b", "c", "d"]})

    batch_size = 3  # Allows up to 3 rows in a batch
    allowed_tokens = 40  # Enough for 4 rows but will split before that

    # Expected behavior:
    # - The first batch accumulates until 3 rows.
    # - Since allowed_tokens is not exceeded, the else block executes for rows 2 and 3.
    # - The last row (4) starts a new batch.

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key=None)

    # Ensure we get two batches
    assert len(batches) == 2
    assert batches[0]["response_id"].tolist() == [
        1,
        2,
        3,
    ]  # Else branch accumulates rows 2 & 3
    assert batches[1]["response_id"].tolist() == [4]  # New batch starts here


# Define a dummy calculate_string_token_length that returns a fixed value.
def dummy_calculate_string_token_length(input_text: str, model="gpt-4o") -> int:
    # For our tests, we simulate that the prompt template always uses 50 tokens.
    return 50


# Define a dummy batch_task_input_df that returns a predetermined list of DataFrame batches.
def dummy_batch_task_input_df(
    df: pd.DataFrame, allowed_tokens: int, batch_size: int, partition_key: str | None
):
    # For simplicity, ignore the inputs and return two fixed batches.
    batch1 = pd.DataFrame({"response_id": [1, 2], "text": ["a", "b"]})
    batch2 = pd.DataFrame({"response_id": [3], "text": ["c"]})
    return [batch1, batch2]


@pytest.fixture
def dummy_input_data():
    # Provide some dummy input data (won't matter because we override batch_task_input_df).
    return pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})


def test_split_overflowing_batch_within_limit(monkeypatch):
    # Every row returns 10 tokens, so the total is below allowed_tokens.
    def dummy_token_length(s: str) -> int:
        return 10

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length,
    )

    # Create a DataFrame with 3 rows.
    df = pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})

    # With allowed_tokens high enough, the entire batch should be valid.
    sub_batches = split_overflowing_batch(df, allowed_tokens=50)

    # Expect one sub-batch containing all rows.
    assert len(sub_batches) == 1
    assert sub_batches[0]["response_id"].tolist() == [1, 2, 3]


def test_split_overflowing_batch_requires_splitting(monkeypatch):
    # Define token lengths per row via the 'text' field:
    # Row with "a": 30 tokens, "b": 30 tokens, "c": 20 tokens.
    def dummy_token_length(s: str) -> int:
        if '"text":"a"' in s:
            return 30
        elif '"text":"b"' in s:
            return 30
        elif '"text":"c"' in s:
            return 20
        return 10

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length,
    )

    df = pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})

    # With allowed_tokens=50, the total token count (30+30+20=80) exceeds the limit.
    # Expected behavior:
    #   - The first row (30 tokens) forms the first sub-batch.
    #   - The next two rows are combined (30+20=50 tokens) in a second sub-batch.
    sub_batches = split_overflowing_batch(df, allowed_tokens=50)

    assert len(sub_batches) == 2
    assert sub_batches[0]["response_id"].tolist() == [1]
    assert sub_batches[1]["response_id"].tolist() == [2, 3]


def test_split_overflowing_batch_skip_invalid(monkeypatch):
    # Define token lengths:
    # "a": 30 tokens, "b": 10 tokens, "c": 20 tokens, "d": 30 tokens.
    def dummy_token_length(s: str) -> int:
        if '"text":"a"' in s:
            return 30
        elif '"text":"b"' in s:
            return 10
        elif '"text":"c"' in s:
            return 20
        elif '"text":"d"' in s:
            return 30
        return 10

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length,
    )

    # Create a DataFrame with four rows.
    df = pd.DataFrame({"response_id": [1, 2, 3, 4], "text": ["a", "b", "c", "d"]})

    # With allowed_tokens=25, rows "a" and "d" (30 tokens each) are individually invalid.
    # For rows "b" and "c":
    #   - Row "b" (10 tokens) becomes one sub-batch.
    #   - Row "c" (20 tokens) cannot join "b" (10+20=30>25) and becomes its own sub-batch.
    sub_batches = split_overflowing_batch(df, allowed_tokens=25)

    assert len(sub_batches) == 2
    # The valid rows are row 2 and row 3 (response_ids 2 and 3).
    assert sub_batches[0]["response_id"].tolist() == [2]
    assert sub_batches[1]["response_id"].tolist() == [3]


def test_split_overflowing_batch_all_invalid(monkeypatch):
    # All rows return 20 tokens, but allowed_tokens is set too low.
    def dummy_token_length(s: str) -> int:
        return 20

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length,
    )

    df = pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})

    # With allowed_tokens=10, every row exceeds the limit.
    sub_batches = split_overflowing_batch(df, allowed_tokens=10)

    # Expect an empty list because all rows are skipped.
    assert sub_batches == []


def test_split_overflowing_batch_edge_case(monkeypatch):
    # Test the edge condition where the sum exactly equals allowed_tokens.
    def dummy_token_length(s: str) -> int:
        if '"text": "a"' in s:
            return 30
        elif '"text": "b"' in s:
            return 20
        return 10

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length,
    )

    df = pd.DataFrame({"response_id": [1, 2], "text": ["a", "b"]})

    # With allowed_tokens=50, the token count is exactly 30+20=50.
    sub_batches = split_overflowing_batch(df, allowed_tokens=50)

    # Expect one sub-batch containing both rows.
    assert len(sub_batches) == 1
    assert sub_batches[0]["response_id"].tolist() == [1, 2]


def test_generate_prompts(monkeypatch, dummy_input_data):
    """
    Test generate_prompts when batch_task_input_df returns two predetermined batches.
    """
    # Monkey-patch dependencies:
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_calculate_string_token_length,
    )
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.batch_task_input_df",
        dummy_batch_task_input_df,
    )

    # Create a dummy prompt template.
    prompt_template = DummyPromptTemplate("dummy template")

    # Call generate_prompts with an extra keyword argument.
    prompts = generate_prompts(
        prompt_template,
        dummy_input_data,
        batch_size=50,
        max_prompt_length=50000,
        partition_key=None,
        extra="foo",
    )

    # We expect two BatchPrompt objects since dummy_batch_task_input_df returns two batches.
    assert len(prompts) == 2

    # For batch1, which has 2 rows, the dummy prompt template returns:
    # "Formatted: 2 responses. foo"
    assert prompts[0].prompt_string == "Prompt with 2 responses. foo"
    # And response IDs should be converted to strings.
    assert prompts[0].response_ids == [1, 2]

    # For batch2, which has 1 row:
    assert prompts[1].prompt_string == "Prompt with 1 responses. foo"
    assert prompts[1].response_ids == [3]


def test_generate_prompts_with_partition(monkeypatch):
    """
    Test generate_prompts when partitioning is applied.
    We'll simulate partitioning by defining a dummy batch_task_input_df that splits the DataFrame
    based on a partition key.
    """

    def dummy_partition_batch_task_input_df(
        df: pd.DataFrame,
        allowed_tokens: int,
        batch_size: int,
        partition_key: str | None,
    ):
        # Partition the DataFrame by the given key and return each partition as one batch.
        if partition_key:
            partitions = [
                group.reset_index(drop=True) for _, group in df.groupby(partition_key)
            ]
            return partitions
        return [df]

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_calculate_string_token_length,
    )
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.batch_task_input_df",
        dummy_partition_batch_task_input_df,
    )

    prompt_template = DummyPromptTemplate("dummy template")
    # Create input data that includes a partition key 'group'.
    df = pd.DataFrame(
        {
            "response_id": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
            "group": ["A", "A", "B", "B"],
        }
    )

    prompts = generate_prompts(
        prompt_template,
        df,
        batch_size=50,
        max_prompt_length=50000,
        partition_key="group",
        extra="bar",
    )

    # We expect two batches—one for each unique group.
    assert len(prompts) == 2

    # Since groupby order is not guaranteed, collect the response IDs from both prompts.
    response_ids = {tuple(prompt.response_ids) for prompt in prompts}
    expected = {(1, 2), (3, 4)}
    assert response_ids == expected

    # Also verify that the prompt string includes the correct number of responses.
    for prompt in prompts:
        if len(prompt.response_ids) == 2:
            assert prompt.prompt_string == "Prompt with 2 responses. bar"


@pytest.mark.asyncio
async def test_call_llm_bad_request(monkeypatch, mock_llm):
    batch_prompts = [BatchPrompt(prompt_string="dummy prompt", response_ids=[1, 2])]

    class DummyBadRequestError(openai.BadRequestError):
        def __init__(self, *args, **kwargs):
            dummy_request = httpx.Request("GET", "http://dummy.url")
            dummy_response = httpx.Response(
                status_code=400,
                content=b'{"error": "dummy"}',
                headers={"Content-Type": "application/json"},
                request=dummy_request,
            )
            super().__init__(
                "dummy message", response=dummy_response, body="dummy body"
            )

    mock_llm.ainvoke.side_effect = DummyBadRequestError()
    results, failed_ids = await call_llm(batch_prompts, mock_llm)

    # Verify that each response id in the result contains the expected failure details.
    for response_id in batch_prompts[0].response_ids:
        assert response_id in failed_ids


def test_get_missing_response_ids_none_missing():
    """All IDs are present → empty list."""
    input_ids = [1, 2, 3]
    parsed = {"responses": [{"response_id": 1}, {"response_id": 2}, {"response_id": 3}]}

    assert get_missing_response_ids(input_ids, parsed) == []


def test_get_missing_response_ids_some_missing():
    """Function returns *only* the absent IDs (order not guaranteed)."""
    input_ids = [1, 2, 3, 4]
    parsed = {"responses": [{"response_id": 1}, {"response_id": 3}]}

    missing = get_missing_response_ids(input_ids, parsed)
    assert set(missing) == {2, 4}


def test_get_missing_response_ids_all_missing():
    """Handles a parsed response with no IDs at all."""
    input_ids = [10, 20]
    parsed = {"responses": []}

    assert set(get_missing_response_ids(input_ids, parsed)) == {10, 20}


def test_get_duplicate_response_ids_no_duplicates():
    """Distinct IDs → empty list."""
    parsed = {"responses": [{"response_id": 5}, {"response_id": 6}, {"response_id": 7}]}

    assert get_duplicate_response_ids(parsed) == []


def test_get_duplicate_response_ids_with_duplicates():
    """
    Two extra appearances of ID=42:
    - after first repeat it is appended once;
    - after second repeat it is appended again.
    """
    parsed = {
        "responses": [
            {"response_id": 42},
            {"response_id": 42},  # 1st duplicate
            {"response_id": 99},
            {"response_id": 42},  # 2nd duplicate
        ]
    }

    dupes = get_duplicate_response_ids(parsed)

    # exactly two entries, each == 42, in the order they occurred
    assert dupes == [42]
