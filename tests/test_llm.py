"""Tests for the OpenAI LLM wrapper's API selection and usage tracking."""

from types import SimpleNamespace

from pydantic import BaseModel

from themefinder.llm import OpenAILLM


class ExampleOutput(BaseModel):
    value: str


class FakeResponsesAPI:
    def __init__(self):
        self.parse_calls = []
        self.create_calls = []

    async def parse(self, text_format=None, **kwargs):
        self.parse_calls.append((text_format, kwargs))
        return SimpleNamespace(
            output_parsed=ExampleOutput(value="parsed"),
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(
            output_text="plain text",
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        )


class FakeChatCompletionsAPI:
    async def parse(self, **kwargs):
        message = SimpleNamespace(parsed=ExampleOutput(value="parsed"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4),
        )


def make_llm(**kwargs) -> OpenAILLM:
    return OpenAILLM(model="test-model", api_key="test", **kwargs)


async def test_responses_api_returns_structured_output_and_records_usage():
    llm = make_llm(use_responses_api=True)
    fake = FakeResponsesAPI()
    llm.client = SimpleNamespace(responses=fake)

    result = await llm.ainvoke("prompt", output_model=ExampleOutput)

    assert result.parsed == ExampleOutput(value="parsed")
    text_format, kwargs = fake.parse_calls[0]
    assert text_format is ExampleOutput
    assert kwargs == {"model": "test-model", "input": "prompt"}
    assert llm.usage.input_tokens == 5
    assert llm.usage.output_tokens == 3
    assert llm.usage.requests == 1


async def test_responses_api_returns_plain_text_without_output_model():
    llm = make_llm(use_responses_api=True)
    llm.client = SimpleNamespace(responses=FakeResponsesAPI())

    result = await llm.ainvoke("prompt")

    assert result.parsed == "plain text"


async def test_chat_api_records_usage_with_prompt_token_naming():
    llm = make_llm()
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletionsAPI())
    )

    result = await llm.ainvoke("prompt", output_model=ExampleOutput)

    assert result.parsed == ExampleOutput(value="parsed")
    assert llm.usage.input_tokens == 7
    assert llm.usage.output_tokens == 4
