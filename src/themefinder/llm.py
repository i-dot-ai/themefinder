"""LLM abstraction layer for themefinder.

Provides a Protocol-based interface for LLM calls with structured output support,
and an OpenAI implementation. Designed for easy extension to other providers.
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import openai
from pydantic import BaseModel


@dataclass
class LLMResponse:
    """Wraps an LLM call result."""

    parsed: BaseModel | str


@dataclass
class Usage:
    """Accumulated token usage across API calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def record(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Add one request's token counts."""
        self.requests += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0


# Backwards-compatible name used by the LLM client.
LLMUsage = Usage


@runtime_checkable
class LLM(Protocol):
    """Protocol defining the LLM interface for themefinder."""

    async def ainvoke(
        self, prompt: str, output_model: type[BaseModel] | None = None
    ) -> LLMResponse: ...

    def invoke(
        self, prompt: str, output_model: type[BaseModel] | None = None
    ) -> LLMResponse: ...


class OpenAILLM:
    """OpenAI SDK implementation of the LLM protocol.

    Uses the Chat Completions API by default; set ``use_responses_api=True``
    for models that are only served through the Responses API.
    """

    def __init__(
        self,
        model,
        request_kwargs: dict | None = None,
        use_responses_api: bool = False,
        **client_kwargs,
    ):
        self.model = model
        self.request_kwargs = request_kwargs or {}
        self.use_responses_api = use_responses_api
        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.usage = LLMUsage()

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self.usage.record(0, 0)
            return
        # Chat Completions reports prompt/completion tokens; Responses
        # reports input/output tokens.
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0)
        self.usage.record(input_tokens, output_tokens)

    async def ainvoke(
        self, prompt: str, output_model: type[BaseModel] | None = None
    ) -> LLMResponse:
        if self.use_responses_api:
            return await self._ainvoke_responses(prompt, output_model)
        return await self._ainvoke_chat(prompt, output_model)

    async def _ainvoke_chat(
        self, prompt: str, output_model: type[BaseModel] | None
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.request_kwargs,
        }
        if output_model:
            kwargs["response_format"] = output_model
            response = await self.client.chat.completions.parse(**kwargs)
            self._record_usage(response)
            return LLMResponse(parsed=response.choices[0].message.parsed)
        else:
            response = await self.client.chat.completions.create(**kwargs)
            self._record_usage(response)
            return LLMResponse(parsed=response.choices[0].message.content)

    async def _ainvoke_responses(
        self, prompt: str, output_model: type[BaseModel] | None
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "input": prompt,
            **self.request_kwargs,
        }
        if output_model:
            response = await self.client.responses.parse(
                text_format=output_model, **kwargs
            )
            self._record_usage(response)
            return LLMResponse(parsed=response.output_parsed)
        else:
            response = await self.client.responses.create(**kwargs)
            self._record_usage(response)
            return LLMResponse(parsed=response.output_text)

    def invoke(
        self, prompt: str, output_model: type[BaseModel] | None = None
    ) -> LLMResponse:
        """Synchronous wrapper around ainvoke."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run, self.ainvoke(prompt, output_model)
                ).result()
        return asyncio.run(self.ainvoke(prompt, output_model))
