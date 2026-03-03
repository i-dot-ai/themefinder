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
    """OpenAI SDK implementation of the LLM protocol."""

    def __init__(
        self,
        model,
        request_kwargs: dict | None = None,
        **client_kwargs,
    ):
        self.model = model
        self.request_kwargs = request_kwargs or {}
        self.client = openai.AsyncOpenAI(**client_kwargs)

    async def ainvoke(
        self, prompt: str, output_model: type[BaseModel] | None = None
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.request_kwargs,
        }
        if output_model:
            kwargs["response_format"] = output_model
            response = await self.client.chat.completions.parse(**kwargs)
            return LLMResponse(parsed=response.choices[0].message.parsed)
        else:
            response = await self.client.chat.completions.create(**kwargs)
            return LLMResponse(parsed=response.choices[0].message.content)

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
