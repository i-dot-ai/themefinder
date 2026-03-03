"""Utility for provider-aware structured output configuration.

Different LLM providers handle structured output differently:
- OpenAI/Azure: Native JSON Schema (works reliably)
- Gemini: Function calling by default, but json_schema is more reliable
- Claude: Tool calling (works reliably)

This module provides a utility function that auto-detects the provider
and applies the appropriate method for structured output.
"""

from langchain_core.runnables import Runnable
from pydantic import BaseModel


def with_structured_output(llm: Runnable, schema: type[BaseModel]) -> Runnable:
    """Get LLM with structured output using provider-appropriate method.

    For Gemini models, uses 'json_schema' method (controlled generation)
    which is more reliable than function calling for structured output.

    Args:
        llm: The LangChain LLM instance
        schema: Pydantic model defining the output schema

    Returns:
        LLM configured for structured output
    """
    llm_class_name = type(llm).__name__

    # Gemini models work better with json_schema method (controlled generation)
    # Detect both deprecated ChatVertexAI and new ChatGoogleGenerativeAI
    is_gemini = (
        "VertexAI" in llm_class_name and "Anthropic" not in llm_class_name
    ) or "GoogleGenerativeAI" in llm_class_name

    if is_gemini:
        return llm.with_structured_output(schema, method="json_schema")

    # Default: use standard method (works for OpenAI, Claude, etc.)
    return llm.with_structured_output(schema)
