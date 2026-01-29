"""Langfuse integration utilities for ThemeFinder evaluations.

Provides graceful fallback when Langfuse is not configured.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger("themefinder.evals.langfuse")


def _get_version() -> str:
    """Get package version from installed metadata.

    Returns:
        Version string (e.g., "0.7.8") or "unknown" if not found.
    """
    try:
        from importlib.metadata import version

        return version("themefinder")
    except Exception:
        return "unknown"


@dataclass
class LangfuseContext:
    """Container for Langfuse client and callback handler."""

    client: "Langfuse | None"
    handler: "CallbackHandler | None"
    session_id: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None

    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse is configured and available."""
        return self.client is not None


@contextmanager
def trace_context(context: LangfuseContext) -> Generator[None, None, None]:
    """Context manager to propagate Langfuse attributes to all nested LLM calls.

    Use this to wrap code sections where you want tags, metadata, and session_id
    to be attached to all Langfuse traces created within the block.

    Args:
        context: LangfuseContext from get_langfuse_context()

    Example:
        with trace_context(langfuse_ctx):
            response = llm.invoke(prompt)  # Trace will have tags/metadata
    """
    if not context.is_enabled:
        yield
        return

    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            session_id=context.session_id,
            tags=context.tags,
            metadata=context.metadata,
        ):
            yield
    except ImportError:
        logger.warning("Langfuse package not available for propagate_attributes")
        yield
    except Exception as e:
        logger.warning(f"Failed to set trace context: {e}")
        yield


def get_langfuse_context(
    session_id: str,
    eval_type: str,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> LangfuseContext:
    """Initialise Langfuse with structured tags and metadata.

    Args:
        session_id: Unique identifier for grouping traces
            (e.g., "eval_generation_20260129_120000")
        eval_type: Type of evaluation (e.g., "generation", "sentiment", "mapping")
        metadata: Optional additional metadata dict to merge with standard metadata
        tags: Optional additional tags to merge with standard tags

    Returns:
        LangfuseContext with client and handler (or None values if not configured)
    """
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL")

    if not all([secret_key, public_key, base_url]):
        logger.info("Langfuse not configured - tracing disabled")
        return LangfuseContext(client=None, handler=None)

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url,
        )

        # Build standard tags and metadata
        version = _get_version()
        environment = os.getenv("ENVIRONMENT", "development")
        git_sha = os.getenv("GITHUB_SHA", "local")[:7]
        model = os.getenv("DEPLOYMENT_NAME", "unknown")

        standard_tags = [
            "eval",
            eval_type,
            f"model:{model}",
            f"v{version}",
            environment,
        ]
        all_tags = standard_tags + (tags or [])

        standard_metadata = {
            "eval_type": eval_type,
            "model": model,
            "version": version,
            "git_sha": git_sha,
            "environment": environment,
        }
        all_metadata = {**standard_metadata, **(metadata or {})}

        handler = CallbackHandler()

        logger.info(
            f"Langfuse initialised: session_id={session_id}, "
            f"eval_type={eval_type}, version={version}, env={environment}"
        )
        return LangfuseContext(
            client=client,
            handler=handler,
            session_id=session_id,
            tags=all_tags,
            metadata=all_metadata,
        )

    except ImportError:
        logger.warning("Langfuse package not available")
        return LangfuseContext(client=None, handler=None)
    except Exception as e:
        logger.warning(f"Failed to initialise Langfuse: {e}")
        return LangfuseContext(client=None, handler=None)


def create_scores(
    context: LangfuseContext,
    scores: dict[str, float | int],
    trace_id: str | None = None,
) -> None:
    """Attach computed metrics as scores to the current trace.

    Args:
        context: LangfuseContext from get_langfuse_context()
        scores: Dict mapping score names to numeric values
        trace_id: Optional trace_id (uses handler's trace if not provided)
    """
    if not context.is_enabled:
        return

    if trace_id is None and context.handler:
        trace_id = context.handler.last_trace_id

    if not trace_id:
        logger.warning("No trace_id available for score attachment")
        return

    for name, value in scores.items():
        # Skip non-numeric values (e.g., tuples like confidence intervals)
        if not isinstance(value, (int, float)):
            logger.debug(f"Skipping non-numeric score {name}={value}")
            continue

        try:
            context.client.create_score(
                name=name,
                value=float(value),
                trace_id=trace_id,
                data_type="NUMERIC",
            )
            logger.debug(f"Attached score {name}={value}")
        except Exception as e:
            logger.warning(f"Failed to attach score {name}: {e}")


def flush(context: LangfuseContext) -> None:
    """Flush pending Langfuse data before exit.

    Args:
        context: LangfuseContext from get_langfuse_context()
    """
    if context.is_enabled and context.client:
        try:
            context.client.flush()
            logger.debug("Langfuse data flushed")
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")
