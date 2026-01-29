"""Langfuse integration utilities for ThemeFinder evaluations.

Provides graceful fallback when Langfuse is not configured.
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger("themefinder.evals.langfuse")


@dataclass
class LangfuseContext:
    """Container for Langfuse client and callback handler."""

    client: "Langfuse | None"
    handler: "CallbackHandler | None"
    session_id: str | None = None

    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse is configured and available."""
        return self.client is not None


def get_langfuse_context(
    session_id: str,
    metadata: dict | None = None,
) -> LangfuseContext:
    """Initialise Langfuse with graceful fallback if not configured.

    Args:
        session_id: Unique identifier for grouping traces
            (e.g., "eval_generation_20260129_120000")
        metadata: Optional metadata dict to attach to all traces

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
        from langfuse import Langfuse, propagate_attributes
        from langfuse.langchain import CallbackHandler

        client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url,
        )

        # Set session and metadata via propagate_attributes for SDK v3
        propagate_attributes(
            session_id=session_id,
            metadata=metadata or {},
        )

        handler = CallbackHandler()

        logger.info(f"Langfuse initialised with session_id={session_id}")
        return LangfuseContext(client=client, handler=handler, session_id=session_id)

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
