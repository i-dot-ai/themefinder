"""Langfuse integration utilities for ThemeFinder evaluations.

Provides graceful fallback when Langfuse is not configured.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse._client.span import LangfuseSpan
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
def trace_context(
    context: LangfuseContext, name: str = "eval_task"
) -> Generator["LangfuseSpan | None", None, None]:
    """Create a parent Langfuse span that captures all nested LangChain traces.

    Uses native Langfuse span with update_trace() for trace-level attributes.
    Also updates the CallbackHandler's trace_context to ensure LangChain traces
    are properly nested under the parent span.

    Args:
        context: LangfuseContext from get_langfuse_context()
        name: Name for the parent span (e.g., "generation_eval", "question_1")

    Yields:
        LangfuseSpan instance (or None if Langfuse is disabled)

    Example:
        with trace_context(langfuse_ctx, name="sentiment_eval") as span:
            response = llm.invoke(prompt)  # Nested under span with tags/metadata
    """
    if not context.is_enabled or not context.client:
        yield None
        return

    # Store original handler trace_context to restore later
    original_trace_context = None
    if context.handler:
        original_trace_context = getattr(context.handler, "trace_context", None)

    try:
        with context.client.start_as_current_span(
            name=name, metadata=context.metadata
        ) as span:
            # Set session_id, tags on the parent trace
            span.update_trace(
                session_id=context.session_id,
                tags=context.tags,
                metadata=context.metadata,
            )

            # Update handler's trace_context to nest LangChain traces under this span
            if context.handler:
                context.handler.trace_context = {"trace_id": span.trace_id}
                logger.debug(f"Set handler trace_context to {span.trace_id}")

            yield span
    except ImportError:
        logger.warning("Langfuse package not available")
        yield None
    except Exception as e:
        logger.warning(f"Failed to create trace context: {e}")
        yield None
    finally:
        # Restore original handler trace_context
        if context.handler and original_trace_context is not None:
            context.handler.trace_context = original_trace_context


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

    Uses score_current_trace() when inside a trace_context() block,
    otherwise falls back to explicit trace_id attachment.

    Args:
        context: LangfuseContext from get_langfuse_context()
        scores: Dict mapping score names to numeric values
        trace_id: Optional trace_id (used only as fallback)
    """
    if not context.is_enabled or not context.client:
        return

    for name, value in scores.items():
        # Skip non-numeric values (e.g., tuples like confidence intervals)
        if not isinstance(value, (int, float)):
            logger.debug(f"Skipping non-numeric score {name}={value}")
            continue

        try:
            # Try to use score_current_trace() first (works inside trace_context)
            context.client.score_current_trace(
                name=name,
                value=float(value),
                data_type="NUMERIC",
            )
            logger.debug(f"Attached score {name}={value} to current trace")
        except Exception as e:
            # Fallback to explicit trace_id if available
            if trace_id is None and context.handler:
                trace_id = context.handler.last_trace_id

            if trace_id:
                try:
                    context.client.create_score(
                        name=name,
                        value=float(value),
                        trace_id=trace_id,
                        data_type="NUMERIC",
                    )
                    logger.debug(f"Attached score {name}={value} via trace_id")
                except Exception as e2:
                    logger.warning(f"Failed to attach score {name}: {e2}")
            else:
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


@contextmanager
def dataset_item_trace(
    context: LangfuseContext,
    dataset_item: Any,
    run_name: str,
) -> Generator[tuple[Any, str | None], None, None]:
    """Create a trace for a single dataset item linked to a dataset run.

    Uses Langfuse's item.run() context manager which automatically:
    - Creates a trace linked to the dataset item
    - Associates the trace with a named dataset run
    - Aggregates metadata and costs at the run level

    Args:
        context: LangfuseContext from get_langfuse_context()
        dataset_item: Langfuse dataset item object (must have .run() method)
        run_name: Name of the experiment run (typically session_id)

    Yields:
        Tuple of (span object, trace_id) for score attachment.
        Both are None if Langfuse is disabled.

    Example:
        for item in dataset.items:
            with dataset_item_trace(ctx, item, ctx.session_id) as (trace, trace_id):
                result = await run_task(item)
                if trace:
                    trace.update(output=result)
                if trace_id:
                    ctx.client.create_score(
                        trace_id=trace_id, name="accuracy", value=0.95, data_type="NUMERIC"
                    )
    """
    if not context.is_enabled or not context.client:
        yield None, None
        return

    # Store original handler trace_context to restore later
    original_trace_context = None
    if context.handler:
        original_trace_context = getattr(context.handler, "trace_context", None)

    try:
        # Use item.run() context manager for automatic dataset run linking
        # This creates a trace that is properly linked to the dataset run,
        # enabling metadata and cost aggregation at the run level
        with dataset_item.run(
            run_name=run_name,
            run_metadata=context.metadata,
        ) as root_span:
            # Update trace-level attributes (session_id, tags)
            root_span.update_trace(
                session_id=context.session_id,
                tags=context.tags,
            )

            # Update handler's trace_context to nest LangChain traces under this span
            if context.handler:
                context.handler.trace_context = {"trace_id": root_span.trace_id}
                logger.debug(f"Set handler trace_context to {root_span.trace_id}")

            yield root_span, root_span.trace_id

    except Exception as e:
        logger.warning(f"Failed to create dataset item trace: {e}")
        yield None, None
    finally:
        # Restore original handler trace_context
        if context.handler and original_trace_context is not None:
            context.handler.trace_context = original_trace_context
