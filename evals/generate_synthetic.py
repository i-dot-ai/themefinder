#!/usr/bin/env python
"""Interactive CLI for generating synthetic consultation datasets.

Usage:
    cd evals && uv run python generate_synthetic.py

This tool guides you through creating a synthetic dataset for ThemeFinder evaluation,
including:
- Defining the consultation topic and questions
- Configuring number of responses
- Configuring demographic fields
- Generating themes using gpt-5-mini with reasoning
- Generating diverse responses with controlled characteristics
"""

import asyncio
import os
import sys
from contextlib import nullcontext

import dotenv
import openai

# Add parent to path for imports
sys.path.insert(0, str(os.path.dirname(__file__)))

import utils_gateway
from synthetic.cli import (
    create_progress_bar,
    print_error,
    print_success,
    run_interactive_cli,
)
from synthetic.config import RESPONSE_GENERATION_MODEL
from synthetic.generator import SyntheticDatasetGenerator

# Optional Langfuse integration
try:
    import langfuse_utils
    from langfuse.openai import AsyncOpenAI as _LangfuseOpenAI

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


async def main() -> None:
    """Main entry point for synthetic dataset generation."""
    dotenv.load_dotenv()

    # Collect configuration interactively
    try:
        config = await run_interactive_cli()
    except SystemExit as e:
        print(str(e))
        return

    # Initialise LLM for response generation (gpt-5-nano with medium reasoning)
    # Medium reasoning ≈ o1 performance, 2x faster throughput than mini/low
    _OpenAIClientClass = _LangfuseOpenAI if LANGFUSE_AVAILABLE else openai.AsyncOpenAI
    base_url, api_key = utils_gateway.gateway_credentials()
    client = _OpenAIClientClass(
        base_url=base_url,
        api_key=api_key,
        timeout=600,  # 10 minute timeout to prevent indefinite hangs (reasoning can be slow)
    )

    # Optional Langfuse tracking
    langfuse_ctx = None

    if LANGFUSE_AVAILABLE:
        langfuse_ctx = langfuse_utils.get_langfuse_context(
            session_id=f"synthetic_{config.dataset_name}",
            eval_type="synthetic_generation",
            metadata={
                "dataset": config.dataset_name,
                "n_responses": config.n_responses,
            },
            tags=[config.dataset_name, "synthetic"],
        )

    # Initialise generator
    generator = SyntheticDatasetGenerator(
        config=config,
        llm=(client, RESPONSE_GENERATION_MODEL),
    )

    # Calculate total responses for summary
    total_responses = config.n_responses * len(config.questions)

    # Run generation with progress tracking
    # Generator manages its own tasks: theme generation, then response generation
    progress = create_progress_bar()
    output_path = None
    n_themes = 0

    _trace = (
        langfuse_utils.trace_context(langfuse_ctx, name="synthetic_generation")
        if langfuse_ctx
        else nullcontext()
    )

    try:
        with _trace:
            with progress:
                output_path = await generator.generate(progress)

        # Count themes from first question for summary (after progress bar closes)
        themes_file = (
            output_path
            / "outputs"
            / "mapping"
            / generator.writer.date_str
            / "question_part_1"
            / "themes.json"
        )
        if themes_file.exists():
            import json

            with open(themes_file) as f:
                n_themes = len(json.load(f))

        print_success(str(output_path), n_themes, total_responses)

    except Exception as e:
        print_error(e)
        raise

    finally:
        if LANGFUSE_AVAILABLE and langfuse_ctx:
            langfuse_utils.flush(langfuse_ctx)


if __name__ == "__main__":
    asyncio.run(main())
