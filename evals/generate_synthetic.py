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

import dotenv
from langchain_openai import AzureChatOpenAI

# Add parent to path for imports
sys.path.insert(0, str(os.path.dirname(__file__)))

from synthetic.cli import (
    create_progress_bar,
    print_error,
    print_success,
    run_interactive_cli,
)
from synthetic.generator import SyntheticDatasetGenerator

# Optional Langfuse integration
try:
    import langfuse_utils

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

    # Initialise LLM for response generation (gpt-5-mini with low reasoning)
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
        reasoning_effort="low",
    )

    # Optional Langfuse tracking
    callbacks = []
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
        if langfuse_ctx.handler:
            callbacks.append(langfuse_ctx.handler)

    # Initialise generator
    generator = SyntheticDatasetGenerator(
        config=config,
        llm=llm,
        callbacks=callbacks,
    )

    # Calculate total responses for progress tracking
    total_responses = config.n_responses * len(config.questions)

    # Run generation with progress tracking
    progress = create_progress_bar()
    with progress:
        progress.add_task(
            f"[cyan]Generating {config.dataset_name}...",
            total=total_responses,
        )

        try:
            output_path = await generator.generate(progress)

            # Count themes from first question for summary
            n_themes = 0
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
