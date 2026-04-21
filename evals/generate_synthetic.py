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
import openai

# Add parent to path for imports
sys.path.insert(0, str(os.path.dirname(__file__)))

from synthetic.cli import (
    create_progress_bar,
    print_error,
    print_success,
    run_interactive_cli,
)
from synthetic.generator import SyntheticDatasetGenerator

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
    client = openai.AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
        timeout=600,  # 10 minute timeout to prevent indefinite hangs (reasoning can be slow)
    )

    # Initialise generator
    generator = SyntheticDatasetGenerator(
        config=config,
        llm=(client, "gpt-5-nano"),
    )

    # Calculate total responses for summary
    total_responses = config.n_responses * len(config.questions)

    # Run generation with progress tracking
    # Generator manages its own tasks: theme generation, then response generation
    progress = create_progress_bar()
    output_path = None
    n_themes = 0

    try:
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


if __name__ == "__main__":
    asyncio.run(main())
