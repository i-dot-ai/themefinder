#!/usr/bin/env python
"""One-time migration of existing eval data to Langfuse datasets.

This script migrates all existing evaluation data to Langfuse datasets,
enabling the use of run_experiment() for systematic evaluation.

Usage:
    # Migrate all datasets for gambling_XS
    uv run python scripts/migrate_to_langfuse.py

    # Migrate specific dataset
    uv run python scripts/migrate_to_langfuse.py --dataset gambling_XS

    # Migrate specific stage only
    uv run python scripts/migrate_to_langfuse.py --stage generation
"""

import argparse
import os
import sys
from pathlib import Path

import dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import DatasetConfig, get_or_create_dataset, load_local_data


def get_langfuse_client():
    """Get Langfuse client, ensuring environment is configured."""
    dotenv.load_dotenv()

    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL")

    if not all([secret_key, public_key, base_url]):
        raise RuntimeError(
            "Langfuse not configured. Set LANGFUSE_SECRET_KEY, "
            "LANGFUSE_PUBLIC_KEY, and LANGFUSE_BASE_URL environment variables."
        )

    from langfuse import Langfuse

    return Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=base_url,
    )


def migrate_stage(client, dataset: str, stage: str) -> None:
    """Migrate a single eval stage to Langfuse.

    Args:
        client: Langfuse client
        dataset: Dataset identifier (e.g., "gambling_XS")
        stage: Eval stage (generation, mapping, condensation, refinement)
    """
    config = DatasetConfig(dataset=dataset, stage=stage)
    print(f"Migrating {config.name}...")

    # Load data using the unified loader
    try:
        items = load_local_data(config)
    except FileNotFoundError as e:
        print(f"  Skipping {config.name}: {e}")
        return

    # Create dataset in Langfuse
    get_or_create_dataset(client, config)

    # Create items
    for item in items:
        client.create_dataset_item(
            dataset_name=config.name,
            input=item["input"],
            expected_output=item.get("expected_output"),
            metadata={
                "dataset": dataset,
                "stage": stage,
                **item.get("metadata", {}),
            },
        )

    print(f"  Created dataset: {config.name} ({len(items)} items)")


def migrate_all(dataset: str = "gambling_XS", stage: str | None = None) -> None:
    """Migrate all eval stages for a dataset.

    Args:
        dataset: Dataset identifier (e.g., "gambling_XS")
        stage: Optional specific stage to migrate (None = all)
    """
    client = get_langfuse_client()

    stages = ["generation", "mapping", "condensation", "refinement"]

    if stage:
        if stage not in stages:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {stages}")
        migrate_stage(client, dataset, stage)
    else:
        for s in stages:
            try:
                migrate_stage(client, dataset, s)
            except Exception as e:
                print(f"  Error migrating {s}: {e}")

    # Flush to ensure all data is sent
    client.flush()
    print("\nMigration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate eval data to Langfuse datasets"
    )
    parser.add_argument(
        "--dataset",
        default="gambling_XS",
        help="Dataset identifier (e.g., gambling_XS)",
    )
    parser.add_argument(
        "--stage", default=None, help="Specific stage to migrate (optional)"
    )
    args = parser.parse_args()

    migrate_all(dataset=args.dataset, stage=args.stage)
