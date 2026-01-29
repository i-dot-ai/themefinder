"""Core orchestrator for synthetic consultation dataset generation."""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from langchain_openai import AzureChatOpenAI
from rich.progress import Progress

from synthetic.config import (
    GenerationConfig,
    NoiseLevel,
    ResponseLength,
    ResponseSpec,
    ResponseType,
)
from synthetic.demographics import sample_demographics
from synthetic.llm_generators.response_generator import generate_response_batch
from synthetic.llm_generators.theme_generator import generate_themes
from synthetic.validators import validate_dataset
from synthetic.writers import DatasetWriter

logger = logging.getLogger(__name__)

BATCH_SIZE = 30  # Responses per LLM batch


class SyntheticDatasetGenerator:
    """Orchestrates synthetic consultation dataset generation."""

    def __init__(
        self,
        config: GenerationConfig,
        llm: AzureChatOpenAI,
        callbacks: list | None = None,
        seed: int = 42,
    ) -> None:
        """Initialise generator with configuration.

        Args:
            config: Generation configuration.
            llm: Azure OpenAI LLM for response generation.
            callbacks: LangChain callbacks for tracing.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.llm = llm
        self.callbacks = callbacks or []
        self.rng = np.random.default_rng(seed)
        self.writer = DatasetWriter(config.output_dir)

        self._checkpoint_path = config.output_dir / ".checkpoint.json"
        self._generated_count = 0

    async def generate(self, progress: Progress | None = None) -> Path:
        """Generate complete synthetic dataset.

        Args:
            progress: Optional Rich progress bar for tracking.

        Returns:
            Path to generated dataset directory.
        """
        # Create output directory structure
        self.writer.initialise_directories(self.config.questions)

        # Sample demographics once for all respondents
        demographics = sample_demographics(
            self.config.demographic_fields,
            self.config.n_responses,
            self.rng,
        )

        # Write respondents file with response IDs starting at 1001
        respondents = [
            {"response_id": 1001 + i, "demographic_data": demo}
            for i, demo in enumerate(demographics)
        ]
        self.writer.write_respondents(respondents)

        # Generate for each question
        for question_config in self.config.questions:
            question_part = f"question_part_{question_config.number}"
            logger.info(f"Generating {question_part}...")

            # Step 1: Generate themes (uses gpt-5-mini with reasoning)
            themes = await generate_themes(
                topic=self.config.topic,
                question=question_config.text,
                demographic_fields=self.config.demographic_fields,
                callbacks=self.callbacks,
            )
            logger.info(f"Generated {len(themes)} themes for {question_part}")

            self.writer.write_themes(question_part, themes)
            self.writer.write_question(question_part, question_config)

            # Step 2: Plan response distribution
            specs = self._plan_responses(themes, demographics)

            # Step 3: Initialise streaming files and generate responses in batches
            self.writer.init_streaming_files(question_part)
            task_id = progress.task_ids[0] if progress else None
            batch_num = 0

            def on_response_complete():
                """Callback for each completed response."""
                self._generated_count += 1
                if progress and task_id is not None:
                    progress.update(task_id, completed=self._generated_count)

            for batch_specs in self._batch(specs, BATCH_SIZE):
                batch_num += 1
                batch_responses = await generate_response_batch(
                    llm=self.llm,
                    question=question_config.text,
                    themes=themes,
                    specs=batch_specs,
                    noise_level=self.config.noise_level,
                    callbacks=self.callbacks,
                    on_response_complete=on_response_complete,
                )

                # Stream to disk immediately (reduces RAM, crash-safe)
                self.writer.append_responses(question_part, batch_responses)

                # Checkpoint with batch progress
                self._save_checkpoint(question_part, batch_num, len(specs))

                # Brief pause between batches for rate limiting
                await asyncio.sleep(0.2)

        # Validate generated dataset
        validation_result = validate_dataset(self.config.output_dir)
        if not validation_result.is_valid:
            logger.warning(f"Validation warnings: {validation_result.errors}")

        # Clean up checkpoint on success
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

        return self.config.output_dir

    def _plan_responses(
        self,
        themes: list[dict],
        demographics: list[dict],
    ) -> list[ResponseSpec]:
        """Plan response specifications ensuring distribution targets.

        Args:
            themes: Generated themes for the question.
            demographics: Sampled demographic profiles.

        Returns:
            List of ResponseSpec objects defining each response to generate.
        """
        specs = []
        n = self.config.n_responses

        # Calculate counts for each response type
        dist = self.config.position_distribution
        type_counts = {
            ResponseType.AGREE: int(n * dist["agree"]),
            ResponseType.DISAGREE: int(n * dist["disagree"]),
            ResponseType.NUANCED: int(n * dist["nuanced"]),
            ResponseType.OFF_TOPIC: int(n * dist.get("off_topic", 0.05)),
            ResponseType.LOW_QUALITY: int(n * dist.get("low_quality", 0.05)),
        }

        # Ensure counts sum to n
        total_assigned = sum(type_counts.values())
        if total_assigned < n:
            type_counts[ResponseType.NUANCED] += n - total_assigned

        # Calculate length counts
        length_dist = self.config.length_distribution
        length_counts = {
            ResponseLength.SHORT: int(n * length_dist["short"]),
            ResponseLength.MEDIUM: int(n * length_dist["medium"]),
            ResponseLength.LONG: int(n * length_dist["long"]),
        }

        # Get regular themes (excluding X and Y)
        regular_themes = [t for t in themes if t["topic_id"] not in ("X", "Y")]

        response_id = 1001
        for response_type, count in type_counts.items():
            for _ in range(count):
                # Select themes for this response
                theme_ids, stances = self._select_themes_for_response(
                    response_type, regular_themes
                )

                # Sample length
                length = self._sample_length(length_counts)

                # Determine noise application
                apply_noise, noise_type = self._sample_noise()

                specs.append(
                    ResponseSpec(
                        response_id=response_id,
                        themes=theme_ids,
                        stances=stances,
                        response_type=response_type,
                        length=length,
                        persona=demographics[(response_id - 1001) % len(demographics)],
                        apply_noise=apply_noise,
                        noise_type=noise_type,
                    )
                )

                response_id += 1

        return specs

    def _select_themes_for_response(
        self,
        response_type: ResponseType,
        regular_themes: list[dict],
    ) -> tuple[list[str], list[str]]:
        """Select themes and stances for a response based on type.

        Args:
            response_type: The type of response being generated.
            regular_themes: List of regular themes (excluding X, Y).

        Returns:
            Tuple of (theme_ids, stances).
        """
        if response_type == ResponseType.OFF_TOPIC:
            return ["X"], ["NEUTRAL"]

        if response_type == ResponseType.LOW_QUALITY:
            return ["Y"], ["NEUTRAL"]

        # For regular responses: select 1-3 themes
        n_themes = int(self.rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))

        # Ensure multi-theme ratio is respected
        if self.rng.random() < self.config.multi_theme_ratio:
            n_themes = max(2, n_themes)

        n_themes = min(n_themes, len(regular_themes))

        selected = self.rng.choice(regular_themes, size=n_themes, replace=False)
        theme_ids = [t["topic_id"] for t in selected]

        # Assign stances based on response type
        if response_type == ResponseType.AGREE:
            stances = ["POSITIVE"] * len(theme_ids)
        elif response_type == ResponseType.DISAGREE:
            stances = ["NEGATIVE"] * len(theme_ids)
        else:  # NUANCED
            stances = self.rng.choice(
                ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                size=len(theme_ids),
            ).tolist()

        return theme_ids, stances

    def _sample_length(
        self, length_counts: dict[ResponseLength, int]
    ) -> ResponseLength:
        """Sample a response length from remaining allocation.

        Args:
            length_counts: Remaining counts per length category.

        Returns:
            Selected ResponseLength.
        """
        available = [length for length, count in length_counts.items() if count > 0]
        if not available:
            return ResponseLength.MEDIUM

        length = self.rng.choice(available)
        length_counts[length] -= 1
        return length

    def _sample_noise(self) -> tuple[bool, str | None]:
        """Determine whether to apply noise and what type.

        Returns:
            Tuple of (apply_noise, noise_type).
        """
        noise_rates = {
            NoiseLevel.LOW: {
                "typo": 0.02,
                "grammar": 0.02,
                "caps": 0.0,
                "emotional": 0.05,
                "sarcasm": 0.0,
            },
            NoiseLevel.MEDIUM: {
                "typo": 0.05,
                "grammar": 0.08,
                "caps": 0.02,
                "emotional": 0.15,
                "sarcasm": 0.03,
            },
            NoiseLevel.HIGH: {
                "typo": 0.15,
                "grammar": 0.20,
                "caps": 0.05,
                "emotional": 0.30,
                "sarcasm": 0.08,
            },
        }

        rates = noise_rates[self.config.noise_level]

        for noise_type, rate in rates.items():
            if self.rng.random() < rate:
                return True, noise_type

        return False, None

    def _batch(self, items: list, size: int):
        """Yield batches of items.

        Args:
            items: List to batch.
            size: Batch size.

        Yields:
            Batches of items.
        """
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _save_checkpoint(
        self, question_part: str, batch_num: int, total_specs: int
    ) -> None:
        """Save checkpoint for recovery.

        Args:
            question_part: Current question part being processed.
            batch_num: Number of batches completed.
            total_specs: Total number of response specs for this question.
        """
        checkpoint = {
            "question_part": question_part,
            "batch_num": batch_num,
            "total_specs": total_specs,
            "generated_count": self._generated_count,
        }
        with open(self._checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
