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
    ResponseType,
)
from synthetic.demographics import sample_demographics
from synthetic.llm_generators.response_generator import (
    RespondentSpec,
    generate_respondent_batch,
)
from synthetic.llm_generators.theme_generator import generate_themes
from synthetic.validators import validate_dataset
from synthetic.writers import DatasetWriter

logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # Respondents per batch (parallelised across respondents)


class SyntheticDatasetGenerator:
    """Orchestrates synthetic consultation dataset generation.

    Uses a per-respondent sequential approach: each respondent answers all
    questions in order, with previous responses passed as context for consistency.
    Respondents are processed in parallel batches.
    """

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

        # Step 1: Generate themes for ALL questions upfront
        logger.info("Generating themes for all questions...")
        themes_by_question: dict[int, list[dict]] = {}

        for question_config in self.config.questions:
            themes = await generate_themes(
                topic=self.config.topic,
                question=question_config.text,
                demographic_fields=self.config.demographic_fields,
                callbacks=self.callbacks,
            )
            themes_by_question[question_config.number] = themes
            logger.info(
                f"Generated {len(themes)} themes for question {question_config.number}"
            )

            # Write themes and question config
            question_part = f"question_part_{question_config.number}"
            self.writer.write_themes(question_part, themes)
            self.writer.write_question(question_part, question_config)

        # Step 2: Sample demographics and create respondent specs
        demographics = sample_demographics(
            self.config.demographic_fields,
            self.config.n_responses,
            self.rng,
        )

        respondent_specs = self._create_respondent_specs(demographics)

        # Write respondents file
        respondents = [
            {"response_id": spec.response_id, "demographic_data": spec.persona}
            for spec in respondent_specs
        ]
        self.writer.write_respondents(respondents)

        # Initialise streaming files for all questions
        for question_config in self.config.questions:
            question_part = f"question_part_{question_config.number}"
            self.writer.init_streaming_files(question_part)

        # Step 3: Generate responses in batches of respondents
        total_responses = self.config.n_responses * len(self.config.questions)
        task_id = progress.task_ids[0] if progress else None

        def on_response_complete():
            """Callback for each completed response."""
            self._generated_count += 1
            if progress and task_id is not None:
                progress.update(task_id, completed=self._generated_count)

        logger.info(
            f"Generating {total_responses} responses "
            f"({self.config.n_responses} respondents × {len(self.config.questions)} questions)..."
        )

        batch_num = 0
        for batch_specs in self._batch(respondent_specs, BATCH_SIZE):
            batch_num += 1
            logger.info(
                f"Processing batch {batch_num} ({len(batch_specs)} respondents)..."
            )

            batch_responses = await generate_respondent_batch(
                llm=self.llm,
                respondents=batch_specs,
                questions=self.config.questions,
                themes_by_question=themes_by_question,
                noise_level=self.config.noise_level,
                callbacks=self.callbacks,
                on_response_complete=on_response_complete,
            )

            # Group responses by question and write
            self._write_batch_responses(batch_responses)

            # Checkpoint
            self._save_checkpoint(batch_num, len(respondent_specs))

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

    def _create_respondent_specs(
        self,
        demographics: list[dict],
    ) -> list[RespondentSpec]:
        """Create respondent specifications with assigned dispositions.

        Args:
            demographics: Sampled demographic profiles.

        Returns:
            List of RespondentSpec objects.
        """
        n = self.config.n_responses
        specs = []

        # Calculate disposition counts based on distribution
        dist = self.config.position_distribution
        disposition_counts = {
            ResponseType.AGREE: int(n * dist["agree"]),
            ResponseType.DISAGREE: int(n * dist["disagree"]),
            ResponseType.NUANCED: int(n * dist["nuanced"]),
            ResponseType.OFF_TOPIC: int(n * dist.get("off_topic", 0.05)),
            ResponseType.LOW_QUALITY: int(n * dist.get("low_quality", 0.05)),
        }

        # Ensure counts sum to n
        total_assigned = sum(disposition_counts.values())
        if total_assigned < n:
            disposition_counts[ResponseType.NUANCED] += n - total_assigned

        # Calculate length counts
        length_dist = self.config.length_distribution
        length_counts = {
            ResponseLength.SHORT: int(n * length_dist["short"]),
            ResponseLength.MEDIUM: int(n * length_dist["medium"]),
            ResponseLength.LONG: int(n * length_dist["long"]),
        }

        response_id = 1001
        for disposition, count in disposition_counts.items():
            for _ in range(count):
                # Sample length
                length = self._sample_length(length_counts)

                # Determine noise application
                apply_noise, noise_type = self._sample_noise()

                specs.append(
                    RespondentSpec(
                        response_id=response_id,
                        persona=demographics[(response_id - 1001) % len(demographics)],
                        base_disposition=disposition,
                        length=length,
                        apply_noise=apply_noise,
                        noise_type=noise_type,
                    )
                )

                response_id += 1

        # Shuffle to mix dispositions
        self.rng.shuffle(specs)

        return specs

    def _write_batch_responses(self, responses: list[dict]) -> None:
        """Write batch responses grouped by question.

        Args:
            responses: Flat list of response dicts with question_number field.
        """
        # Group by question
        by_question: dict[int, list[dict]] = {}
        for response in responses:
            q_num = response["question_number"]
            if q_num not in by_question:
                by_question[q_num] = []
            by_question[q_num].append(response)

        # Write each question's responses
        for q_num, q_responses in by_question.items():
            question_part = f"question_part_{q_num}"
            self.writer.append_responses(question_part, q_responses)

    def _sample_length(
        self, length_counts: dict[ResponseLength, int]
    ) -> ResponseLength:
        """Sample a response length from remaining allocation."""
        available = [length for length, count in length_counts.items() if count > 0]
        if not available:
            return ResponseLength.MEDIUM

        length = self.rng.choice(available)
        length_counts[length] -= 1
        return length

    def _sample_noise(self) -> tuple[bool, str | None]:
        """Determine whether to apply noise and what type."""
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
        """Yield batches of items."""
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _save_checkpoint(self, batch_num: int, total_respondents: int) -> None:
        """Save checkpoint for recovery."""
        checkpoint = {
            "batch_num": batch_num,
            "total_respondents": total_respondents,
            "generated_count": self._generated_count,
        }
        with open(self._checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
