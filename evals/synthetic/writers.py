"""File output handlers for synthetic dataset generation."""

import json
from datetime import date
from pathlib import Path

from synthetic.config import QuestionConfig


class DatasetWriter:
    """Handles writing dataset files in the expected output format.

    Supports both batch writes (all at once) and streaming writes (append per batch)
    for reduced memory usage and crash resilience.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialise writer with output directory.

        Args:
            output_dir: Path to dataset output directory.
        """
        self.output_dir = output_dir
        self.date_str = date.today().isoformat()
        self._initialised_questions: set[str] = set()

    def initialise_directories(self, questions: list[QuestionConfig]) -> None:
        """Create directory structure for dataset.

        Args:
            questions: List of question configurations.
        """
        inputs_dir = self.output_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        for q in questions:
            question_dir = inputs_dir / f"question_part_{q.number}"
            question_dir.mkdir(exist_ok=True)

            outputs_dir = (
                self.output_dir
                / "outputs"
                / "mapping"
                / self.date_str
                / f"question_part_{q.number}"
            )
            outputs_dir.mkdir(parents=True, exist_ok=True)

    def write_question(self, question_part: str, config: QuestionConfig) -> None:
        """Write question.json file.

        Args:
            question_part: Question part folder name (e.g., "question_part_1").
            config: Question configuration.
        """
        path = self.output_dir / "inputs" / question_part / "question.json"
        data = {
            "question_number": config.number,
            "question_text": config.text,
        }
        if config.multi_choice_options:
            data["multi_choice_options"] = config.multi_choice_options

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def write_responses(self, question_part: str, responses: list[dict]) -> None:
        """Write responses.jsonl file.

        Args:
            question_part: Question part folder name.
            responses: List of response dicts with response_id and response.
        """
        path = self.output_dir / "inputs" / question_part / "responses.jsonl"
        with open(path, "w") as f:
            for r in responses:
                line = json.dumps(
                    {
                        "response_id": r["response_id"],
                        "response": r["response"],
                    }
                )
                f.write(line + "\n")

    def write_respondents(self, respondents: list[dict]) -> None:
        """Write respondents.jsonl file.

        Args:
            respondents: List of respondent dicts with response_id and demographic_data.
        """
        path = self.output_dir / "inputs" / "respondents.jsonl"
        with open(path, "w") as f:
            for r in respondents:
                f.write(json.dumps(r) + "\n")

    def write_themes(self, question_part: str, themes: list[dict]) -> None:
        """Write themes.json file.

        Args:
            question_part: Question part folder name.
            themes: List of theme dicts with topic_id, topic_label, etc.
        """
        path = (
            self.output_dir
            / "outputs"
            / "mapping"
            / self.date_str
            / question_part
            / "themes.json"
        )
        with open(path, "w") as f:
            json.dump(themes, f, indent=4)

    def write_sentiment(self, question_part: str, responses: list[dict]) -> None:
        """Write sentiment.jsonl file.

        Args:
            question_part: Question part folder name.
            responses: List of response dicts with response_id and position.
        """
        path = (
            self.output_dir
            / "outputs"
            / "mapping"
            / self.date_str
            / question_part
            / "sentiment.jsonl"
        )
        with open(path, "w") as f:
            for r in responses:
                line = json.dumps(
                    {
                        "response_id": r["response_id"],
                        "position": r["position"],
                    }
                )
                f.write(line + "\n")

    def write_mappings(self, question_part: str, responses: list[dict]) -> None:
        """Write mapping.jsonl file.

        Args:
            question_part: Question part folder name.
            responses: List of response dicts with response_id, labels, and stances.
        """
        path = (
            self.output_dir
            / "outputs"
            / "mapping"
            / self.date_str
            / question_part
            / "mapping.jsonl"
        )
        with open(path, "w") as f:
            for r in responses:
                line = json.dumps(
                    {
                        "response_id": r["response_id"],
                        "labels": r["labels"],
                        "stances": r["stances"],
                    }
                )
                f.write(line + "\n")

    def write_detail_detection(self, question_part: str, responses: list[dict]) -> None:
        """Write detail_detection.jsonl file.

        Args:
            question_part: Question part folder name.
            responses: List of response dicts with response_id and evidence_rich.
        """
        path = (
            self.output_dir
            / "outputs"
            / "mapping"
            / self.date_str
            / question_part
            / "detail_detection.jsonl"
        )
        with open(path, "w") as f:
            for r in responses:
                line = json.dumps(
                    {
                        "response_id": r["response_id"],
                        "evidence_rich": r["evidence_rich"],
                    }
                )
                f.write(line + "\n")

    # =========================================================================
    # Streaming write methods (append per batch for reduced RAM / crash safety)
    # =========================================================================

    def init_streaming_files(self, question_part: str) -> None:
        """Initialise empty JSONL files for streaming writes.

        Call this once per question before streaming batches.

        Args:
            question_part: Question part folder name.
        """
        if question_part in self._initialised_questions:
            return

        # Clear/create all JSONL files for this question
        responses_path = self.output_dir / "inputs" / question_part / "responses.jsonl"
        responses_path.write_text("")

        outputs_base = (
            self.output_dir / "outputs" / "mapping" / self.date_str / question_part
        )
        (outputs_base / "sentiment.jsonl").write_text("")
        (outputs_base / "mapping.jsonl").write_text("")
        (outputs_base / "detail_detection.jsonl").write_text("")

        self._initialised_questions.add(question_part)

    def append_responses(self, question_part: str, responses: list[dict]) -> None:
        """Append a batch of responses to JSONL files (streaming mode).

        Args:
            question_part: Question part folder name.
            responses: Batch of response dicts to append.
        """
        # Append to responses.jsonl
        responses_path = self.output_dir / "inputs" / question_part / "responses.jsonl"
        with open(responses_path, "a") as f:
            for r in responses:
                f.write(
                    json.dumps(
                        {
                            "response_id": r["response_id"],
                            "response": r["response"],
                        }
                    )
                    + "\n"
                )

        outputs_base = (
            self.output_dir / "outputs" / "mapping" / self.date_str / question_part
        )

        # Append to sentiment.jsonl
        with open(outputs_base / "sentiment.jsonl", "a") as f:
            for r in responses:
                f.write(
                    json.dumps(
                        {
                            "response_id": r["response_id"],
                            "position": r["position"],
                        }
                    )
                    + "\n"
                )

        # Append to mapping.jsonl
        with open(outputs_base / "mapping.jsonl", "a") as f:
            for r in responses:
                f.write(
                    json.dumps(
                        {
                            "response_id": r["response_id"],
                            "labels": r["labels"],
                            "stances": r["stances"],
                        }
                    )
                    + "\n"
                )

        # Append to detail_detection.jsonl
        with open(outputs_base / "detail_detection.jsonl", "a") as f:
            for r in responses:
                f.write(
                    json.dumps(
                        {
                            "response_id": r["response_id"],
                            "evidence_rich": r["evidence_rich"],
                        }
                    )
                    + "\n"
                )
