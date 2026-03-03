"""Dataset validation utilities for synthetic data generation."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_dataset(dataset_path: Path) -> ValidationResult:
    """Validate generated dataset for completeness and referential integrity.

    Checks:
    1. All required files exist
    2. JSON/JSONL files parse correctly
    3. Referential integrity (response_id, topic_id links)
    4. Required fields present
    5. Special themes X and Y present

    Args:
        dataset_path: Path to dataset directory.

    Returns:
        ValidationResult with is_valid flag and any errors/warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check directory structure
    inputs_dir = dataset_path / "inputs"
    if not inputs_dir.exists():
        errors.append("Missing inputs/ directory")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # Find question parts
    question_parts = [
        d
        for d in inputs_dir.iterdir()
        if d.is_dir() and d.name.startswith("question_part")
    ]
    if not question_parts:
        errors.append("No question_part directories found")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # Collect all response IDs across questions
    all_response_ids: set[int] = set()

    for qp in question_parts:
        qp_errors = _validate_question_part(qp, all_response_ids)
        errors.extend(qp_errors)

    # Check respondents.jsonl
    respondents_file = inputs_dir / "respondents.jsonl"
    if respondents_file.exists():
        respondent_ids = set()
        with open(respondents_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    respondent_ids.add(data.get("response_id"))
                except json.JSONDecodeError:
                    pass

        missing_respondents = all_response_ids - respondent_ids
        if missing_respondents:
            sample = sorted(missing_respondents)[:5]
            errors.append(f"Missing respondent entries for response_ids: {sample}...")
    else:
        errors.append("Missing respondents.jsonl")

    # Check outputs
    outputs_dir = dataset_path / "outputs" / "mapping"
    if outputs_dir.exists():
        date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
        if date_dirs:
            latest_date = sorted(date_dirs, reverse=True)[0]
            output_errors = _validate_outputs(latest_date, question_parts, dataset_path)
            errors.extend(output_errors)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _validate_question_part(qp: Path, all_response_ids: set[int]) -> list[str]:
    """Validate a single question part directory.

    Args:
        qp: Path to question_part_N directory.
        all_response_ids: Set to populate with found response IDs.

    Returns:
        List of error messages.
    """
    errors: list[str] = []
    qp_name = qp.name

    # Check question.json
    question_file = qp / "question.json"
    if not question_file.exists():
        errors.append(f"Missing question.json in {qp_name}")
    else:
        try:
            with open(question_file) as f:
                question_data = json.load(f)
            if "question_text" not in question_data:
                errors.append(f"Missing question_text in {qp_name}/question.json")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in {qp_name}/question.json: {e}")

    # Check responses.jsonl
    responses_file = qp / "responses.jsonl"
    if not responses_file.exists():
        errors.append(f"Missing responses.jsonl in {qp_name}")
    else:
        response_ids_in_part: set[int] = set()
        with open(responses_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    rid = data.get("response_id")
                    if rid is None:
                        errors.append(
                            f"Missing response_id at line {line_num} in {qp_name}/responses.jsonl"
                        )
                    elif rid in response_ids_in_part:
                        errors.append(
                            f"Duplicate response_id {rid} in {qp_name}/responses.jsonl"
                        )
                    else:
                        response_ids_in_part.add(rid)
                        all_response_ids.add(rid)
                except json.JSONDecodeError as e:
                    errors.append(
                        f"Invalid JSON at line {line_num} in {qp_name}/responses.jsonl: {e}"
                    )

    return errors


def _validate_outputs(
    date_dir: Path, question_parts: list[Path], dataset_path: Path
) -> list[str]:
    """Validate output files for all question parts.

    Args:
        date_dir: Path to date-stamped output directory.
        question_parts: List of question part input directories.
        dataset_path: Root dataset path for relative path display.

    Returns:
        List of error messages.
    """
    errors: list[str] = []

    for qp in question_parts:
        qp_name = qp.name
        output_qp = date_dir / qp_name

        if not output_qp.exists():
            errors.append(f"Missing output directory for {qp_name}")
            continue

        # Check themes.json
        themes_file = output_qp / "themes.json"
        theme_ids: set[str] = set()
        if themes_file.exists():
            try:
                with open(themes_file) as f:
                    themes = json.load(f)

                theme_ids = {t["topic_id"] for t in themes}

                # Check special themes
                if "X" not in theme_ids:
                    errors.append(
                        f"Missing special theme X in {themes_file.relative_to(dataset_path)}"
                    )
                if "Y" not in theme_ids:
                    errors.append(
                        f"Missing special theme Y in {themes_file.relative_to(dataset_path)}"
                    )
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in {qp_name}/themes.json: {e}")
        else:
            errors.append(f"Missing themes.json in {qp_name}")

        # Check mapping.jsonl referential integrity
        mapping_file = output_qp / "mapping.jsonl"
        if mapping_file.exists() and theme_ids:
            with open(mapping_file) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        for label in data.get("labels", []):
                            if label not in theme_ids:
                                errors.append(
                                    f"Invalid theme reference '{label}' at line {line_num} in {qp_name}/mapping.jsonl"
                                )

                        labels_len = len(data.get("labels", []))
                        stances_len = len(data.get("stances", []))
                        if labels_len != stances_len:
                            errors.append(
                                f"Labels/stances length mismatch for response {data.get('response_id')} in {qp_name}"
                            )
                    except json.JSONDecodeError:
                        pass

        # Check required output files exist
        for required_file in [
            "mapping.jsonl",
            "detail_detection.jsonl",
        ]:
            if not (output_qp / required_file).exists():
                errors.append(f"Missing {required_file} in {qp_name}")

    return errors
