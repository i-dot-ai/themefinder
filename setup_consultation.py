"""CLI script to set up a new consultation for ThemeFinder."""

import argparse
import os
import re
import sys
import json
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd


CONFLUENCE_URL = "https://incubatorforartificialintelligence.atlassian.net/wiki/spaces/Consult/pages/136445956/1.2+Set+up+the+consultation+in+the+app"

VALID_EXTENSIONS = {".csv", ".xlsx", ".xls"}

EXCEL_COLUMN_RE = re.compile(r"^[A-Z]{1,3}$")


def to_snake_case(s: str) -> str:
    """Convert a string to snake_case."""
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[\s\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


# --- Data processing functions ---


def get_excel_column_name(n: int) -> str:
    """Convert number to Excel column name (e.g., 0->A, 25->Z, 26->AA)."""
    result = ""
    n += 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def create_respondents_jsonl(
    df: pd.DataFrame,
    demographic_columns: list[str],
    demographic_labels: list[str],
    output_dir: str,
) -> None:
    for c in demographic_columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace("_x000D_", "", regex=False)
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )
    for c in demographic_columns:
        df[c] = df[c].apply(lambda x: list(dict.fromkeys(x.split(","))))
    df.rename(columns=dict(zip(demographic_columns, demographic_labels)), inplace=True)
    df["demographic_data"] = df[demographic_labels].to_dict(orient="records")
    df[["themefinder_id", "demographic_data"]].to_json(
        os.path.join(output_dir, "respondents.jsonl"), orient="records", lines=True
    )


def save_demographic_data(
    responses_df: pd.DataFrame, question_understanding_path: str, output_dir: str
) -> None:
    demographic_info = pd.read_excel(
        question_understanding_path, sheet_name="Demographic", skiprows=3, header=None
    )
    if demographic_info.empty:
        print("  No demographic data found, skipping.")
        return
    demographic_info = demographic_info[
        demographic_info[demographic_info.columns[0]]
        .astype(str)
        .str.match(EXCEL_COLUMN_RE)
    ].reset_index(drop=True)
    demographic_questions = demographic_info[demographic_info.columns[0]].tolist()
    demographic_labels = demographic_info[demographic_info.columns[1]].tolist()
    demographic_labels = [label.replace("/", "-") for label in demographic_labels]

    for c in demographic_questions:
        responses_df[c] = responses_df[c].fillna("Not Provided")
        responses_df[c] = responses_df[c].apply(
            lambda x: "Other" if isinstance(x, str) and "Other" in x else x
        )
    create_respondents_jsonl(
        responses_df, demographic_questions, demographic_labels, output_dir
    )


def create_open_question_inputs(
    df: pd.DataFrame,
    open_questions: list[dict],
    output_dir: str,
    characters_to_remove: list[str] = ["/", "\\", "- Text", "_x000D_"],
    sample_size: Optional[int] = None,
) -> None:
    for question in open_questions:
        q_num = question["question_number"]
        question_col = question["column_name"]
        q_dir = os.path.join(output_dir, f"question_part_{q_num}")
        os.makedirs(q_dir, exist_ok=True)

        question_string = question["question_text"]
        question_answers = df[["themefinder_id", question_col]].dropna()
        if sample_size is not None and sample_size < len(question_answers):
            question_answers = question_answers.sample(sample_size)

        for bad_string in characters_to_remove:
            question_answers[question_col] = question_answers[question_col].apply(
                lambda x, bs=bad_string: x.replace(bs, " ")
            )
        question_answers[question_col] = (
            question_answers[question_col]
            .astype(str)
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )
        question_answers.columns = ["themefinder_id", "text"]
        question_answers[["themefinder_id", "text"]].to_json(
            os.path.join(q_dir, "responses.jsonl"), orient="records", lines=True
        )

        question_data = {
            "question_number": q_num,
            "question_text": question_string,
            "has_free_text": True,
        }
        with open(os.path.join(q_dir, "question.json"), "w") as f:
            json.dump(question_data, f, indent=4)


def save_open_questions(
    responses_df: pd.DataFrame, question_understanding_path: str, output_dir: str
) -> None:
    question_info = pd.read_excel(
        question_understanding_path,
        sheet_name="Open questions",
        skiprows=3,
        header=None,
    )
    if question_info.empty:
        print("  No open questions found, skipping.")
        return
    question_info = question_info[
        question_info[question_info.columns[0]].astype(str).str.match(EXCEL_COLUMN_RE)
    ].reset_index(drop=True)
    question_info.columns = ["column_name", "question_number", "question_text"]

    only_nans = responses_df[question_info["column_name"].tolist()].isna().all()
    column_names_with_only_nans = only_nans[only_nans].index.tolist()
    question_info = question_info[
        ~question_info["column_name"].isin(column_names_with_only_nans)
    ]

    question_info["question_number"] = (
        question_info["question_number"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .astype(int)
    )
    if not question_info["question_number"].is_unique:
        dupes = question_info[question_info["question_number"].duplicated(keep=False)][
            "question_number"
        ].tolist()
        raise AssertionError(
            f"Non-unique values found in 'question_number' column of 'Open questions' sheet: {dupes}"
        )

    create_open_question_inputs(
        responses_df, question_info.to_dict(orient="records"), output_dir
    )


def create_hybrid_question_inputs(
    df: pd.DataFrame,
    hybrid_questions: list[dict],
    output_dir: str,
    characters_to_remove: list[str] = ["/", "\\", "- Text", "_x000D_"],
    sample_size: Optional[int] = None,
) -> None:
    for question in hybrid_questions:
        q_num = question["question_number"]
        q_dir = os.path.join(output_dir, f"question_part_{q_num}")
        closed_col = question["closed_column"]
        open_col = question["open_column"]
        question_string = question["question_text"]
        os.makedirs(q_dir, exist_ok=True)

        question_answers = df[["themefinder_id"] + [closed_col, open_col]].dropna(
            subset=[closed_col, open_col], how="all"
        )
        if sample_size is not None and sample_size < len(question_answers):
            question_answers = question_answers.sample(sample_size)

        question_answers[closed_col] = question_answers[closed_col].fillna(
            "Not Provided"
        )
        question_answers[open_col] = question_answers[open_col].fillna("Not Provided")

        question_answers[closed_col] = (
            question_answers[closed_col]
            .astype(str)
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )
        question_answers[open_col] = (
            question_answers[open_col]
            .astype(str)
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )

        for bad_string in characters_to_remove:
            question_answers[closed_col] = question_answers[closed_col].apply(
                lambda x, bs=bad_string: x.replace(bs, " ")
            )
            question_answers[open_col] = question_answers[open_col].apply(
                lambda x, bs=bad_string: x.replace(bs, " ")
            )

        question_answers[closed_col] = question_answers[closed_col].apply(
            lambda x: list(dict.fromkeys(x.split(",")))
        )
        question_answers.rename(
            columns={closed_col: "options", open_col: "text"}, inplace=True
        )

        question_answers[["themefinder_id", "options"]].to_json(
            os.path.join(q_dir, "multi_choice.jsonl"), orient="records", lines=True
        )
        question_answers[["themefinder_id", "text"]].to_json(
            os.path.join(q_dir, "responses.jsonl"), orient="records", lines=True
        )

        question_data = {
            "question_number": q_num,
            "question_text": question_string,
            "has_free_text": True,
            "multi_choice_options": list(
                set(
                    [
                        item
                        for sublist in question_answers["options"]
                        for item in sublist
                    ]
                )
            ),
        }
        with open(os.path.join(q_dir, "question.json"), "w") as f:
            json.dump(question_data, f, indent=4)


def save_hybrid_questions(
    responses_df: pd.DataFrame, question_understanding_path: str, output_dir: str
) -> None:
    question_info = pd.read_excel(
        question_understanding_path,
        sheet_name="Hybrid questions",
        skiprows=3,
        header=None,
    )
    if question_info.empty:
        print("  No hybrid questions found, skipping.")
        return
    question_info = question_info[
        question_info[question_info.columns[0]].astype(str).str.match(EXCEL_COLUMN_RE)
    ].reset_index(drop=True)
    question_info.columns = [
        "open_column",
        "question_number",
        "question_text",
        "closed_column",
    ]

    question_info["question_number"] = (
        question_info["question_number"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .astype(int)
    )
    if not question_info["question_number"].is_unique:
        dupes = question_info[question_info["question_number"].duplicated(keep=False)][
            "question_number"
        ].tolist()
        raise AssertionError(
            f"Non-unique values found in 'question_number' column of 'Hybrid questions' sheet: {dupes}"
        )

    create_hybrid_question_inputs(
        responses_df, question_info.to_dict(orient="records"), output_dir
    )


def create_closed_question_inputs(
    df: pd.DataFrame,
    closed_questions: list[dict],
    output_dir: str,
    characters_to_remove: list[str] = ["/", "\\", "- Text", "_x000D_"],
    sample_size: Optional[int] = None,
) -> None:
    for question in closed_questions:
        q_num = question["question_number"]
        question_col = question["column_name"]
        q_dir = os.path.join(output_dir, f"question_part_{q_num}")
        os.makedirs(q_dir, exist_ok=True)

        question_string = question["question_text"]
        question_answers = df[["themefinder_id", question_col]].dropna()
        if sample_size is not None:
            question_answers = question_answers.sample(sample_size)

        question_answers[question_col] = (
            question_answers[question_col]
            .astype(str)
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )
        for bad_string in characters_to_remove:
            question_answers[question_col] = question_answers[question_col].apply(
                lambda x, bs=bad_string: x.replace(bs, " ")
            )

        question_answers[question_col] = question_answers[question_col].apply(
            lambda x: list(dict.fromkeys(x.split(",")))
        )
        question_answers.columns = ["themefinder_id", "options"]
        question_answers[["themefinder_id", "options"]].to_json(
            os.path.join(q_dir, "multi_choice.jsonl"), orient="records", lines=True
        )

        question_data = {
            "question_number": q_num,
            "question_text": question_string,
            "has_free_text": False,
            "multi_choice_options": list(
                set(
                    [
                        item
                        for sublist in question_answers["options"]
                        for item in sublist
                    ]
                )
            ),
        }
        with open(os.path.join(q_dir, "question.json"), "w") as f:
            json.dump(question_data, f, indent=4)


def save_closed_questions(
    responses_df: pd.DataFrame, question_understanding_path: str, output_dir: str
) -> None:
    question_info = pd.read_excel(
        question_understanding_path,
        sheet_name="Multiple Choice",
        skiprows=3,
        header=None,
    )
    if question_info.empty:
        print("  No closed questions found, skipping.")
        return
    question_info = question_info[
        question_info[question_info.columns[0]].astype(str).str.match(EXCEL_COLUMN_RE)
    ].reset_index(drop=True)
    question_info.columns = ["column_name", "question_number", "question_text"]

    question_info["question_number"] = (
        question_info["question_number"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .astype(int)
    )
    if not question_info["question_number"].is_unique:
        dupes = question_info[question_info["question_number"].duplicated(keep=False)][
            "question_number"
        ].tolist()
        raise AssertionError(
            f"Non-unique values found in 'question_number' column of 'Multiple Choice' sheet: {dupes}"
        )

    create_closed_question_inputs(
        responses_df, question_info.to_dict(orient="records"), output_dir
    )


# --- CLI logic ---


def find_data_files(consultation_dir: str) -> list[Path]:
    """Find CSV and Excel files in the consultation directory, ignoring temp files."""
    files = []
    for f in Path(consultation_dir).iterdir():
        if f.name.startswith("~$"):
            continue
        if f.suffix.lower() in VALID_EXTENSIONS:
            files.append(f)
    return sorted(files)


def load_responses(path: Path) -> pd.DataFrame:
    """Load responses from CSV or Excel file."""
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, header=0)
    else:
        df = pd.read_excel(path, header=0)
    df.columns = [get_excel_column_name(i) for i in range(len(df.columns))]
    df["themefinder_id"] = range(1, len(df) + 1)
    return df


def prompt_file_selection(files: list[Path], role: str) -> Path:
    """Ask the user to select which file serves a given role."""
    print(f"\nWhich file is the {role}?")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f.name}")
    while True:
        choice = input(f"Enter number (1-{len(files)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("Invalid choice, try again.")


def run_ingestion(
    responses_path: Path, question_understanding_path: Path, output_dir: str
) -> None:
    """Run the full ingestion pipeline."""
    print(f"\nLoading responses from: {responses_path.name}")
    responses_df = load_responses(responses_path)
    print(f"  Loaded {len(responses_df)} responses")

    os.makedirs(output_dir, exist_ok=True)
    qu_path = str(question_understanding_path)

    print("Processing demographics...")
    save_demographic_data(responses_df, qu_path, output_dir)

    print("Processing open questions...")
    save_open_questions(responses_df, qu_path, output_dir)

    print("Processing hybrid questions...")
    save_hybrid_questions(responses_df, qu_path, output_dir)

    print("Processing closed questions...")
    save_closed_questions(responses_df, qu_path, output_dir)

    print(f"\nAll input files written to: {output_dir}")


def upload_inputs_to_s3(local_dir: str, bucket: str, s3_prefix: str) -> None:
    """Upload all files in local_dir to s3://bucket/s3_prefix, preserving directory structure."""
    s3 = boto3.client("s3")
    local_path = Path(local_dir)
    files = [f for f in local_path.rglob("*") if f.is_file()]
    if not files:
        print(f"No files found in {local_dir} to upload.")
        return
    print(f"\nUploading {len(files)} file(s) to s3://{bucket}/{s3_prefix}")
    for file_path in files:
        relative = file_path.relative_to(local_path)
        s3_key = s3_prefix + str(relative)
        print(f"  {relative} -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(file_path), bucket, s3_key)
    print("Upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up a new consultation for ThemeFinder."
    )
    parser.add_argument(
        "name", nargs="?", help="Consultation name (used as folder name)"
    )
    args = parser.parse_args()

    name = args.name
    if not name:
        name = input("Enter consultation name: ").strip()
        if not name:
            print("Error: consultation name cannot be empty.")
            sys.exit(1)

    name = to_snake_case(name)
    print(f"Using consultation name: {name}")

    base_dir = Path(__file__).resolve().parent / "consultations"
    consultation_dir = base_dir / name

    # Step 1: Create the consultation folder
    consultation_dir.mkdir(parents=True, exist_ok=True)
    print(f"Consultation directory: {consultation_dir}")

    # Step 2: Check for data files
    print(
        "\nPlease copy the consultation response data and the template question"
        " understanding file into:"
    )
    print(f"  {consultation_dir}")
    input("\nPress Enter when the files are in place...")

    files = find_data_files(consultation_dir)
    if len(files) < 2:
        print(
            f"\nError: Expected at least 2 data files (.csv/.xlsx/.xls) but found"
            f" {len(files)}."
        )
        print("Please add the missing files and re-run the script.")
        sys.exit(1)

    # Step 3: Identify which file is which
    responses_path = prompt_file_selection(files, "consultation response data")
    remaining = [f for f in files if f != responses_path]
    if len(remaining) == 1:
        qu_path = remaining[0]
        print(f"\nUsing '{qu_path.name}' as the template question understanding file.")
    else:
        qu_path = prompt_file_selection(
            remaining, "template question understanding data"
        )

    # Step 4: Run ingestion
    output_dir = str(consultation_dir / "inputs")
    run_ingestion(responses_path, qu_path, output_dir)

    # Step 5: Upload inputs to S3
    s3_prefix = f"app_data/consultations/{name}/inputs/"
    upload_inputs_to_s3(output_dir, "i-dot-ai-prod-consult-data", s3_prefix)

    # Step 6: Point to Confluence
    print("\n" + "=" * 60)
    print("Setup complete! For further instructions, see:")
    print(f"  {CONFLUENCE_URL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
