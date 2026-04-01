"""CLI script to set up a new consultation for ThemeFinder."""

import argparse
import logging
import os
import re
import sys
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


CONFLUENCE_URL = "https://incubatorforartificialintelligence.atlassian.net/wiki/spaces/Consult/pages/136445956/1.2+Set+up+the+consultation+in+the+app"

VALID_EXTENSIONS = {".csv", ".xlsx", ".xls"}


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


def excel_column_to_number(col: str) -> int:
    """Convert Excel column name to number for sorting (A=1, Z=26, AA=27)."""
    result = 0
    for c in col.strip().upper():
        result = result * 26 + (ord(c) - ord("A") + 1)
    return result


def _parse_question_numbers(values: pd.Series) -> list[int] | None:
    """Try to parse question numbers from a Series. Returns list of ints or None if any fail."""
    parsed = []
    for val in values.astype(str):
        stripped = re.sub(r"\D", "", val)
        if not stripped:
            return None
        parsed.append(int(stripped))
    return parsed


def _extract_numbers(text: str) -> list[int]:
    """Extract all integers from a string."""
    return [int(x) for x in re.findall(r"\d+", str(text))]


def validate_data(
    question_sheets: dict[str, pd.DataFrame],
    original_headers: dict[str, str],
    responses_df: pd.DataFrame,
) -> None:
    """Validate QU sheets against response data.

    Prints summaries of responses and QU sheets, then checks for:
    - QU columns that don't exist in the response data
    - More QU columns referenced than response columns available
    - Low string similarity between QU labels and response headers
    - Mismatched numbers extracted from QU labels vs response headers
    If any issues found, prompts user to confirm before continuing.
    """
    col_id_field = {
        "open": ["column_name"],
        "hybrid": ["open_column", "closed_column"],
        "closed": ["column_name"],
    }

    issues: list[str] = []

    # --- Response summary ---
    n_rows, n_cols = responses_df.shape
    # Exclude themefinder_id from column count (added by load_responses)
    resp_col_count = n_cols - 1
    total_cells = n_rows * resp_col_count
    nan_count = int(responses_df.drop(columns=["themefinder_id"]).isna().sum().sum())
    nan_pct = (nan_count / total_cells * 100) if total_cells else 0
    print(f"\n  Response data summary:")
    print(f"    Rows: {n_rows}")
    print(f"    Columns: {resp_col_count}")
    print(f"    NaN cells: {nan_count} / {total_cells} ({nan_pct:.1f}%)")

    # --- QU summary ---
    all_qu_columns: set[str] = set()
    total_questions = 0
    print(f"\n  Question Understanding summary:")
    for sheet_key, df in question_sheets.items():
        n_questions = len(df)
        total_questions += n_questions
        cols_in_sheet: set[str] = set()
        for col_field in col_id_field[sheet_key]:
            cols_in_sheet.update(df[col_field].astype(str).str.strip().tolist())
        all_qu_columns.update(cols_in_sheet)
        print(f"    {sheet_key}: {n_questions} question(s), {len(cols_in_sheet)} distinct column(s) referenced")
    print(f"    Total: {total_questions} question(s), {len(all_qu_columns)} distinct column(s) referenced")

    # --- Column count check ---
    if len(all_qu_columns) > resp_col_count:
        msg = (
            f"Q.U. sheets reference {len(all_qu_columns)} distinct columns "
            f"but response data only has {resp_col_count} columns"
        )
        logger.warning(msg)
        issues.append(msg)

    # --- Check each QU column exists in responses ---
    max_resp_col = max(original_headers, key=excel_column_to_number) if original_headers else "?"
    for col_id in sorted(all_qu_columns, key=excel_column_to_number):
        if col_id not in original_headers:
            msg = (
                f"Q.U. references column {col_id} which does not exist in response data "
                f"(response columns go up to {max_resp_col})"
            )
            logger.warning(msg)
            issues.append(msg)

    # --- Label matching ---
    for sheet_key, df in question_sheets.items():
        for _, row in df.iterrows():
            for col_field in col_id_field[sheet_key]:
                col_id = str(row[col_field]).strip()
                qu_label = str(row.get("question_text", "")).strip()
                resp_header = original_headers.get(col_id)

                if resp_header is None:
                    continue  # already reported above

                ratio = SequenceMatcher(None, qu_label.lower(), resp_header.lower()).ratio()

                if ratio < 0.4:
                    msg = (
                        f"Low similarity ({ratio:.2f}) between Q.U. label and Response header for column {col_id}:\n"
                        f"  Question Understanding = '{qu_label}'\n"
                        f"  Response Header        = '{resp_header}'"
                    )
                    logger.warning(msg)
                    issues.append(msg)

                qu_nums = set(_extract_numbers(qu_label))
                resp_nums = set(_extract_numbers(resp_header))
                if qu_nums and resp_nums and qu_nums != resp_nums:
                    msg = (
                        f"Number mismatch for column {col_id}: Q.U. has {sorted(qu_nums)}, response has {sorted(resp_nums)}.\n"
                        f"  Question Understanding = '{qu_label}'\n"
                        f"  Response Header        = '{resp_header}'"
                    )
                    logger.warning(msg)
                    issues.append(msg)

    if issues:
        print(f"\n  Found {len(issues)} validation issue(s) (see warnings above).")
        answer = input("Continue anyway? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborting.")
            sys.exit(1)
    else:
        print("  Validation passed — no issues found.")


def load_and_number_question_sheets(
    question_understanding_path: str,
) -> dict[str, pd.DataFrame]:
    """Load all question sheets, truncate to useful columns, and assign question numbers.

    If any question_number value across any sheet cannot be parsed as an integer,
    falls back to numbering all questions sequentially by sorting their Excel column
    IDs across all sheets.
    """
    sheets: dict[str, pd.DataFrame] = {}
    col_id_field = {
        "open": "column_name",
        "hybrid": "open_column",
        "closed": "column_name",
    }

    # Load and truncate each sheet to its useful columns
    for sheet_key, (sheet_name, ncols, col_names) in {
        "open": ("Open questions", 3, ["column_name", "question_number", "question_text"]),
        "hybrid": ("Hybrid questions", 4, ["open_column", "question_number", "question_text", "closed_column"]),
        "closed": ("Multiple Choice", 3, ["column_name", "question_number", "question_text"]),
    }.items():
        try:
            df = pd.read_excel(
                question_understanding_path, sheet_name=sheet_name, skiprows=3
            )
            if not df.empty:
                df = df.iloc[:, :ncols]
                df.columns = col_names
                sheets[sheet_key] = df
        except Exception:
            pass

    if not sheets:
        return sheets

    # Check whether all question_number values can be parsed as integers
    needs_fallback = False
    for df in sheets.values():
        if _parse_question_numbers(df["question_number"]) is None:
            needs_fallback = True
            break

    if needs_fallback:
        print("  Some question numbers are not numeric, using column-based numbering fallback.")
        # Collect (excel_col_id, sheet_key, df_index) from every row across all sheets
        all_entries: list[tuple[str, str, int]] = []
        for key, df in sheets.items():
            for idx in df.index:
                all_entries.append((str(df.at[idx, col_id_field[key]]).strip(), key, idx))

        # Sort by Excel column order and assign sequential numbers
        all_entries.sort(key=lambda x: excel_column_to_number(x[0]))
        number_map: dict[tuple[str, int], int] = {}
        for i, (_, key, idx) in enumerate(all_entries, 1):
            number_map[(key, idx)] = i

        for key, df in sheets.items():
            df["question_number"] = [number_map[(key, idx)] for idx in df.index]
    else:
        for key, df in sheets.items():
            df["question_number"] = _parse_question_numbers(df["question_number"])

    # Validate global uniqueness (numbers are used as directory names)
    all_numbers: list[tuple[str, int]] = []
    for key, df in sheets.items():
        for num in df["question_number"].tolist():
            all_numbers.append((key, num))
    seen: dict[int, list[str]] = {}
    for key, num in all_numbers:
        seen.setdefault(num, []).append(key)
    duplicates = {num: keys for num, keys in seen.items() if len(keys) > 1}
    if duplicates:
        detail = ", ".join(
            f"question_number={num} appears in [{', '.join(keys)}]"
            for num, keys in sorted(duplicates.items())
        )
        raise AssertionError(f"Non-unique question numbers found across sheets: {detail}")

    return sheets


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
        df[c] = df[c].apply(lambda x: x.split(","))
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
        raise AssertionError("Non-unique values found in 'question_number' column")

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
            lambda x: x.split(",")
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
        raise AssertionError("Non-unique values found in 'question_number' column")

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
            lambda x: x.split(",")
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
    question_info.columns = ["column_name", "question_number", "question_text"]

    question_info["question_number"] = (
        question_info["question_number"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .astype(int)
    )
    if not question_info["question_number"].is_unique:
        raise AssertionError("Non-unique values found in 'question_number' column")

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


def load_responses(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load responses from CSV or Excel file.

    Returns the DataFrame (with columns renamed to Excel letters) and a
    dict mapping Excel column letter -> original column header string.
    """
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, header=0)
    else:
        df = pd.read_excel(path, header=0)
    original_headers = {
        get_excel_column_name(i): str(col)
        for i, col in enumerate(df.columns)
    }
    df.columns = [get_excel_column_name(i) for i in range(len(df.columns))]
    df["themefinder_id"] = range(1, len(df) + 1)
    return df, original_headers


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
    responses_df, original_headers = load_responses(responses_path)
    print(f"  Loaded {len(responses_df)} responses")

    os.makedirs(output_dir, exist_ok=True)
    qu_path = str(question_understanding_path)

    print("Processing demographics...")
    save_demographic_data(responses_df, qu_path, output_dir)

    # Load all question sheets with robust numbering
    question_sheets = load_and_number_question_sheets(qu_path)

    # Validate QU sheets against response data
    validate_data(question_sheets, original_headers, responses_df)

    open_q = question_sheets.get("open")
    if open_q is not None and not open_q.empty:
        print("Processing open questions...")
        only_nans = responses_df[open_q["column_name"].tolist()].isna().all()
        nan_cols = only_nans[only_nans].index.tolist()
        open_q = open_q[~open_q["column_name"].isin(nan_cols)]
        create_open_question_inputs(
            responses_df, open_q.to_dict(orient="records"), output_dir
        )
    else:
        print("  No open questions found, skipping.")

    hybrid_q = question_sheets.get("hybrid")
    if hybrid_q is not None and not hybrid_q.empty:
        print("Processing hybrid questions...")
        create_hybrid_question_inputs(
            responses_df, hybrid_q.to_dict(orient="records"), output_dir
        )
    else:
        print("  No hybrid questions found, skipping.")

    closed_q = question_sheets.get("closed")
    if closed_q is not None and not closed_q.empty:
        print("Processing closed questions...")
        create_closed_question_inputs(
            responses_df, closed_q.to_dict(orient="records"), output_dir
        )
    else:
        print("  No closed questions found, skipping.")

    print(f"\nAll input files written to: {output_dir}")


def upload_inputs_to_s3(local_dir: str, bucket: str, s3_prefix: str) -> None:
    """Upload all files in local_dir to s3://bucket/s3_prefix, preserving directory structure.

    Checks for existing objects at the S3 prefix before uploading. If any exist,
    warns and requires confirmation. Always prompts before uploading.
    """
    s3 = boto3.client("s3")
    local_path = Path(local_dir)
    files = [f for f in local_path.rglob("*") if f.is_file()]
    if not files:
        print(f"No files found in {local_dir} to upload.")
        return

    # Check for existing data at this S3 prefix
    print(f"\nChecking for existing data at s3://{bucket}/{s3_prefix} ...")
    existing = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix, MaxKeys=10)
    existing_keys = [obj["Key"] for obj in existing.get("Contents", [])]
    if existing_keys:
        print(f"  Found {len(existing_keys)} existing object(s) at this prefix:")
        for key in existing_keys:
            print(f"    {key}")
        if existing.get("IsTruncated"):
            print("    ... (more objects not shown)")
        logger.warning(
            "Uploading will overwrite existing data at s3://%s/%s",
            bucket, s3_prefix,
        )

    print(f"\nReady to upload {len(files)} file(s) to s3://{bucket}/{s3_prefix}")
    for file_path in files:
        relative = file_path.relative_to(local_path)
        print(f"  {relative}")
    answer = input("Proceed with upload? (y/n): ").strip().lower()
    if answer != "y":
        print("Upload skipped.")
        return

    for file_path in files:
        relative = file_path.relative_to(local_path)
        s3_key = s3_prefix + str(relative)
        print(f"  Uploading {relative} -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(file_path), bucket, s3_key)
    print("Upload complete.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

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
