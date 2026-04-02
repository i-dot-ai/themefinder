"""CLI script to set up a new consultation for ThemeFinder."""

import argparse
import json
import logging
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


CONFLUENCE_URL = "https://incubatorforartificialintelligence.atlassian.net/wiki/spaces/Consult/pages/136445956/1.2+Set+up+the+consultation+in+the+app"

VALID_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Maps question type -> list of column-ID fields in the Q.U. sheet.
COL_ID_FIELDS: dict[str, list[str]] = {
    "open": ["column_name"],
    "hybrid": ["open_column", "closed_column"],
    "closed": ["column_name"],
}

# The primary column used for sorting/numbering (first entry in each list).
PRIMARY_COL_ID_FIELD: dict[str, str] = {
    key: fields[0] for key, fields in COL_ID_FIELDS.items()
}

# Sheet name, expected column count, and column names for each Q.U. sheet type.
QU_SHEET_SPECS: dict[str, tuple[str, int, list[str]]] = {
    "open": ("Open questions", 3, ["column_name", "question_number", "question_text"]),
    "hybrid": ("Hybrid questions", 4, ["open_column", "question_number", "question_text", "closed_column"]),
    "closed": ("Multiple Choice", 3, ["column_name", "question_number", "question_text"]),
}

# Characters stripped from free-text columns during ingestion.
CHARACTERS_TO_REMOVE: list[str] = ["/", "\\", "- Text", "_x000D_"]


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


# Matches valid Excel column IDs: 1-3 uppercase letters, possibly with trailing whitespace.
# Used to distinguish real data rows from instruction sub-header rows in Q.U. files.
_EXCEL_COL_RE = re.compile(r"^[A-Z]{1,3}\s*$")


def _load_qu_sheet(
    path: Path,
    sheet_name: str,
    n_columns: int,
    column_names: list[str],
) -> pd.DataFrame | None:
    """Load a single sheet from a Question Understanding Excel file.

    Q.U. files come in two formats:
      Format A (e.g. mhclg_cur047): 3 header rows then data at row 4.
      Format B (e.g. LGR files):    3 header rows, then an instruction sub-header
                                     row ("Which column does the question appear
                                     in..."), then data at row 5.

    Rather than hardcoding skiprows, we always skip the first 3 rows (title,
    description, column-header labels) and then auto-detect the instruction row.
    Real data rows start with an Excel column ID like "A", "AC", or "BF";
    instruction rows contain long descriptive text that fails that check.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, skiprows=3, header=None)
    except Exception:
        return None

    if df.empty:
        return None

    # Only keep the columns we expect — some files have extra unnamed columns
    # (e.g. abbreviated titles or notes in columns 4-5 of Multiple Choice)
    df = df.iloc[:, :n_columns]

    # Drop fully-empty trailing rows (the template has ~1000 rows in some sheets)
    df = df.dropna(how="all").reset_index(drop=True)

    if df.empty:
        return None

    # Auto-detect instruction sub-header row.
    # If the first cell is NOT a valid Excel column ID, it's an instruction row
    # like "Which column does the question appear in on the export spreadsheet?"
    # or "Very short title to appear in the dashboard".
    first_cell = str(df.iloc[0, 0]).strip()
    if not _EXCEL_COL_RE.match(first_cell):
        logger.info(
            "Detected instruction row in sheet '%s', skipping: '%s'",
            sheet_name, first_cell[:60],
        )
        df = df.iloc[1:].reset_index(drop=True)

    if df.empty:
        return None

    df.columns = column_names

    # Strip whitespace from column-ID fields — some files have "C " instead of "C"
    for col in column_names:
        if "column" in col.lower():
            df[col] = df[col].astype(str).str.strip()

    return df


def _parse_question_numbers(values: pd.Series) -> list[int] | None:
    """Try to parse question numbers from a Series. Returns list of ints or None if any fail."""
    parsed = []
    for val in values.astype(str):
        try:
            parsed.append(int(val.strip()))
        except ValueError:
            return None
    return parsed


def _extract_numbers(text: str) -> list[int]:
    """Extract all integers from a string."""
    return [int(x) for x in re.findall(r"\d+", str(text))]


def validate_data(
    question_sheets: dict[str, pd.DataFrame],
    original_headers: dict[str, str],
    responses_df: pd.DataFrame,
    demographic_columns: list[str] | None = None,
    demographic_labels: list[str] | None = None,
) -> None:
    """Validate Q.U. sheets against response data.

    Prints summaries of responses and Q.U. sheets, then checks for:
    - Q.U. columns that don't exist in the response data
    - More Q.U. columns referenced than response columns available
    - Low string similarity between Q.U. labels and response headers
    - Mismatched numbers extracted from Q.U. labels vs response headers
    - Demographic column value distributions
    If any issues found, prompts user to confirm before continuing.
    """
    issues: list[str] = []

    # ── Response summary ──────────────────────────────────────────────
    n_rows, n_cols = responses_df.shape
    resp_col_count = n_cols - 1  # exclude themefinder_id
    total_cells = n_rows * resp_col_count
    nan_count = int(responses_df.drop(columns=["themefinder_id"]).isna().sum().sum())
    nan_pct = (nan_count / total_cells * 100) if total_cells else 0
    print(f"\n  Response data: {n_rows} rows x {resp_col_count} cols, "
          f"{nan_count}/{total_cells} NaN ({nan_pct:.1f}%)")

    # ── Demographics ──────────────────────────────────────────────────
    if demographic_columns and demographic_labels:
        print(f"\n  Demographics ({len(demographic_columns)} field(s)):")
        for col_id, label in zip(demographic_columns, demographic_labels):
            if col_id not in responses_df.columns:
                print(f"    ✗ {label} (col {col_id}) — not found in response data")
                issues.append(f"Demographic column {col_id} ({label}) not in response data")
                continue
            series = responses_df[col_id].fillna("Not Provided")
            n_unique = series.nunique()
            n_missing = int((series == "Not Provided").sum())
            missing_pct = n_missing / n_rows * 100 if n_rows else 0
            print(f"    {label} (col {col_id}): {n_unique} unique, {n_missing} missing ({missing_pct:.1f}%)")
            for value, count in series.value_counts().head(5).items():
                pct = count / n_rows * 100
                print(f"      {value}: {count} ({pct:.1f}%)")

    # ── Q.U. summary ────────────────────────────────────────────────────
    all_qu_columns: set[str] = set()
    total_questions = 0
    print("\n  Question Understanding:")
    for sheet_key, df in question_sheets.items():
        n_questions = len(df)
        total_questions += n_questions
        cols_in_sheet: set[str] = set()
        for col_field in COL_ID_FIELDS[sheet_key]:
            cols_in_sheet.update(df[col_field].astype(str).str.strip().tolist())
        all_qu_columns.update(cols_in_sheet)
        print(f"    {sheet_key}: {n_questions} question(s), {len(cols_in_sheet)} column(s)")
    print(f"    Total: {total_questions} question(s), {len(all_qu_columns)} column(s)")

    # ── Missing fields in Q.U. rows ─────────────────────────────────────
    incomplete_rows: list[str] = []
    for sheet_key, df in question_sheets.items():
        sheet_name = QU_SHEET_SPECS[sheet_key][0]
        expected_fields = QU_SHEET_SPECS[sheet_key][2]
        for idx, row in df.iterrows():
            missing = [
                f for f in expected_fields
                if pd.isna(row.get(f)) or str(row.get(f)).strip() in ("", "nan")
            ]
            if missing:
                q_text = str(row.get("question_text", "")).strip()
                label = q_text[:50] + "..." if len(q_text) > 50 else q_text
                incomplete_rows.append(
                    f"      \"{sheet_name}\" row {idx + 1}: missing {', '.join(missing)}"
                    f"  ({label!r})" if label else ""
                )
    if incomplete_rows:
        print("\n  ⚠ Q.U. rows with missing fields:")
        for line in incomplete_rows:
            print(line)
        issues.append(f"Incomplete Q.U. rows: {len(incomplete_rows)}")

    # ── Column ID format check ────────────────────────────────────────
    bad_col_ids: list[str] = []
    for sheet_key, df in question_sheets.items():
        for col_field in COL_ID_FIELDS[sheet_key]:
            for _, row in df.iterrows():
                val = str(row[col_field]).strip()
                if not _EXCEL_COL_RE.match(val):
                    bad_col_ids.append(f"      {sheet_key}.{col_field} = {val!r}")
    if bad_col_ids:
        print("\n  ⚠ Column IDs that don't look like Excel columns (expected A, B, AA, ...):")
        for line in bad_col_ids:
            print(line)
        issues.append("Non-Excel column IDs found in Q.U. sheets")

    # ── Question number format check ──────────────────────────────────
    bad_qnums: list[str] = []
    for sheet_key, df in question_sheets.items():
        for val in df["question_number"].astype(str):
            try:
                int(val.strip())
            except ValueError:
                bad_qnums.append(f"      {sheet_key}: {val!r}")
    if bad_qnums:
        print("\n  ⚠ Non-integer question numbers (column ID-based fallback will be used):")
        for line in bad_qnums:
            print(line)
        issues.append("Non-integer question numbers in Q.U. sheets")

    # ── Duplicate column reference check ─────────────────────────────────
    # Hybrid questions legitimately use two columns (open + closed) for the
    # same question, so we track (col_id, q_num) pairs and only flag a column
    # when it's referenced by more than one distinct question.
    col_refs: list[tuple[str, str, int]] = []  # (col_id, sheet_key, q_num)
    for sheet_key, df in question_sheets.items():
        if sheet_key == "hybrid":
            continue
        for _, row in df.iterrows():
            q_num = row["question_number"]
            for col_field in COL_ID_FIELDS[sheet_key]:
                col_refs.append((str(row[col_field]).strip(), sheet_key, q_num))
    seen: dict[str, list[tuple[str, int]]] = {}
    for col_id, sheet_key, q_num in col_refs:
        seen.setdefault(col_id, []).append((sheet_key, q_num))
    dupes = {col_id: refs for col_id, refs in seen.items() if len(refs) > 1}
    if dupes:
        print("\n  ⚠ Columns referenced more than once across Q.U. sheets:")
        for col_id, refs in sorted(dupes.items(), key=lambda x: excel_column_to_number(x[0])):
            ref_strs = [f"Q{q_num} ({sheet_key})" for sheet_key, q_num in refs]
            print(f"      Column {col_id}: {', '.join(ref_strs)}")
        issues.append(f"Duplicate column references: {', '.join(sorted(dupes, key=excel_column_to_number))}")

    # ── Column existence check ────────────────────────────────────────
    if len(all_qu_columns) > resp_col_count:
        msg = (f"Q.U. references {len(all_qu_columns)} columns but response data only has {resp_col_count}")
        print(f"\n  ⚠ {msg}")
        issues.append(msg)

    max_resp_col = max(original_headers, key=excel_column_to_number) if original_headers else "?"
    missing_cols = sorted(
        [c for c in all_qu_columns if c not in original_headers],
        key=excel_column_to_number,
    )
    if missing_cols:
        print(f"\n  ⚠ Q.U. references columns not in response data (max is {max_resp_col}):")
        for col_id in missing_cols:
            print(f"      {col_id}")
        issues.append(f"Q.U. references missing columns: {', '.join(missing_cols)}")

    # ── Column data type checks ─────────────────────────────────────────
    # Multichoice columns should have few unique values and few distinct
    # options after comma-splitting. Free-text columns should have high
    # uniqueness. These checks catch misclassified question types.
    MAX_EXPECTED_OPTIONS = 10
    CLOSED_UNIQUENESS_THRESHOLD = 0.3
    OPEN_UNIQUENESS_THRESHOLD = 0.3

    def _check_looks_like_multichoice(
        col_id: str, q_num: int, sheet_key: str, col_role: str,
    ) -> None:
        """Warn if a column expected to be multichoice looks like free text."""
        if col_id not in responses_df.columns:
            return
        series = responses_df[col_id].dropna().astype(str)
        n_responses = len(series)
        if n_responses == 0:
            return
        sheet_name = QU_SHEET_SPECS[sheet_key][0]

        # Gather both signals, report as one warning
        problems: list[str] = []

        all_options: set[str] = set()
        for cell in series:
            all_options.update(item.strip() for item in cell.split(","))
        if len(all_options) > MAX_EXPECTED_OPTIONS:
            problems.append(f"{len(all_options)} unique options after comma-split (expected ≤{MAX_EXPECTED_OPTIONS})")

        n_unique = series.nunique()
        ratio = n_unique / n_responses
        if ratio > CLOSED_UNIQUENESS_THRESHOLD:
            problems.append(f"{n_unique}/{n_responses} responses are unique ({ratio:.0%}, expected ≤{CLOSED_UNIQUENESS_THRESHOLD:.0%})")

        if problems:
            msg = (
                f"Column {col_id} (Q{q_num}, {col_role}) — on Q.U. sheet \"{sheet_name}\" — "
                f"looks like free text, not multichoice: {'; '.join(problems)}"
            )
            print(f"\n  ⚠ {msg}")
            issues.append(msg)

    def _check_looks_like_free_text(
        col_id: str, q_num: int, sheet_key: str, col_role: str,
    ) -> None:
        """Warn if a column expected to be free text looks like multichoice."""
        if col_id not in responses_df.columns:
            return
        series = responses_df[col_id].dropna().astype(str)
        n_responses = len(series)
        if n_responses == 0:
            return
        sheet_name = QU_SHEET_SPECS[sheet_key][0]
        n_unique = series.nunique()
        ratio = n_unique / n_responses
        if ratio < OPEN_UNIQUENESS_THRESHOLD:
            msg = (
                f"Column {col_id} (Q{q_num}, {col_role}) — on Q.U. sheet \"{sheet_name}\" — "
                f"only {n_unique}/{n_responses} responses are unique ({ratio:.0%}) — "
                f"expected >{OPEN_UNIQUENESS_THRESHOLD:.0%} for free text. "
                f"Should this be on the \"Multiple Choice\" sheet instead?"
            )
            print(f"\n  ⚠ {msg}")
            issues.append(msg)

    for sheet_key in ("closed", "hybrid", "open"):
        df = question_sheets.get(sheet_key)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            q_num = row["question_number"]
            if sheet_key == "closed":
                _check_looks_like_multichoice(
                    str(row["column_name"]).strip(), q_num, sheet_key, "closed column",
                )
            elif sheet_key == "hybrid":
                _check_looks_like_multichoice(
                    str(row["closed_column"]).strip(), q_num, sheet_key, "closed part",
                )
                _check_looks_like_free_text(
                    str(row["open_column"]).strip(), q_num, sheet_key, "open part",
                )
            elif sheet_key == "open":
                _check_looks_like_free_text(
                    str(row["column_name"]).strip(), q_num, sheet_key, "open column",
                )

    # ── Label matching ────────────────────────────────────────────────
    label_issues: list[tuple[str, list[str], str, str]] = []
    for sheet_key, df in question_sheets.items():
        for _, row in df.iterrows():
            for col_field in COL_ID_FIELDS[sheet_key]:
                col_id = str(row[col_field]).strip()
                qu_label = str(row.get("question_text", "")).strip()
                resp_header = original_headers.get(col_id)

                if resp_header is None:
                    continue  # already reported above

                ratio = SequenceMatcher(None, qu_label.lower(), resp_header.lower(), autojunk=False).ratio()
                qu_nums = set(_extract_numbers(qu_label))
                resp_nums = set(_extract_numbers(resp_header))

                problems: list[str] = []
                if ratio < 0.4:
                    problems.append(f"low similarity ({ratio:.0%})")
                if qu_nums and resp_nums and qu_nums != resp_nums:
                    problems.append(f"number mismatch (Q.U.:{sorted(qu_nums)} vs resp:{sorted(resp_nums)})")

                if problems:
                    label_issues.append((col_id, problems, qu_label, resp_header))

    if label_issues:
        print(f"\n  ⚠ Label mismatches ({len(label_issues)}):")
        for col_id, problems, qu_label, resp_header in label_issues:
            print(f"\n      col {col_id}: {', '.join(problems)}")
            print(f"      ┌ Q.U.:   {qu_label}")
            print(f"      └ Resp: {resp_header}")
        issues.extend(["Label mismatch"] * len(label_issues))

    # ── Unreferenced columns check ──────────────────────────────────────
    # Flag response columns not referenced by any question or demographic.
    demographic_set = set(demographic_columns) if demographic_columns else set()
    all_resp_cols = set(responses_df.columns) - {"themefinder_id"}
    referenced_cols = all_qu_columns | demographic_set
    unreferenced = sorted(
        all_resp_cols - referenced_cols,
        key=excel_column_to_number,
    )
    if unreferenced:
        print(f"\n  ⚠ {len(unreferenced)} response column(s) not referenced by any question or demographic:")
        for col_id in unreferenced:
            series = responses_df[col_id].dropna().astype(str)
            n_responses = len(series)
            n_unique = series.nunique()
            header = original_headers.get(col_id, "?")
            if len(header) > 60:
                header = header[:57] + "..."
            print(f"      {col_id}: \"{header}\" — {n_responses} non-null, {n_unique} unique")
        print("      Should any of these be included as demographics?")
        issues.append(f"Unreferenced columns: {', '.join(unreferenced)}")

    # ── Result ────────────────────────────────────────────────────────
    if issues:
        print(f"\n  Found {len(issues)} issue(s).")
        answer = input("  Continue anyway? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborting.")
            sys.exit(1)
    else:
        print("\n  ✓ Validation passed.")


def load_and_number_question_sheets(
    question_understanding_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load all question sheets, truncate to useful columns, and assign question numbers.

    If any question_number value across any sheet cannot be parsed as an integer,
    falls back to numbering all questions sequentially by sorting their Excel column
    IDs across all sheets.
    """
    sheets: dict[str, pd.DataFrame] = {}

    for sheet_key, (sheet_name, ncols, col_names) in QU_SHEET_SPECS.items():
        df = _load_qu_sheet(question_understanding_path, sheet_name, ncols, col_names)
        if df is not None:
            sheets[sheet_key] = df

    if not sheets:
        return sheets

    # Check whether all question_number values can be parsed as integers
    needs_fallback = False
    for df in sheets.values():
        if _parse_question_numbers(df["question_number"]) is None:
            needs_fallback = True
            break

    if needs_fallback:
        # Collect (excel_col_id, sheet_key, df_index) from every row across all sheets
        all_entries: list[tuple[str, str, int]] = []
        for key, df in sheets.items():
            for idx in df.index:
                all_entries.append((str(df.at[idx, PRIMARY_COL_ID_FIELD[key]]).strip(), key, idx))

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

    # Show final numbering, sorted by question number
    all_rows: list[tuple[int, str, str, str]] = []
    for key, df in sheets.items():
        for _, row in df.iterrows():
            col = str(row.get(PRIMARY_COL_ID_FIELD[key], "?")).strip()
            label = str(row.get("question_text", ""))
            if len(label) > 120:
                label = label[:117] + "..."
            all_rows.append((row["question_number"], col, key, label))
    all_rows.sort()
    print("\n  Question numbering:")
    for q_num, col, key, label in all_rows:
        print(f"    Q{q_num:>3}  col {col:<4}  [{key}]  {label}")

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
        raise ValueError(f"Non-unique question numbers found across sheets: {detail}")

    return sheets


def create_respondents_jsonl(
    df: pd.DataFrame,
    demographic_columns: list[str],
    demographic_labels: list[str],
    output_dir: Path,
) -> None:
    for c in demographic_columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace("_x000D_", "", regex=False)
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )
        df[c] = df[c].apply(lambda x: x.split(","))
    df.rename(columns=dict(zip(demographic_columns, demographic_labels)), inplace=True)
    df["demographic_data"] = df[demographic_labels].to_dict(orient="records")
    df[["themefinder_id", "demographic_data"]].to_json(
        output_dir / "respondents.jsonl", orient="records", lines=True
    )


def _clean_text_column(series: pd.Series) -> pd.Series:
    """Remove unwanted characters and non-ASCII bytes from a text column."""
    series = series.astype(str).str.encode("ascii", "ignore").str.decode("ascii")
    for bad_string in CHARACTERS_TO_REMOVE:
        series = series.apply(lambda x, bs=bad_string: x.replace(bs, " "))
    return series


def create_question_inputs(
    df: pd.DataFrame,
    questions: list[dict],
    question_type: str,
    output_dir: Path,
    sample_size: int | None = None,
) -> None:
    """Write responses.jsonl, multi_choice.jsonl, and question.json for each question."""
    has_free_text = question_type in ("open", "hybrid")
    has_options = question_type in ("closed", "hybrid")

    for question in questions:
        q_num = question["question_number"]
        q_dir = output_dir / f"question_part_{q_num}"
        q_dir.mkdir(parents=True, exist_ok=True)

        # Select relevant columns and drop empty rows
        if question_type == "hybrid":
            open_col = question["open_column"]
            closed_col = question["closed_column"]
            data_cols = [closed_col, open_col]
            answers = df[["themefinder_id"] + data_cols].dropna(subset=data_cols, how="all")
            answers[closed_col] = answers[closed_col].fillna("Not Provided")
            answers[open_col] = answers[open_col].fillna("Not Provided")
        else:
            col = question["column_name"]
            data_cols = [col]
            answers = df[["themefinder_id"] + data_cols].dropna()

        if sample_size is not None and sample_size < len(answers):
            answers = answers.sample(sample_size)

        # Clean text columns
        for c in data_cols:
            answers[c] = _clean_text_column(answers[c])

        # Write multi_choice.jsonl
        if has_options:
            options_col = question["closed_column"] if question_type == "hybrid" else question["column_name"]
            answers[options_col] = answers[options_col].apply(lambda x: x.split(","))
            answers.rename(columns={options_col: "options"}, inplace=True)
            answers[["themefinder_id", "options"]].to_json(
                q_dir / "multi_choice.jsonl", orient="records", lines=True
            )

        # Write responses.jsonl
        if has_free_text:
            text_col = question["open_column"] if question_type == "hybrid" else question["column_name"]
            # For hybrid, text_col is already the original name (not renamed)
            answers.rename(columns={text_col: "text"}, inplace=True)
            answers[["themefinder_id", "text"]].to_json(
                q_dir / "responses.jsonl", orient="records", lines=True
            )

        # Write question.json
        question_data: dict = {
            "question_number": q_num,
            "question_text": question["question_text"],
            "has_free_text": has_free_text,
        }
        if has_options:
            question_data["multi_choice_options"] = sorted(
                set(item for sublist in answers["options"] for item in sublist)
            )
        with open(q_dir / "question.json", "w") as f:
            json.dump(question_data, f, indent=4)


# --- CLI logic ---


def find_data_files(consultation_dir: Path) -> list[Path]:
    """Find CSV and Excel files in the consultation directory, ignoring temp files."""
    files = []
    for f in consultation_dir.iterdir():
        if f.name.startswith("~$"):
            continue
        if f.suffix.lower() in VALID_EXTENSIONS:
            files.append(f)
    return sorted(files)


def _is_subheader_row(row: pd.Series) -> bool:
    """Detect whether a row is a descriptive sub-header rather than data.

    Two-tier header files have short IDs on row 0 and long question text on
    row 1.  Data rows have shorter, more varied values with more NaN cells.
    """
    non_null_1 = row.dropna()
    if len(non_null_1) == 0:
        return False

    # Sub-header rows are nearly fully populated (>90% non-null)
    fill_ratio = len(non_null_1) / len(row)
    if fill_ratio < 0.9:
        return False

    # Sub-header rows have long descriptive text (median length > 25 chars)
    median_len = non_null_1.astype(str).str.len().median()
    return median_len > 25


def load_responses(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load responses from CSV or Excel file.

    Returns the DataFrame (with columns renamed to Excel letters) and a
    dict mapping Excel column letter -> original column header string.

    Handles three layout patterns automatically:
      - Single header row, then data  (LGR files)
      - Header row, blank row, then data  (SCR files)
      - Two header rows (short IDs + full question text), then data  (Biomass)
    """
    ext = path.suffix.lower()
    read_fn = pd.read_csv if ext == ".csv" else pd.read_excel

    # Read first 3 rows raw to detect layout
    raw = read_fn(path, header=None, nrows=3)

    header_row = 0

    if len(raw) > 1 and _is_subheader_row(raw.iloc[1]):
        # Two-tier header — use row 1 (full question text) as the header
        logger.info("Detected two-tier header in %s, using row 2 as header", path.name)
        header_row = 1

    # Full read with detected layout
    df = read_fn(path, header=header_row)

    # Drop all-NaN rows (handles blank separator rows and trailing empties)
    df = df.dropna(how="all").reset_index(drop=True)

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
    responses_path: Path, question_understanding_path: Path, output_dir: Path
) -> None:
    """Run the full ingestion pipeline: load → validate → ingest."""

    # ── Phase 1: Load ─────────────────────────────────────────────────
    print(f"\nLoading responses from: {responses_path.name}")
    responses_df, original_headers = load_responses(responses_path)
    print(f"  Loaded {len(responses_df)} responses")

    question_sheets = load_and_number_question_sheets(question_understanding_path)

    demographic_columns: list[str] | None = None
    demographic_labels: list[str] | None = None
    demographic_info = _load_qu_sheet(
        question_understanding_path, "Demographic", 2, ["column_id", "label"]
    )
    if demographic_info is not None:
        demographic_columns = demographic_info["column_id"].tolist()
        demographic_labels = [
            label.replace("/", "-") for label in demographic_info["label"].tolist()
        ]

    # ── Phase 2: Validate ─────────────────────────────────────────────
    validate_data(
        question_sheets, original_headers, responses_df,
        demographic_columns, demographic_labels,
    )

    # ── Phase 3: Ingest ───────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Demographics
    if demographic_columns and demographic_labels:
        print("Writing demographics...")
        for c in demographic_columns:
            responses_df[c] = responses_df[c].fillna("Not Provided")
            responses_df[c] = responses_df[c].apply(
                lambda x: "Other" if isinstance(x, str) and "Other" in x else x
            )
        create_respondents_jsonl(
            responses_df, demographic_columns, demographic_labels, output_dir
        )
    else:
        print("  No demographic data found, skipping.")

    # Questions
    for qtype in ("open", "hybrid", "closed"):
        q_df = question_sheets.get(qtype)
        if q_df is None or q_df.empty:
            print(f"  No {qtype} questions found, skipping.")
            continue
        # Skip open questions where the response column is entirely NaN
        if qtype == "open":
            all_nan = responses_df[q_df["column_name"].tolist()].isna().all()
            q_df = q_df[~q_df["column_name"].isin(all_nan[all_nan].index)]
        print(f"Writing {qtype} questions...")
        create_question_inputs(
            responses_df, q_df.to_dict(orient="records"), qtype, output_dir
        )

    print(f"\nAll input files written to: {output_dir}")


def upload_inputs_to_s3(local_dir: Path, bucket: str, s3_prefix: str) -> None:
    """Upload all files in local_dir to s3://bucket/s3_prefix, preserving directory structure.

    Checks for existing objects at the S3 prefix before uploading. If any exist,
    warns and requires confirmation. Always prompts before uploading.
    """
    s3 = boto3.client("s3")
    files = [f for f in local_dir.rglob("*") if f.is_file()]
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
        relative = file_path.relative_to(local_dir)
        print(f"  {relative}")
    answer = input("Proceed with upload? (y/n): ").strip().lower()
    if answer != "y":
        print("Upload skipped.")
        return

    for file_path in files:
        relative = file_path.relative_to(local_dir)
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
    parser.add_argument(
        "--dir", type=Path, help="Path to consultation directory (skip interactive prompt)"
    )
    parser.add_argument(
        "--responses", type=Path, help="Path to response data file (skip file selection)"
    )
    parser.add_argument(
        "--qu", type=Path, help="Path to question understanding file (skip file selection)"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload inputs to S3 after ingestion"
    )
    args = parser.parse_args()

    name = args.name
    if not name and args.dir:
        # Derive name from directory
        name = args.dir.resolve().name
    if not name:
        name = input("Enter consultation name: ").strip()
        if not name:
            print("Error: consultation name cannot be empty.")
            sys.exit(1)

    name = to_snake_case(name)
    print(f"Using consultation name: {name}")

    base_dir = Path(__file__).resolve().parent / "consultations"

    # Step 1: Resolve consultation directory
    if args.dir:
        consultation_dir = args.dir.resolve()
    else:
        consultation_dir = base_dir / name
    consultation_dir.mkdir(parents=True, exist_ok=True)
    print(f"Consultation directory: {consultation_dir}")

    # Step 2: Resolve file paths
    if args.responses and args.qu:
        responses_path = args.responses.resolve()
        qu_path = args.qu.resolve()
        for label, path in [("Responses", responses_path), ("Q.U.", qu_path)]:
            if not path.exists():
                print(f"Error: {label} file not found: {path}")
                sys.exit(1)
        print(f"Responses file: {responses_path.name}")
        print(f"Q.U. file: {qu_path.name}")
    else:
        if not (args.responses or args.qu):
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

        if args.responses:
            responses_path = args.responses.resolve()
        else:
            responses_path = prompt_file_selection(files, "consultation response data")

        if args.qu:
            qu_path = args.qu.resolve()
        else:
            remaining = [f for f in files if f != responses_path]
            if len(remaining) == 1:
                qu_path = remaining[0]
                print(f"\nUsing '{qu_path.name}' as the template question understanding file.")
            else:
                qu_path = prompt_file_selection(
                    remaining, "template question understanding data"
                )

    # Step 3: Run ingestion
    output_dir = consultation_dir / "inputs"
    run_ingestion(responses_path, qu_path, output_dir)

    # Step 4: Upload inputs to S3 (only if --upload flag is set)
    if args.upload:
        s3_prefix = f"app_data/consultations/{name}/inputs/"
        upload_inputs_to_s3(output_dir, "i-dot-ai-prod-consult-data", s3_prefix)

    # Step 5: Point to Confluence
    print("\n" + "=" * 60)
    print("Setup complete! For further instructions, see:")
    print(f"  {CONFLUENCE_URL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
