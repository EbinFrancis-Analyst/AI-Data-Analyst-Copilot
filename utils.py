"""
utils.py
Shared utility functions for the Data Cleaning Tool
"""

import pandas as pd
import numpy as np
import io
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────

def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a CSV or Excel file from a Streamlit UploadedFile object.

    Returns:
        (DataFrame, None) on success
        (None, error_message) on failure
    """
    if uploaded_file is None:
        return None, "No file provided."

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            df = _try_csv_encodings(uploaded_file)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="xlrd")
        else:
            return None, f"Unsupported file format: '{uploaded_file.name}'. Please upload CSV or Excel."

        if df is None or df.empty:
            return None, "The uploaded file is empty or could not be parsed."

        # Strip column name whitespace
        df.columns = [str(c).strip() for c in df.columns]

        # Drop fully unnamed columns that look like index leakage
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed: \d+$')]

        return df, None

    except MemoryError:
        return None, "File is too large to process in memory. Please upload a smaller file."
    except Exception as e:
        return None, f"Failed to read file: {str(e)}"


def _try_csv_encodings(uploaded_file) -> pd.DataFrame:
    """Try multiple encodings when reading CSV."""
    content = uploaded_file.read()
    for enc in ("utf-8", "latin-1", "cp1252", "utf-16"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("Could not decode the CSV file with any supported encoding.")


# ─────────────────────────────────────────────
# DATA QUALITY SCORE
# ─────────────────────────────────────────────

def compute_quality_score(profile: dict, issues: list) -> Tuple[int, str, str]:
    """
    Compute a 0–100 data quality score.

    Returns: (score, label, colour)
    """
    rows, cols = profile["shape"]
    if rows == 0:
        return 0, "No Data", "red"

    total_cells = rows * cols
    missing_pct = profile["total_missing"] / total_cells if total_cells > 0 else 0
    dup_pct = profile["duplicate_rows"] / rows if rows > 0 else 0

    # Penalty weights
    score = 100
    score -= missing_pct * 40      # up to -40 for missing data
    score -= dup_pct * 20          # up to -20 for duplicates

    high_issues = sum(1 for i in issues if i["severity"] == "High")
    med_issues = sum(1 for i in issues if i["severity"] == "Medium")
    score -= high_issues * 5
    score -= med_issues * 2

    score = max(0, min(100, round(score)))

    if score >= 80:
        label, colour = "Good 🟢", "green"
    elif score >= 60:
        label, colour = "Fair 🟡", "orange"
    elif score >= 40:
        label, colour = "Poor 🟠", "darkorange"
    else:
        label, colour = "Critical 🔴", "red"

    return score, label, colour


# ─────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 encoded CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes (in memory)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return buffer.getvalue()


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def severity_colour(severity: str) -> str:
    """Return a CSS-friendly colour for a severity level."""
    return {"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#2196F3"}.get(severity, "#888")


def format_profile_table(profile: dict) -> pd.DataFrame:
    """Convert the column profile dict to a display-ready DataFrame."""
    rows = []
    for col, cp in profile["columns"].items():
        rows.append({
            "Column": col,
            "Raw Type": cp["dtype_raw"],
            "Inferred Type": cp["inferred_type"],
            "Missing": cp["missing_count"],
            "Missing %": f"{cp['missing_pct']}%",
            "Unique": cp["unique_count"],
            "Completeness %": f"{cp['completeness_pct']}%",
            "Cardinality": cp["cardinality"],
        })
    return pd.DataFrame(rows)


def format_issues_table(issues: list) -> pd.DataFrame:
    """Convert the issues list to a display-ready DataFrame."""
    if not issues:
        return pd.DataFrame(columns=["Column", "Issue Type", "Severity", "Description", "Affected"])
    return pd.DataFrame([{
        "Column": i["column"],
        "Issue Type": i["issue_type"],
        "Severity": i["severity"],
        "Description": i["description"],
        "Affected": i["affected_count"],
    } for i in issues])
