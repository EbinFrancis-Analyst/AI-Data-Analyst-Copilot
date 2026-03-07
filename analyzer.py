"""
analyzer.py
Data Profiling Engine and Quality Issue Detector
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any


# ─────────────────────────────────────────────
# DATA TYPE INFERENCE
# ─────────────────────────────────────────────

def infer_column_type(series: pd.Series) -> str:
    """
    Intelligently infer the semantic type of a column.

    Returns one of: 'boolean', 'datetime', 'integer', 'float',
                    'categorical', 'string'
    """
    s = series.dropna()
    if s.empty:
        return "string"

    # Boolean check
    bool_values = {"yes", "no", "true", "false", "1", "0", "y", "n"}
    if s.astype(str).str.strip().str.lower().isin(bool_values).all():
        return "boolean"

    # Datetime check
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if s.dtype == object:
        try:
            parsed = pd.to_datetime(s.astype(str).str.strip(), infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() / len(s) >= 0.7:
                return "datetime"
        except Exception:
            pass

    # Numeric checks
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if s.dtype == object:
        numeric_converted = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
        ratio = numeric_converted.notna().sum() / len(s)
        if ratio >= 0.8:
            all_int = numeric_converted.dropna().apply(lambda x: float(x).is_integer()).all()
            return "integer" if all_int else "float"

    # Categorical vs string
    if s.dtype == object:
        unique_ratio = s.nunique() / len(s)
        if unique_ratio < 0.2 or s.nunique() <= 20:
            return "categorical"
        return "string"

    return "string"


# ─────────────────────────────────────────────
# DATA PROFILING
# ─────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a full data quality profile for the dataset.

    Returns a dict with column-level and dataset-level metrics.
    """
    profile = {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isnull().sum().sum()),
        "columns": {}
    }

    for col in df.columns:
        series = df[col]
        inferred_type = infer_column_type(series)
        missing = int(series.isnull().sum())
        total = len(series)
        completeness = round((total - missing) / total * 100, 1) if total > 0 else 0.0

        col_profile = {
            "dtype_raw": str(series.dtype),
            "inferred_type": inferred_type,
            "missing_count": missing,
            "missing_pct": round(missing / total * 100, 1) if total > 0 else 0.0,
            "unique_count": int(series.nunique()),
            "completeness_pct": completeness,
            "cardinality": int(series.nunique()),
            "is_constant": int(series.nunique(dropna=False)) <= 1,
            "is_empty": missing == total,
        }

        # Numeric stats
        if inferred_type in ("integer", "float"):
            numeric_s = pd.to_numeric(series, errors="coerce")
            col_profile["mean"] = round(float(numeric_s.mean()), 4) if numeric_s.notna().any() else None
            col_profile["median"] = round(float(numeric_s.median()), 4) if numeric_s.notna().any() else None
            col_profile["std"] = round(float(numeric_s.std()), 4) if numeric_s.notna().any() else None
            col_profile["min"] = float(numeric_s.min()) if numeric_s.notna().any() else None
            col_profile["max"] = float(numeric_s.max()) if numeric_s.notna().any() else None
        else:
            col_profile["mean"] = col_profile["median"] = col_profile["std"] = None
            col_profile["min"] = col_profile["max"] = None

        profile["columns"][col] = col_profile

    return profile


# ─────────────────────────────────────────────
# ISSUE DETECTION
# ─────────────────────────────────────────────

def detect_issues(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect a comprehensive list of data quality issues.

    Returns a list of issue dicts with keys:
      column, issue_type, severity, description, affected_count
    """
    issues = []

    # Dataset-level issues
    dupes = profile["duplicate_rows"]
    if dupes > 0:
        issues.append({
            "column": "(dataset)",
            "issue_type": "Duplicate Rows",
            "severity": "High" if dupes > 10 else "Medium",
            "description": f"{dupes} duplicate rows found.",
            "affected_count": dupes,
        })

    for col, cp in profile["columns"].items():
        series = df[col]

        # Empty column
        if cp["is_empty"]:
            issues.append(_issue(col, "Empty Column", "High",
                                 "Column contains no data at all.", len(series)))
            continue

        # Constant column
        if cp["is_constant"] and not cp["is_empty"]:
            issues.append(_issue(col, "Constant Column", "Medium",
                                 "Column has only one unique value — likely useless.", len(series)))

        # Missing values
        if cp["missing_count"] > 0:
            sev = "High" if cp["missing_pct"] > 30 else ("Medium" if cp["missing_pct"] > 5 else "Low")
            issues.append(_issue(col, "Missing Values", sev,
                                 f"{cp['missing_count']} missing values ({cp['missing_pct']}%).",
                                 cp["missing_count"]))

        # Whitespace issues (string/categorical only)
        if cp["inferred_type"] in ("string", "categorical"):
            s_str = series.dropna().astype(str)
            whitespace_count = (s_str != s_str.str.strip()).sum()
            if whitespace_count > 0:
                issues.append(_issue(col, "Leading/Trailing Whitespace", "Low",
                                     f"{whitespace_count} values have extra whitespace.",
                                     whitespace_count))

        # Mixed datatypes / numeric stored as string
        if series.dtype == object and cp["inferred_type"] in ("integer", "float"):
            issues.append(_issue(col, "Numeric Stored as String", "High",
                                 "Column contains numeric values stored as text.",
                                 cp["missing_count"] or 1))

        # Date stored as text
        if series.dtype == object and cp["inferred_type"] == "datetime":
            issues.append(_issue(col, "Date Stored as Text", "Medium",
                                 "Column appears to contain dates stored as strings.",
                                 len(series.dropna())))

        # Boolean inconsistencies
        if cp["inferred_type"] == "boolean":
            raw_vals = series.dropna().astype(str).str.strip().str.lower().unique()
            if len(raw_vals) > 2:
                issues.append(_issue(col, "Boolean Inconsistency", "Medium",
                                     f"Multiple boolean representations found: {list(raw_vals)[:6]}",
                                     len(series.dropna())))

        # Category inconsistencies (case/spacing)
        if cp["inferred_type"] == "categorical":
            s_lower = series.dropna().astype(str).str.strip().str.lower()
            s_raw = series.dropna().astype(str).str.strip()
            if s_lower.nunique() < s_raw.nunique():
                issues.append(_issue(col, "Category Inconsistency", "Medium",
                                     "Mixed casing or spacing creates duplicate categories.",
                                     s_raw.nunique() - s_lower.nunique()))

        # High cardinality categorical
        if cp["inferred_type"] == "categorical" and cp["cardinality"] > 50:
            issues.append(_issue(col, "High Cardinality Categorical", "Low",
                                 f"{cp['cardinality']} unique categories — consider grouping.",
                                 cp["cardinality"]))

        # Outliers in numeric columns
        if cp["inferred_type"] in ("integer", "float"):
            numeric_s = pd.to_numeric(series, errors="coerce").dropna()
            outlier_count = _count_outliers(numeric_s)
            if outlier_count > 0:
                issues.append(_issue(col, "Outliers Detected (IQR)", "Medium",
                                     f"{outlier_count} outlier values detected via IQR method.",
                                     outlier_count))

        # Special / invisible characters
        if cp["inferred_type"] in ("string", "categorical"):
            s_str = series.dropna().astype(str)
            special_count = s_str.apply(lambda x: bool(re.search(r'[^\x20-\x7E]', x))).sum()
            if special_count > 0:
                issues.append(_issue(col, "Special/Invisible Characters", "Low",
                                     f"{special_count} values contain non-printable or special characters.",
                                     special_count))

        # Invalid date formats (only if inferred datetime but already object dtype)
        if cp["inferred_type"] == "datetime" and series.dtype == object:
            parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
            invalid = parsed.isna().sum() - series.isna().sum()
            if invalid > 0:
                issues.append(_issue(col, "Invalid Date Format", "Medium",
                                     f"{invalid} values could not be parsed as dates.",
                                     int(invalid)))

    return issues


def _issue(col, issue_type, severity, description, affected_count):
    return {
        "column": col,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "affected_count": int(affected_count),
    }


def _count_outliers(series: pd.Series) -> int:
    """Count outliers using the IQR method."""
    if len(series) < 4:
        return 0
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return 0
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return int(((series < lower) | (series > upper)).sum())


# ─────────────────────────────────────────────
# CLEANING SUGGESTIONS
# ─────────────────────────────────────────────

def generate_suggestions(df: pd.DataFrame, profile: Dict[str, Any],
                          issues: List[Dict]) -> List[Dict[str, Any]]:
    """
    Generate intelligent cleaning suggestions based on detected issues.

    Returns a list of suggestion dicts with keys:
      column, action, description, default_enabled
    """
    suggestions = []
    issue_types_by_col = {}
    for iss in issues:
        issue_types_by_col.setdefault(iss["column"], []).append(iss["issue_type"])

    # Dataset-level
    if "Duplicate Rows" in issue_types_by_col.get("(dataset)", []):
        suggestions.append(_suggestion(
            "(dataset)", "remove_duplicates",
            "Remove duplicate rows", True))

    for col, cp in profile["columns"].items():
        col_issues = issue_types_by_col.get(col, [])
        inferred = cp["inferred_type"]

        if cp["is_empty"]:
            suggestions.append(_suggestion(col, "drop_empty_column",
                                           f"Drop empty column '{col}'", True))
            continue

        if cp["is_constant"]:
            suggestions.append(_suggestion(col, "drop_constant_column",
                                           f"Drop constant column '{col}' (no variance)", False))

        if "Missing Values" in col_issues:
            if inferred in ("integer", "float"):
                suggestions.append(_suggestion(col, "fill_missing_median",
                                               f"Fill missing values in '{col}' with median", True))
                suggestions.append(_suggestion(col, "fill_missing_mean",
                                               f"Fill missing values in '{col}' with mean", False))
                suggestions.append(_suggestion(col, "fill_missing_zero",
                                               f"Fill missing values in '{col}' with 0", False))
            elif inferred == "datetime":
                suggestions.append(_suggestion(col, "drop_missing_rows",
                                               f"Drop rows where '{col}' is missing", False))
            elif inferred == "boolean":
                suggestions.append(_suggestion(col, "fill_missing_mode",
                                               f"Fill missing '{col}' with most frequent value", True))
            else:
                suggestions.append(_suggestion(col, "fill_missing_unknown",
                                               f"Fill missing text in '{col}' with 'Unknown'", True))
                suggestions.append(_suggestion(col, "fill_missing_notprovided",
                                               f"Fill missing text in '{col}' with 'Not_Provided'", False))

        if "Leading/Trailing Whitespace" in col_issues:
            suggestions.append(_suggestion(col, "trim_whitespace",
                                           f"Trim whitespace in '{col}'", True))

        if "Numeric Stored as String" in col_issues:
            suggestions.append(_suggestion(col, "convert_to_numeric",
                                           f"Convert '{col}' from string to numeric", True))

        if "Date Stored as Text" in col_issues or "Invalid Date Format" in col_issues:
            suggestions.append(_suggestion(col, "convert_to_datetime",
                                           f"Convert '{col}' to datetime format", True))

        if "Boolean Inconsistency" in col_issues:
            suggestions.append(_suggestion(col, "standardize_boolean",
                                           f"Standardize boolean values in '{col}' to True/False", True))

        if "Category Inconsistency" in col_issues:
            suggestions.append(_suggestion(col, "standardize_categories",
                                           f"Standardize category casing in '{col}'", True))

        if "Special/Invisible Characters" in col_issues:
            suggestions.append(_suggestion(col, "remove_special_chars",
                                           f"Remove special/invisible characters from '{col}'", True))

        if "Outliers Detected (IQR)" in col_issues:
            suggestions.append(_suggestion(col, "cap_outliers",
                                           f"Cap outliers in '{col}' to IQR bounds", True))
            suggestions.append(_suggestion(col, "remove_outliers",
                                           f"Remove rows with outliers in '{col}'", False))

        if inferred in ("string", "categorical") and "Missing Values" not in col_issues:
            suggestions.append(_suggestion(col, "standardize_case",
                                           f"Standardize text casing in '{col}' to Title Case", False))

    return suggestions


def _suggestion(col, action, description, default_enabled):
    return {
        "column": col,
        "action": action,
        "description": description,
        "enabled": default_enabled,
    }
