"""
cleaner.py
Intelligent Cleaning Execution Engine
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple


# ─────────────────────────────────────────────
# MAIN CLEANING ORCHESTRATOR
# ─────────────────────────────────────────────

def apply_cleaning(df: pd.DataFrame,
                   selected_suggestions: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply selected cleaning suggestions to the DataFrame.

    Returns:
        cleaned_df: The cleaned DataFrame
        report: A dict summarising every cleaning action taken
    """
    df_clean = df.copy()
    report = {
        "rows_before": len(df),
        "rows_after": len(df),
        "duplicates_removed": 0,
        "missing_fixed": 0,
        "columns_standardized": 0,
        "outliers_handled": 0,
        "columns_dropped": 0,
        "type_conversions": 0,
        "actions_log": [],
    }

    for suggestion in selected_suggestions:
        col = suggestion["column"]
        action = suggestion["action"]

        try:
            df_clean, report = _dispatch(df_clean, col, action, report)
        except Exception as e:
            report["actions_log"].append({
                "column": col,
                "action": action,
                "status": "❌ Failed",
                "detail": str(e),
            })

    report["rows_after"] = len(df_clean)
    return df_clean, report


# ─────────────────────────────────────────────
# ACTION DISPATCHER
# ─────────────────────────────────────────────

def _dispatch(df: pd.DataFrame, col: str, action: str,
              report: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Route cleaning action to the correct handler."""

    # ── Dataset-level ──────────────────────────────────────────────
    if action == "remove_duplicates":
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        report["duplicates_removed"] += removed
        _log(report, col, action, f"Removed {removed} duplicate rows.")

    # ── Column drops ──────────────────────────────────────────────
    elif action in ("drop_empty_column", "drop_constant_column"):
        if col in df.columns:
            df = df.drop(columns=[col])
            report["columns_dropped"] += 1
            _log(report, col, action, f"Dropped column '{col}'.")

    # ── Missing value fills ───────────────────────────────────────
    elif action == "fill_missing_median":
        df, n = _fill_numeric(df, col, "median")
        report["missing_fixed"] += n
        _log(report, col, action, f"Filled {n} missing values with median.")

    elif action == "fill_missing_mean":
        df, n = _fill_numeric(df, col, "mean")
        report["missing_fixed"] += n
        _log(report, col, action, f"Filled {n} missing values with mean.")

    elif action == "fill_missing_zero":
        df, n = _fill_numeric(df, col, "zero")
        report["missing_fixed"] += n
        _log(report, col, action, f"Filled {n} missing values with 0.")

    elif action == "fill_missing_unknown":
        df, n = _fill_text(df, col, "Unknown")
        report["missing_fixed"] += n
        _log(report, col, action, f"Filled {n} missing values with 'Unknown'.")

    elif action == "fill_missing_notprovided":
        df, n = _fill_text(df, col, "Not_Provided")
        report["missing_fixed"] += n
        _log(report, col, action, f"Filled {n} missing values with 'Not_Provided'.")

    elif action == "fill_missing_mode":
        if col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                n = df[col].isna().sum()
                df[col] = df[col].fillna(mode_val[0])
                report["missing_fixed"] += n
                _log(report, col, action, f"Filled {n} missing values with mode '{mode_val[0]}'.")

    elif action == "drop_missing_rows":
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            n = before - len(df)
            report["missing_fixed"] += n
            _log(report, col, action, f"Dropped {n} rows with missing '{col}'.")

    # ── Text cleaning ─────────────────────────────────────────────
    elif action == "trim_whitespace":
        if col in df.columns:
            n = (df[col].astype(str) != df[col].astype(str).str.strip()).sum()
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            report["columns_standardized"] += 1
            _log(report, col, action, f"Trimmed whitespace in {n} values.")

    elif action == "remove_special_chars":
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: re.sub(r'[^\x20-\x7E]', '', str(x)).strip()
                if pd.notna(x) else x
            )
            report["columns_standardized"] += 1
            _log(report, col, action, "Removed special/invisible characters.")

    elif action == "standardize_case":
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.strip().title() if isinstance(x, str) else x)
            report["columns_standardized"] += 1
            _log(report, col, action, "Standardized text to Title Case.")

    # ── Type conversions ──────────────────────────────────────────
    elif action == "convert_to_numeric":
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            report["type_conversions"] += 1
            _log(report, col, action, "Converted column to numeric type.")

    elif action == "convert_to_datetime":
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            report["type_conversions"] += 1
            _log(report, col, action, "Converted column to datetime type.")

    # ── Boolean standardisation ───────────────────────────────────
    elif action == "standardize_boolean":
        if col in df.columns:
            bool_map = {
                "yes": True, "no": False,
                "true": True, "false": False,
                "1": True, "0": False,
                "y": True, "n": False,
                "t": True, "f": False,
            }
            df[col] = df[col].apply(
                lambda x: bool_map.get(str(x).strip().lower(), x)
                if pd.notna(x) else x
            )
            report["columns_standardized"] += 1
            _log(report, col, action, "Standardized boolean values to True/False.")

    # ── Category standardisation ──────────────────────────────────
    elif action == "standardize_categories":
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).strip().title() if isinstance(x, str) else x)
            report["columns_standardized"] += 1
            _log(report, col, action, "Standardized category casing.")

    # ── Outlier handling ──────────────────────────────────────────
    elif action == "cap_outliers":
        if col in df.columns:
            numeric_s = pd.to_numeric(df[col], errors="coerce")
            Q1 = numeric_s.quantile(0.25)
            Q3 = numeric_s.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n = ((numeric_s < lower) | (numeric_s > upper)).sum()
            df[col] = numeric_s.clip(lower=lower, upper=upper)
            report["outliers_handled"] += n
            _log(report, col, action, f"Capped {n} outliers to IQR bounds [{lower:.2f}, {upper:.2f}].")

    elif action == "remove_outliers":
        if col in df.columns:
            numeric_s = pd.to_numeric(df[col], errors="coerce")
            Q1 = numeric_s.quantile(0.25)
            Q3 = numeric_s.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (numeric_s >= lower) & (numeric_s <= upper) | numeric_s.isna()
            n = (~mask).sum()
            df = df[mask].reset_index(drop=True)
            report["outliers_handled"] += n
            _log(report, col, action, f"Removed {n} rows with outlier values.")

    return df, report


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _fill_numeric(df: pd.DataFrame, col: str, method: str) -> Tuple[pd.DataFrame, int]:
    if col not in df.columns:
        return df, 0
    numeric_s = pd.to_numeric(df[col], errors="coerce")
    n = numeric_s.isna().sum()
    if method == "median":
        fill_val = numeric_s.median()
    elif method == "mean":
        fill_val = numeric_s.mean()
    else:
        fill_val = 0
    df[col] = numeric_s.fillna(fill_val)
    return df, int(n)


def _fill_text(df: pd.DataFrame, col: str, placeholder: str) -> Tuple[pd.DataFrame, int]:
    if col not in df.columns:
        return df, 0
    n = int(df[col].isna().sum())
    df[col] = df[col].fillna(placeholder)
    return df, n


def _log(report: Dict, col: str, action: str, detail: str):
    report["actions_log"].append({
        "column": col,
        "action": action,
        "status": "✅ Applied",
        "detail": detail,
    })
