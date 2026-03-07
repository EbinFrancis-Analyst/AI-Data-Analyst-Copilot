"""
insights.py
Automatic Data Insights Engine
Analyzes cleaned datasets and generates structured insights.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a cleaned DataFrame and return structured insights.

    Returns a dict with keys:
        numeric     : list of per-column numeric insight dicts
        categorical : list of per-column categorical insight dicts
        correlations: list of notable correlation strings
        top_values  : dict of {col: [(value, count), ...]}
        summary     : high-level dict of dataset facts
    """
    if df is None or df.empty:
        return _empty_insights()

    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return {
        "numeric":      [_numeric_insight(df, col) for col in numeric_cols],
        "categorical":  [_categorical_insight(df, col) for col in categorical_cols],
        "correlations": _correlation_insights(df, numeric_cols),
        "top_values":   _top_values(df, categorical_cols),
        "summary":      _dataset_summary(df, numeric_cols, categorical_cols),
    }


# ─────────────────────────────────────────────
# NUMERIC INSIGHTS
# ─────────────────────────────────────────────

def _numeric_insight(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Compute descriptive stats and distribution shape for a numeric column."""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return {"column": col, "empty": True}

    q1, q3  = s.quantile(0.25), s.quantile(0.75)
    iqr     = q3 - q1
    outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    skew    = float(s.skew())

    if abs(skew) < 0.5:
        skew_label = "approximately symmetric"
    elif skew > 1:
        skew_label = "heavily right-skewed (long tail toward high values)"
    elif skew > 0.5:
        skew_label = "slightly right-skewed"
    elif skew < -1:
        skew_label = "heavily left-skewed (long tail toward low values)"
    else:
        skew_label = "slightly left-skewed"

    return {
        "column":    col,
        "empty":     False,
        "count":     int(s.count()),
        "mean":      round(float(s.mean()), 4),
        "median":    round(float(s.median()), 4),
        "std":       round(float(s.std()), 4),
        "min":       round(float(s.min()), 4),
        "max":       round(float(s.max()), 4),
        "q1":        round(float(q1), 4),
        "q3":        round(float(q3), 4),
        "iqr":       round(float(iqr), 4),
        "skew":      round(skew, 4),
        "skew_label": skew_label,
        "outliers":  outliers,
        "cv":        round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else None,
    }


# ─────────────────────────────────────────────
# CATEGORICAL INSIGHTS
# ─────────────────────────────────────────────

def _categorical_insight(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Analyze a categorical / text column."""
    s = df[col].dropna().astype(str).str.strip()
    if s.empty:
        return {"column": col, "empty": True}

    vc          = s.value_counts()
    top_val     = vc.index[0] if not vc.empty else None
    top_pct     = round(vc.iloc[0] / len(s) * 100, 1) if not vc.empty else 0
    unique_cnt  = int(s.nunique())
    total       = len(s)

    if unique_cnt / total > 0.9:
        concentration = "highly unique (likely ID or free text)"
    elif top_pct > 60:
        concentration = "dominated by one category"
    elif top_pct > 30:
        concentration = "moderately concentrated"
    else:
        concentration = "well distributed"

    return {
        "column":        col,
        "empty":         False,
        "unique_count":  unique_cnt,
        "total":         total,
        "top_value":     top_val,
        "top_pct":       top_pct,
        "top_10":        vc.head(10).to_dict(),
        "concentration": concentration,
    }


# ─────────────────────────────────────────────
# CORRELATION INSIGHTS
# ─────────────────────────────────────────────

def _correlation_insights(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    """Return human-readable strings for notable pairwise correlations."""
    insights = []
    if len(numeric_cols) < 2:
        return insights

    try:
        corr = df[numeric_cols].corr()
        pairs_seen = set()

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                pair_key = tuple(sorted([col_a, col_b]))
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)

                val = corr.loc[col_a, col_b]
                if pd.isna(val):
                    continue
                abs_val = abs(val)
                direction = "positive" if val > 0 else "negative"

                if abs_val >= 0.8:
                    label = f"Very strong {direction} correlation"
                elif abs_val >= 0.6:
                    label = f"Strong {direction} correlation"
                elif abs_val >= 0.4:
                    label = f"Moderate {direction} correlation"
                elif abs_val >= 0.2:
                    label = f"Weak {direction} correlation"
                else:
                    continue  # ignore near-zero correlations

                insights.append(
                    f"{label} ({val:+.2f}) between **{col_a}** and **{col_b}**."
                )
    except Exception:
        pass

    return insights


# ─────────────────────────────────────────────
# TOP VALUES
# ─────────────────────────────────────────────

def _top_values(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, List[Tuple]]:
    """Return top 5 value/count pairs for each categorical column."""
    result = {}
    for col in categorical_cols:
        try:
            s = df[col].dropna().astype(str).str.strip()
            vc = s.value_counts().head(5)
            result[col] = list(zip(vc.index.tolist(), vc.values.tolist()))
        except Exception:
            result[col] = []
    return result


# ─────────────────────────────────────────────
# DATASET SUMMARY
# ─────────────────────────────────────────────

def _dataset_summary(df: pd.DataFrame,
                     numeric_cols: List[str],
                     categorical_cols: List[str]) -> Dict[str, Any]:
    """High-level dataset facts."""
    return {
        "rows":              len(df),
        "cols":              len(df.columns),
        "numeric_count":     len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "memory_kb":         round(df.memory_usage(deep=True).sum() / 1024, 1),
        "completeness_pct":  round(
            (1 - df.isnull().sum().sum() / max(df.size, 1)) * 100, 1
        ),
    }


def _empty_insights():
    return {
        "numeric": [], "categorical": [], "correlations": [],
        "top_values": {}, "summary": {},
    }
