"""
ai_insights.py
Rule-Based AI Insight Generator
Produces natural language analysis summaries without any external API.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_ai_summary(df: pd.DataFrame, insights: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generate a structured natural language summary from insight data.

    Returns a dict with keys:
        overview      : 1–2 sentences describing the dataset
        numeric_obs   : observations about numeric columns
        category_obs  : observations about categorical columns
        correlation_obs: observations about relationships
        anomalies     : flags for potential data quality concerns
        recommendations: actionable suggestions for the analyst
    """
    if not insights or not insights.get("summary"):
        return _empty_summary()

    summary   = insights["summary"]
    numerics  = [n for n in insights.get("numeric", []) if not n.get("empty")]
    cats      = [c for c in insights.get("categorical", []) if not c.get("empty")]
    corrs     = insights.get("correlations", [])

    return {
        "overview":         _build_overview(df, summary, numerics, cats),
        "numeric_obs":      _build_numeric_observations(numerics),
        "category_obs":     _build_category_observations(cats),
        "correlation_obs":  _build_correlation_observations(corrs),
        "anomalies":        _build_anomaly_flags(numerics, cats, summary),
        "recommendations":  _build_recommendations(numerics, cats, corrs, summary),
    }


# ─────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────

def _build_overview(df: pd.DataFrame, summary: Dict,
                    numerics: List, cats: List) -> List[str]:
    lines = []
    rows, cols = summary["rows"], summary["cols"]
    comp = summary["completeness_pct"]
    mem  = summary["memory_kb"]

    lines.append(
        f"The dataset contains **{rows:,} rows** and **{cols} columns** "
        f"({summary['numeric_count']} numeric, {summary['categorical_count']} categorical), "
        f"occupying **{mem} KB** in memory."
    )

    if comp >= 98:
        lines.append(f"Data completeness is excellent at **{comp}%** — very few missing values.")
    elif comp >= 90:
        lines.append(f"Data completeness is good at **{comp}%** with minor gaps.")
    elif comp >= 75:
        lines.append(f"Data completeness is moderate at **{comp}%** — some columns may need attention.")
    else:
        lines.append(f"⚠️ Data completeness is low at **{comp}%** — significant missing data present.")

    return lines


# ─────────────────────────────────────────────
# NUMERIC OBSERVATIONS
# ─────────────────────────────────────────────

def _build_numeric_observations(numerics: List[Dict]) -> List[str]:
    if not numerics:
        return ["No numeric columns found in the dataset."]

    lines = []
    for n in numerics:
        col   = n["column"]
        mean  = n["mean"]
        med   = n["median"]
        mn    = n["min"]
        mx    = n["max"]
        skew  = n["skew_label"]
        outs  = n["outliers"]
        cv    = n.get("cv")

        line = (f"**{col}** ranges from {mn:,} to {mx:,} "
                f"(mean: {mean:,}, median: {med:,}). "
                f"Distribution is {skew}.")

        if outs > 0:
            line += f" Contains **{outs} outlier(s)**."

        if cv is not None:
            if cv > 100:
                line += f" High variability (CV: {cv}%) suggests diverse values."
            elif cv < 15:
                line += f" Low variability (CV: {cv}%) — values are tightly clustered."

        if abs(mean - med) / max(abs(med), 1) > 0.2:
            line += " Mean and median diverge noticeably — consider checking for extreme values."

        lines.append(line)

    return lines


# ─────────────────────────────────────────────
# CATEGORICAL OBSERVATIONS
# ─────────────────────────────────────────────

def _build_category_observations(cats: List[Dict]) -> List[str]:
    if not cats:
        return ["No categorical columns found in the dataset."]

    lines = []
    for c in cats:
        col   = c["column"]
        top   = c["top_value"]
        pct   = c["top_pct"]
        uniq  = c["unique_count"]
        conc  = c["concentration"]

        line = f"**{col}** has {uniq} unique value(s) and is {conc}."

        if top and pct > 0:
            line += f" The most frequent value is **'{top}'** ({pct}% of records)."

        if pct > 80:
            line += " This dominant category may indicate a data imbalance."
        elif uniq > 50:
            line += " High cardinality — consider grouping into broader categories."

        lines.append(line)

    return lines


# ─────────────────────────────────────────────
# CORRELATION OBSERVATIONS
# ─────────────────────────────────────────────

def _build_correlation_observations(corrs: List[str]) -> List[str]:
    if not corrs:
        return ["No notable correlations detected among numeric columns."]

    lines = ["The following relationships were detected between numeric columns:"]
    lines.extend([f"• {c}" for c in corrs])

    strong = [c for c in corrs if "Very strong" in c or "Strong" in c]
    if strong:
        lines.append(
            "Strong correlations may indicate redundant features — "
            "consider dimensionality reduction if building a model."
        )
    return lines


# ─────────────────────────────────────────────
# ANOMALY FLAGS
# ─────────────────────────────────────────────

def _build_anomaly_flags(numerics: List, cats: List, summary: Dict) -> List[str]:
    flags = []

    for n in numerics:
        if n["outliers"] > 0:
            pct = round(n["outliers"] / n["count"] * 100, 1)
            if pct > 10:
                flags.append(
                    f"⚠️ **{n['column']}** has {pct}% outliers — "
                    "unusually high; verify if these are valid extreme values."
                )

        if n.get("std", 0) == 0:
            flags.append(f"⚠️ **{n['column']}** has zero variance — column may be constant.")

        if n["min"] < 0 and "price" in n["column"].lower():
            flags.append(f"⚠️ **{n['column']}** contains negative values — "
                         "unexpected for a price/amount field.")

    for c in cats:
        if c["top_pct"] > 95 and c["unique_count"] > 1:
            flags.append(
                f"⚠️ **{c['column']}** is almost entirely one value ('{c['top_value']}', "
                f"{c['top_pct']}%) — may not be useful for analysis."
            )

    if summary.get("completeness_pct", 100) < 80:
        flags.append("⚠️ Overall data completeness is below 80% — imputation strategies may affect analysis quality.")

    if not flags:
        flags.append("✅ No significant anomalies detected in the cleaned dataset.")

    return flags


# ─────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────

def _build_recommendations(numerics: List, cats: List,
                            corrs: List, summary: Dict) -> List[str]:
    recs = []

    outlier_cols = [n["column"] for n in numerics if n["outliers"] > 0]
    if outlier_cols:
        recs.append(
            f"🔎 Investigate outliers in: {', '.join(f'**{c}**' for c in outlier_cols)}. "
            "Decide whether to cap, remove, or retain them based on domain knowledge."
        )

    skewed_cols = [n["column"] for n in numerics if abs(n["skew"]) > 1]
    if skewed_cols:
        recs.append(
            f"📐 Consider log-transforming skewed columns: "
            f"{', '.join(f'**{c}**' for c in skewed_cols)} before statistical modelling."
        )

    high_card_cols = [c["column"] for c in cats if c["unique_count"] > 50]
    if high_card_cols:
        recs.append(
            f"🗂️ High-cardinality columns ({', '.join(f'**{c}**' for c in high_card_cols)}) "
            "may benefit from grouping or encoding before use in ML models."
        )

    strong_corrs = [c for c in corrs if "Very strong" in c or "Strong" in c]
    if strong_corrs:
        recs.append(
            "🔗 Strongly correlated features detected. If training a model, consider removing "
            "redundant features to reduce multicollinearity."
        )

    if summary.get("numeric_count", 0) >= 3:
        recs.append(
            "📊 With multiple numeric columns, a correlation heatmap and pairplot "
            "are recommended for a deeper understanding of feature relationships."
        )

    if summary.get("completeness_pct", 100) < 95:
        recs.append(
            "🩹 Some missing values remain. Review imputation choices — "
            "median is safer than mean for skewed distributions."
        )

    if not recs:
        recs.append("✅ Dataset looks analysis-ready. No specific preprocessing recommendations.")

    return recs


# ─────────────────────────────────────────────
# EMPTY FALLBACK
# ─────────────────────────────────────────────

def _empty_summary():
    return {
        "overview": ["No data available for analysis."],
        "numeric_obs": [],
        "category_obs": [],
        "correlation_obs": [],
        "anomalies": [],
        "recommendations": [],
    }
