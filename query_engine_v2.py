"""
query_engine_v2.py
Natural Language Query Engine for the AI Data Analyst Copilot.
Supports 20+ business question intents using keyword matching and pandas operations.
No external APIs required.
"""

import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────
# SYNONYM DICTIONARIES
# ─────────────────────────────────────────────

SALES_KW       = ["sales", "revenue", "income", "earning", "turnover", "amount", "price", "value"]
HIGHEST_KW     = ["highest", "top", "most", "maximum", "max", "best", "greatest", "largest", "leading"]
LOWEST_KW      = ["lowest", "bottom", "least", "minimum", "min", "worst", "smallest", "weakest", "poor"]
AVERAGE_KW     = ["average", "mean", "avg", "typical", "median"]
TOTAL_KW       = ["total", "sum", "overall", "aggregate", "combined", "all", "entire"]
CITY_KW        = ["city", "region", "location", "area", "zone", "place", "district", "state", "branch"]
PRODUCT_KW     = ["product", "item", "goods", "merchandise", "sku", "category", "brand"]
CUSTOMER_KW    = ["customer", "client", "buyer", "user", "consumer", "account", "purchaser"]
QUANTITY_KW    = ["quantity", "units", "count", "volume", "pieces", "amount sold", "sold"]
TREND_KW       = ["trend", "over time", "monthly", "daily", "weekly", "growth", "progress", "timeline"]
FORECAST_KW    = ["forecast", "predict", "next month", "projection", "future", "estimate", "expected"]
CORRELATION_KW = ["correlation", "relationship", "related", "depend", "link", "association", "between"]
DISTRIBUTION_KW= ["distribution", "spread", "range", "histogram", "breakdown", "split"]
UNIQUE_KW      = ["unique", "distinct", "different", "how many"]
SHARE_KW       = ["share", "percentage", "percent", "portion", "proportion", "contribution", "%"]
PERCENTILE_KW  = ["percentile", "top 5%", "top 10%", "elite", "vip", "premium", "top percent"]
DECLINING_KW   = ["declining", "falling", "drop", "decrease", "worst performing", "underperform", "losing"]
FREQUENT_KW    = ["frequent", "popular", "common", "mostly", "mainly", "often", "repeat"]
ORDER_KW       = ["order", "transaction", "purchase", "deal", "invoice"]


# ─────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────

def _find_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    Find the best matching column name from a list of keywords.
    Prefers exact match → startswith → contains.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    for kw in keywords:
        if kw in cols_lower:
            return cols_lower[kw]

    for kw in keywords:
        for col_l, col in cols_lower.items():
            if col_l.startswith(kw):
                return col

    for kw in keywords:
        for col_l, col in cols_lower.items():
            if kw in col_l:
                return col

    return None


def _numeric_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    """Return a numeric column matching hints, else the first numeric column."""
    col = _find_col(df, hints)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        return col
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[0] if num_cols else None


def _category_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    """Return a string/categorical column matching hints, else the first such column."""
    col = _find_col(df, hints)
    if col and not pd.api.types.is_numeric_dtype(df[col]):
        return col
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cat_cols[0] if cat_cols else None


def _date_col(df: pd.DataFrame) -> Optional[str]:
    """Find the first datetime column."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if "date" in col.lower() or "time" in col.lower():
            try:
                pd.to_datetime(df[col], errors="raise")
                return col
            except Exception:
                pass
    return None


def _fmt(val) -> str:
    """Format a number nicely."""
    try:
        f = float(val)
        if f == int(f) and abs(f) < 1e12:
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except Exception:
        return str(val)


def _extract_n(question: str, default: int = 5) -> int:
    """Extract a number N from phrases like 'top 3', 'top ten'."""
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    m = re.search(r'top\s+(\d+)', question)
    if m:
        return int(m.group(1))
    for word, num in word_map.items():
        if f"top {word}" in question:
            return num
    return default


def _any(question: str, keywords: List[str]) -> bool:
    return any(kw in question for kw in keywords)


# ─────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────

def _detect_intent(question: str) -> str:
    """
    Map a normalized question string to one of 22 intent labels.
    Order matters — more specific intents are checked first.
    """
    q = question

    # Compound intents first
    if _any(q, FORECAST_KW):
        return "forecast_next_month"
    if _any(q, CORRELATION_KW):
        return "correlation"
    if _any(q, DECLINING_KW):
        return "declining_region"
    if _any(q, PERCENTILE_KW):
        return "top_percentile_customers"
    if _any(q, TREND_KW):
        return "sales_trend"
    if _any(q, DISTRIBUTION_KW):
        return "sales_distribution"

    if _any(q, SHARE_KW) and _any(q, PRODUCT_KW + CITY_KW):
        return "revenue_share_category"

    if _any(q, UNIQUE_KW) and _any(q, CUSTOMER_KW):
        return "unique_customer_count"

    if _any(q, ORDER_KW) and _any(q, AVERAGE_KW):
        return "average_order_value"

    if _any(q, TOTAL_KW) and _any(q, QUANTITY_KW):
        return "total_quantity_sold"

    if _any(q, TOTAL_KW) and _any(q, SALES_KW):
        return "total_revenue"

    if _any(q, AVERAGE_KW) and _any(q, SALES_KW + ["price", "order"]):
        return "average_price"

    if _any(q, HIGHEST_KW) and _any(q, CITY_KW):
        return "highest_sales_city"

    if _any(q, LOWEST_KW) and _any(q, CITY_KW):
        return "lowest_sales_city"

    if _any(q, HIGHEST_KW) and _any(q, PRODUCT_KW):
        return "highest_selling_product"

    if (_any(q, SALES_KW) or _any(q, FREQUENT_KW)) and _any(q, PRODUCT_KW):
        if _any(q, FREQUENT_KW) or "most" in q:
            return "most_frequent_product"
        return "revenue_by_product"

    if _any(q, SALES_KW) and _any(q, CITY_KW):
        return "sales_by_city"

    if _any(q, CUSTOMER_KW) and _any(q, HIGHEST_KW + ["top"]):
        return "top_n_customers"

    if _any(q, QUANTITY_KW):
        return "total_quantity_sold"

    if _any(q, CUSTOMER_KW):
        return "unique_customer_count"

    if _any(q, PRODUCT_KW):
        return "most_frequent_product"

    if _any(q, CITY_KW):
        return "sales_by_city"

    if _any(q, SALES_KW) or _any(q, TOTAL_KW):
        return "total_revenue"

    if _any(q, AVERAGE_KW):
        return "average_price"

    return "unknown"


# ─────────────────────────────────────────────
# INTENT HANDLERS
# ─────────────────────────────────────────────

def _handle_total_revenue(df, q):
    col = _numeric_col(df, ["price", "sales", "revenue", "amount", "total", "value"])
    if not col:
        return "❌ No numeric column found to calculate total revenue."
    total = df[col].sum()
    return f"💰 The total revenue is **{_fmt(total)}** (column: `{col}`)."


def _handle_average_price(df, q):
    col = _numeric_col(df, ["price", "sales", "revenue", "amount", "value"])
    if not col:
        return "❌ No numeric column found to calculate the average."
    avg = df[col].mean()
    return f"📊 The average value in `{col}` is **{_fmt(avg)}**."


def _handle_highest_city(df, q):
    city_col  = _category_col(df, CITY_KW)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not city_col or not sales_col:
        return "❌ Could not find city and sales columns."
    grp   = df.groupby(city_col)[sales_col].sum()
    city  = grp.idxmax()
    value = grp.max()
    return f"🏆 **{city}** has the highest revenue with **{_fmt(value)}** (column: `{sales_col}`)."


def _handle_lowest_city(df, q):
    city_col  = _category_col(df, CITY_KW)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not city_col or not sales_col:
        return "❌ Could not find city and sales columns."
    grp   = df.groupby(city_col)[sales_col].sum()
    city  = grp.idxmin()
    value = grp.min()
    return f"📉 **{city}** has the lowest revenue with **{_fmt(value)}** (column: `{sales_col}`)."


def _handle_highest_product(df, q):
    prod_col  = _category_col(df, PRODUCT_KW)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not prod_col or not sales_col:
        return "❌ Could not find product and sales columns."
    grp     = df.groupby(prod_col)[sales_col].sum()
    product = grp.idxmax()
    value   = grp.max()
    return f"🥇 **{product}** is the highest-selling product with **{_fmt(value)}** in `{sales_col}`."


def _handle_sales_by_city(df, q):
    city_col  = _category_col(df, CITY_KW)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not city_col or not sales_col:
        return "❌ Could not find city and sales columns."
    grp = df.groupby(city_col)[sales_col].sum().sort_values(ascending=False)
    rows = "\n".join([f"  • **{k}**: {_fmt(v)}" for k, v in grp.items()])
    return f"🗺️ **Sales by {city_col}:**\n{rows}"


def _handle_revenue_by_product(df, q):
    prod_col  = _category_col(df, PRODUCT_KW)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not prod_col or not sales_col:
        return "❌ Could not find product and sales columns."
    grp  = df.groupby(prod_col)[sales_col].sum().sort_values(ascending=False)
    top5 = grp.head(5)
    rows = "\n".join([f"  • **{k}**: {_fmt(v)}" for k, v in top5.items()])
    return f"📦 **Revenue by {prod_col} (top 5):**\n{rows}"


def _handle_total_quantity(df, q):
    col = _numeric_col(df, ["quantity", "qty", "units", "count", "volume"])
    if not col:
        col = _numeric_col(df, QUANTITY_KW)
    if not col:
        return "❌ No quantity column found."
    total = df[col].sum()
    return f"📦 Total quantity sold: **{_fmt(total)}** units (column: `{col}`)."


def _handle_average_order_value(df, q):
    col = _numeric_col(df, ["order", "value", "amount", "total", "price", "sales"])
    if not col:
        return "❌ No order value column found."
    aov = df[col].mean()
    return f"🧾 The average order value is **{_fmt(aov)}** (column: `{col}`)."


def _handle_top_n_customers(df, q):
    n        = _extract_n(q, default=5)
    cust_col = _category_col(df, CUSTOMER_KW)
    val_col  = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not cust_col:
        return "❌ No customer column found."
    if val_col:
        top = df.groupby(cust_col)[val_col].sum().sort_values(ascending=False).head(n)
        rows = "\n".join([f"  {i+1}. **{k}**: {_fmt(v)}" for i, (k, v) in enumerate(top.items())])
        return f"👥 **Top {n} customers by revenue:**\n{rows}"
    else:
        top = df[cust_col].value_counts().head(n)
        rows = "\n".join([f"  {i+1}. **{k}**: {_fmt(v)} orders" for i, (k, v) in enumerate(top.items())])
        return f"👥 **Top {n} customers by order count:**\n{rows}"


def _handle_correlation(df, q):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return "❌ Need at least 2 numeric columns to compute correlations."
    corr = df[num_cols].corr()
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val = corr.iloc[i, j]
            if not pd.isna(val):
                pairs.append((num_cols[i], num_cols[j], val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = pairs[:5]
    rows = []
    for a, b, v in top_pairs:
        label = ("very strong" if abs(v) > 0.8 else
                 "strong" if abs(v) > 0.6 else
                 "moderate" if abs(v) > 0.4 else "weak")
        direction = "positive" if v > 0 else "negative"
        rows.append(f"  • `{a}` ↔ `{b}`: **{v:.2f}** ({label} {direction})")
    return "🔗 **Top correlations between numeric columns:**\n" + "\n".join(rows)


def _handle_sales_trend(df, q):
    date_col  = _date_col(df)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not date_col or not sales_col:
        return "❌ A datetime column and a numeric sales column are required for trend analysis."
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    monthly = tmp.resample("ME", on=date_col)[sales_col].sum()
    if monthly.empty:
        return "❌ Could not compute monthly trend — check date column format."
    rows = "\n".join([f"  • **{d.strftime('%b %Y')}**: {_fmt(v)}"
                      for d, v in monthly.tail(6).items()])
    delta = monthly.iloc[-1] - monthly.iloc[-2] if len(monthly) >= 2 else 0
    trend = "📈 upward" if delta > 0 else "📉 downward"
    return (f"📅 **Sales trend (last 6 months):**\n{rows}\n\n"
            f"Latest month is {trend} by **{_fmt(abs(delta))}**.")


def _handle_declining_region(df, q):
    city_col  = _category_col(df, CITY_KW)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    date_col  = _date_col(df)
    if not city_col or not sales_col:
        return "❌ City and sales columns required."
    if date_col:
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
        mid = tmp[date_col].median()
        first_half  = tmp[tmp[date_col] <= mid].groupby(city_col)[sales_col].sum()
        second_half = tmp[tmp[date_col] >  mid].groupby(city_col)[sales_col].sum()
        diff        = (second_half - first_half).dropna().sort_values()
        if diff.empty:
            return "❌ Could not compute decline — insufficient date range."
        worst_city  = diff.idxmin()
        worst_delta = diff.min()
        return (f"📉 **{worst_city}** is the most declining region with a "
                f"revenue change of **{_fmt(worst_delta)}** in the second half vs first half.")
    grp   = df.groupby(city_col)[sales_col].sum()
    city  = grp.idxmin()
    value = grp.min()
    return f"📉 **{city}** has the lowest overall revenue at **{_fmt(value)}** — may be declining."


def _handle_most_frequent_product(df, q):
    prod_col = _category_col(df, PRODUCT_KW)
    if not prod_col:
        return "❌ No product column found."
    vc      = df[prod_col].value_counts()
    product = vc.index[0]
    count   = vc.iloc[0]
    top5    = vc.head(5)
    rows    = "\n".join([f"  {i+1}. **{k}**: {_fmt(v)} times"
                         for i, (k, v) in enumerate(top5.items())])
    return (f"🏅 Most frequent product: **{product}** ({_fmt(count)} occurrences)\n\n"
            f"**Top 5 products by frequency:**\n{rows}")


def _handle_unique_customers(df, q):
    cust_col = _category_col(df, CUSTOMER_KW)
    if not cust_col:
        return "❌ No customer column found."
    count = df[cust_col].nunique()
    total = len(df)
    return (f"👤 There are **{_fmt(count)} unique customers** "
            f"across **{_fmt(total)}** total records.")


def _handle_top_percentile_customers(df, q):
    cust_col = _category_col(df, CUSTOMER_KW)
    val_col  = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    pct      = 5.0
    m = re.search(r'top\s+(\d+)\s*%', q)
    if m:
        pct = float(m.group(1))
    if not cust_col or not val_col:
        return "❌ Customer and value columns required."
    cust_rev = df.groupby(cust_col)[val_col].sum().sort_values(ascending=False)
    n_top    = max(1, int(len(cust_rev) * pct / 100))
    top_rev  = cust_rev.head(n_top)
    total    = cust_rev.sum()
    share    = top_rev.sum() / total * 100 if total > 0 else 0
    rows     = "\n".join([f"  {i+1}. **{k}**: {_fmt(v)}"
                          for i, (k, v) in enumerate(top_rev.head(5).items())])
    return (f"💎 **Top {pct:.0f}% of customers ({n_top} customers)** "
            f"account for **{share:.1f}%** of total revenue.\n\n"
            f"**Top earners:**\n{rows}")


def _handle_revenue_share(df, q):
    cat_col  = _category_col(df, PRODUCT_KW + CITY_KW)
    val_col  = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not cat_col or not val_col:
        return "❌ A category column and a numeric column are required."
    grp    = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
    total  = grp.sum()
    top5   = grp.head(5)
    rows   = "\n".join([f"  • **{k}**: {_fmt(v)} ({v/total*100:.1f}%)"
                        for k, v in top5.items()])
    return f"🥧 **Revenue share by {cat_col} (top 5):**\n{rows}"


def _handle_sales_distribution(df, q):
    col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not col:
        return "❌ No numeric column found for distribution analysis."
    s      = df[col].dropna()
    p10    = s.quantile(0.10)
    p25    = s.quantile(0.25)
    median = s.median()
    p75    = s.quantile(0.75)
    p90    = s.quantile(0.90)
    skew   = s.skew()
    skew_label = ("right-skewed (most values are low)" if skew > 0.5
                  else "left-skewed (most values are high)" if skew < -0.5
                  else "approximately symmetric")
    return (f"📊 **Distribution of `{col}`:**\n"
            f"  • 10th pct: {_fmt(p10)}\n"
            f"  • 25th pct: {_fmt(p25)}\n"
            f"  • Median:   {_fmt(median)}\n"
            f"  • 75th pct: {_fmt(p75)}\n"
            f"  • 90th pct: {_fmt(p90)}\n\n"
            f"Distribution is **{skew_label}** (skewness: {skew:.2f}).")


def _handle_top_categories(df, q):
    n       = _extract_n(q, default=5)
    cat_col = _category_col(df, PRODUCT_KW + CITY_KW)
    val_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not cat_col:
        return "❌ No categorical column found."
    if val_col:
        top = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False).head(n)
        rows = "\n".join([f"  {i+1}. **{k}**: {_fmt(v)}"
                          for i, (k, v) in enumerate(top.items())])
        return f"🏆 **Top {n} categories by {val_col}:**\n{rows}"
    top  = df[cat_col].value_counts().head(n)
    rows = "\n".join([f"  {i+1}. **{k}**: {_fmt(v)} records"
                      for i, (k, v) in enumerate(top.items())])
    return f"🏆 **Top {n} {cat_col} by frequency:**\n{rows}"


def _handle_forecast(df, q):
    date_col  = _date_col(df)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])
    if not date_col or not sales_col:
        return "❌ Datetime and numeric columns are required for forecasting."
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    monthly = tmp.resample("ME", on=date_col)[sales_col].sum()
    if len(monthly) < 2:
        return "❌ Not enough monthly data points to generate a forecast."
    last_3    = monthly.tail(3)
    avg       = last_3.mean()
    trend     = monthly.iloc[-1] - monthly.iloc[-2]
    forecast  = avg + trend * 0.5
    direction = "📈 growth" if trend > 0 else "📉 decline"
    return (f"🔮 **Forecast for next month:**\n"
            f"  • Last 3-month average: **{_fmt(avg)}**\n"
            f"  • Recent trend: {direction} of **{_fmt(abs(trend))}**\n"
            f"  • Projected next month: **{_fmt(forecast)}**\n\n"
            f"_This is a simple moving-average projection — not a statistical model._")


def _handle_unknown():
    return (
        "🤔 Sorry, I couldn't understand that question.\n\n"
        "**Try asking about:**\n"
        "  • Totals → *'What is the total revenue?'*\n"
        "  • Averages → *'What is the average price?'*\n"
        "  • Rankings → *'Which city has the highest sales?'*\n"
        "  • Trends → *'Show sales trend over time'*\n"
        "  • Products → *'Which product sells the most?'*\n"
        "  • Customers → *'Who are the top 5 customers?'*\n"
        "  • Forecasts → *'Forecast next month revenue'*"
    )


# ─────────────────────────────────────────────
# INTENT → HANDLER MAP
# ─────────────────────────────────────────────

INTENT_MAP = {
    "total_revenue":            _handle_total_revenue,
    "average_price":            _handle_average_price,
    "highest_sales_city":       _handle_highest_city,
    "lowest_sales_city":        _handle_lowest_city,
    "highest_selling_product":  _handle_highest_product,
    "sales_by_city":            _handle_sales_by_city,
    "revenue_by_product":       _handle_revenue_by_product,
    "total_quantity_sold":      _handle_total_quantity,
    "average_order_value":      _handle_average_order_value,
    "top_n_customers":          _handle_top_n_customers,
    "correlation":              _handle_correlation,
    "sales_trend":              _handle_sales_trend,
    "declining_region":         _handle_declining_region,
    "most_frequent_product":    _handle_most_frequent_product,
    "unique_customer_count":    _handle_unique_customers,
    "top_percentile_customers": _handle_top_percentile_customers,
    "revenue_share_category":   _handle_revenue_share,
    "sales_distribution":       _handle_sales_distribution,
    "top_categories":           _handle_top_categories,
    "forecast_next_month":      _handle_forecast,
}


# ─────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────

def answer_query(df: pd.DataFrame, question: str) -> str:
    """
    Answer a natural language business question about a DataFrame.

    Steps:
        1. Normalize the question
        2. Detect intent via keyword matching
        3. Dispatch to appropriate pandas handler
        4. Return a human-readable answer string

    Args:
        df:       The (cleaned) DataFrame to query
        question: A free-text business question string

    Returns:
        A human-readable answer string (may contain Markdown).
    """
    if df is None or df.empty:
        return "❌ No dataset loaded. Please upload and clean a dataset first."

    if not question or not question.strip():
        return "❓ Please enter a question."

    # Step 1 — Normalize
    q = question.lower().strip()
    q = re.sub(r'[^\w\s%]', ' ', q)
    q = re.sub(r'\s+', ' ', q)

    # Step 2 — Detect intent
    intent = _detect_intent(q)

    # Step 3 — Dispatch
    handler = INTENT_MAP.get(intent)
    if handler is None:
        return _handle_unknown()

    try:
        return handler(df, q)
    except Exception as e:
        return (f"⚠️ I understood your question (intent: `{intent}`) "
                f"but ran into an error: `{e}`.\n"
                f"Check that the expected columns exist in your dataset.")
