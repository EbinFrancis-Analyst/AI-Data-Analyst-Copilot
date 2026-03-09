"""
query_engine_v3.py
Natural Language → Plotly Chart Generator (v3)
Rule-based intent detection. Zero external API or statsmodels required.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics_engine import group_by_col, trend_analysis, correlation_matrix, _fmt

# ── Design tokens ──────────────────────────────────────────────────────────────

COLORS = ["#4F8BF9","#A259FF","#43D9AD","#FFB347","#FF6B6B",
          "#87CEEB","#FF69B4","#98FB98","#DDA0DD","#F0E68C"]

_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter,sans-serif", color="#E0E0E0", size=12),
    margin=dict(l=40,r=20,t=50,b=40),
    hoverlabel=dict(bgcolor="#1E1E2E", bordercolor="#444", font=dict(size=12,color="#fff")),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False,
               linecolor="rgba(255,255,255,0.3)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False,
               linecolor="rgba(255,255,255,0.3)"),
)

def _lay(fig: go.Figure, title: str = "", h: int = 420) -> go.Figure:
    fig.update_layout(**_BASE, height=h,
                      title=dict(text=f"<b>{title}</b>",
                                 font=dict(size=14, color="#E0E0E0"), x=.02))
    return fig


# ── Keywords ───────────────────────────────────────────────────────────────────

KW_SALES   = ["revenue","sales","income","amount","price","earning","total","spend"]
KW_TOP     = ["top","best","highest","most","leading","greatest","maximum","rank"]
KW_LOW     = ["lowest","worst","least","bottom","minimum","declining","poor"]
KW_AVG     = ["average","mean","avg","typical"]
KW_TREND   = ["trend","over time","monthly","time series","growth","timeline","month"]
KW_DIST    = ["distribution","spread","histogram","range","frequency","how many"]
KW_CORR    = ["correlation","relationship","heatmap","related","between","matrix"]
KW_PIE     = ["share","pie","proportion","percentage","contribution","percent","breakdown"]
KW_CUST    = ["customer","client","buyer","user","consumer","cust"]
KW_PROD    = ["product","item","goods","sku","merchandise","category","cat"]
KW_REGION  = ["city","region","state","location","area","zone","district","place"]
KW_BOX     = ["box","boxplot","quartile","outlier","iqr","median"]
KW_SCATTER = ["scatter","compare","vs","versus","against","plot"]

EXAMPLE_QUERIES = [
    "Show revenue by city",
    "Top 10 products by sales",
    "Monthly sales trend",
    "Customer purchase frequency",
    "Revenue share by category",
    "Correlation heatmap",
    "Distribution of amount",
    "Top 5 customers by spend",
    "Lowest performing regions",
    "Average order value by product",
    "Box plot of sales",
    "Sales breakdown pie chart",
]


# ── Column resolvers ───────────────────────────────────────────────────────────

def _resolve(df: pd.DataFrame, kws: List[str]) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for kw in kws:
        if kw in cl: return cl[kw]
    for kw in kws:
        for nl, nc in cl.items():
            if kw in nl: return nc
    return None

def _num_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    c = _resolve(df, hints)
    if c and pd.api.types.is_numeric_dtype(df[c]): return c
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None

def _cat_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    c = _resolve(df, hints)
    if c and not pd.api.types.is_numeric_dtype(df[c]): return c
    cats = df.select_dtypes(include=["object","category"]).columns.tolist()
    return cats[0] if cats else None

def _any(q: str, kws: List[str]) -> bool:
    return any(kw in q for kw in kws)

def _n(q: str, default: int = 10) -> int:
    m = re.search(r"top\s+(\d+)", q)
    if m: return int(m.group(1))
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,
             "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for w, v in words.items():
        if f"top {w}" in q: return v
    return default


# ── Intent detection ───────────────────────────────────────────────────────────

def _intent(q: str) -> str:
    if _any(q, KW_CORR):                              return "correlation"
    if _any(q, KW_TREND):                             return "trend"
    if _any(q, KW_BOX):                               return "box"
    if _any(q, KW_SCATTER):                           return "scatter"
    if _any(q, KW_PIE):                               return "pie"
    if _any(q, KW_DIST):                              return "histogram"
    if _any(q, KW_TOP+KW_LOW) and _any(q, KW_CUST):  return "top_customers"
    if _any(q, KW_TOP+KW_LOW) and _any(q, KW_PROD):  return "top_products"
    if _any(q, KW_TOP+KW_LOW) and _any(q, KW_REGION):return "top_regions"
    if _any(q, KW_CUST) and _any(q, KW_SALES):       return "top_customers"
    if _any(q, KW_PROD) and _any(q, KW_SALES):       return "top_products"
    if _any(q, KW_REGION) and _any(q, KW_SALES):     return "top_regions"
    if _any(q, KW_AVG):                               return "avg_bar"
    if _any(q, KW_CUST):                              return "customer_freq"
    if _any(q, KW_PROD):                              return "top_products"
    if _any(q, KW_REGION):                            return "top_regions"
    if _any(q, KW_SALES):                             return "top_regions"
    return "unknown"


# ── Chart builders ─────────────────────────────────────────────────────────────

def _bar_h(data: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    data = data.sort_values(x, ascending=True)
    fig  = px.bar(data, x=x, y=y, orientation="h", color=x,
                  color_continuous_scale=[[0,"#1a2a4a"],[.5,"#4F8BF9"],[1,"#A259FF"]],
                  text=data[x].apply(lambda v: f"{v:,.0f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _lay(fig, title)

def _bar_v(data: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.bar(data, x=x, y=y, color=y,
                 color_continuous_scale=[[0,"#1a2a4a"],[.5,"#A259FF"],[1,"#4F8BF9"]],
                 text=data[y].apply(lambda v: f"{v:,.0f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _lay(fig, title)


# ── Handlers ───────────────────────────────────────────────────────────────────

def _h_trend(df, q):
    r = trend_analysis(df)
    if r is None: return None, "No date + numeric column found."
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r["month"], y=r["revenue"], name="Revenue",
                         marker_color="#4F8BF9", opacity=.7))
    fig.add_trace(go.Scatter(x=r["month"], y=r["ma3"], name="3M MA",
                             mode="lines+markers",
                             line=dict(color="#FFB347", width=2.5, dash="dot"),
                             marker=dict(size=5)))
    return _lay(fig, "Monthly Revenue Trend"), "trend"

def _h_top_customers(df, q):
    cu = _cat_col(df, KW_CUST); sc = _num_col(df, KW_SALES)
    if not cu: return None, "No customer column found."
    n    = _n(q)
    asc  = _any(q, KW_LOW)
    data = (group_by_col(df, cu, sc, "sum", n if not asc else 999) if sc
            else df[cu].value_counts().head(n).reset_index().rename(columns={cu:"count"}))
    if asc and sc:
        data = data.sort_values(sc).head(n)
    yc = sc if sc else "count"
    t  = f"{'Bottom' if asc else 'Top'} {n} Customers"
    return _bar_h(data, yc, cu, t), "top_customers"

def _h_top_products(df, q):
    pc = _cat_col(df, KW_PROD); sc = _num_col(df, KW_SALES)
    if not pc: return None, "No product column found."
    n    = _n(q)
    asc  = _any(q, KW_LOW)
    data = (group_by_col(df, pc, sc, "sum", n if not asc else 999) if sc
            else df[pc].value_counts().head(n).reset_index())
    if asc and sc:
        data = data.sort_values(sc).head(n)
    yc = sc if sc else "count"
    t  = f"{'Bottom' if asc else 'Top'} {n} Products by Revenue"
    return _bar_h(data, yc, pc, t), "top_products"

def _h_top_regions(df, q):
    rc = _cat_col(df, KW_REGION); sc = _num_col(df, KW_SALES)
    if not rc: return None, "No region column found."
    n    = _n(q)
    asc  = _any(q, KW_LOW)
    data = (group_by_col(df, rc, sc, "sum", n if not asc else 999) if sc
            else df[rc].value_counts().head(n).reset_index())
    if asc and sc:
        data = data.sort_values(sc).head(n)
    yc = sc if sc else "count"
    t  = f"{'Bottom' if asc else 'Top'} {n} Regions by Revenue"
    return _bar_h(data, yc, rc, t), "top_regions"

def _h_pie(df, q):
    cat = _cat_col(df, KW_PROD + KW_REGION + KW_CUST)
    sc  = _num_col(df, KW_SALES)
    if not cat or not sc: return None, "Need a category and numeric column."
    data = group_by_col(df, cat, sc, "sum", 8)
    fig  = px.pie(data, names=cat, values=sc, hole=.42, color_discrete_sequence=COLORS)
    fig.update_traces(textinfo="percent+label", textfont_size=11,
                      pull=[.04]+[0]*(len(data)-1))
    return _lay(fig, f"Revenue Share by {cat}"), "pie"

def _h_histogram(df, q):
    col = _num_col(df, KW_SALES + ["age","qty","quantity","amount","price","value"])
    if not col: return None, "No numeric column found."
    plot_df = df if len(df) <= 50_000 else df.sample(20_000, random_state=42)
    fig = px.histogram(plot_df, x=col, nbins=30, opacity=.85,
                       color_discrete_sequence=["#4F8BF9"], marginal="box")
    return _lay(fig, f"Distribution of {col}"), "histogram"

def _h_correlation(df, q):
    corr = correlation_matrix(df)
    if corr is None: return None, "Need ≥ 2 numeric columns."
    fig  = px.imshow(corr,
                     color_continuous_scale=[[0,"#FF6B6B"],[.5,"#222"],[1,"#43D9AD"]],
                     zmin=-1, zmax=1, text_auto=True, aspect="auto")
    fig.update_traces(textfont_size=10)
    return _lay(fig, "Correlation Matrix", h=460), "correlation"

def _h_scatter(df, q):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(nums) < 2: return None, "Need ≥ 2 numeric columns for scatter."
    x, y = nums[0], nums[1]
    cat  = _cat_col(df, KW_REGION + KW_PROD + KW_CUST)
    plot_df = df if len(df) <= 20_000 else df.sample(10_000, random_state=42)
    fig = px.scatter(plot_df, x=x, y=y, color=cat, opacity=.6,
                     color_discrete_sequence=COLORS)
    return _lay(fig, f"Scatter: {x} vs {y}"), "scatter"

def _h_box(df, q):
    col = _num_col(df, KW_SALES + ["age","qty","amount","price"])
    cat = _cat_col(df, KW_REGION + KW_PROD + KW_CUST)
    if not col: return None, "No numeric column found."
    if cat and df[cat].nunique() <= 20:
        fig = px.box(df, x=cat, y=col, color=cat,
                     color_discrete_sequence=COLORS, points=False)
    else:
        fig = px.box(df, y=col, color_discrete_sequence=["#4F8BF9"], points="outliers")
    return _lay(fig, f"Box Plot: {col}"), "box"

def _h_avg_bar(df, q):
    cat = _cat_col(df, KW_REGION + KW_PROD + KW_CUST)
    sc  = _num_col(df, KW_SALES)
    if not cat or not sc: return None, "Need a category and numeric column."
    data = group_by_col(df, cat, sc, "mean", 15)
    return _bar_v(data, cat, sc, f"Average {sc} by {cat}"), "avg_bar"

def _h_customer_freq(df, q):
    col = _cat_col(df, KW_CUST)
    if not col: return None, "No customer column found."
    freq = df[col].value_counts().head(15).reset_index()
    freq.columns = [col, "orders"]
    return _bar_h(freq, "orders", col, "Customer Purchase Frequency"), "customer_freq"


# ── Dispatch table ─────────────────────────────────────────────────────────────

_DISPATCH = {
    "trend":          _h_trend,
    "top_customers":  _h_top_customers,
    "top_products":   _h_top_products,
    "top_regions":    _h_top_regions,
    "pie":            _h_pie,
    "histogram":      _h_histogram,
    "correlation":    _h_correlation,
    "scatter":        _h_scatter,
    "box":            _h_box,
    "avg_bar":        _h_avg_bar,
    "customer_freq":  _h_customer_freq,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, question: str) -> Tuple[Optional[go.Figure], str]:
    """
    Parse *question* and return (plotly_figure, intent_label).
    Returns (None, error_message) on failure.
    """
    if df is None or df.empty:
        return None, "No data loaded."
    if not question.strip():
        return None, "Please enter a question."

    q = question.lower().strip()
    q = re.sub(r"[^\w\s%]", " ", q)
    q = re.sub(r"\s+", " ", q)

    intent  = _intent(q)
    handler = _DISPATCH.get(intent)

    if handler is None:
        return None, (
            "I couldn't interpret that. Try:\n"
            + "\n".join(f"• *{e}*" for e in EXAMPLE_QUERIES[:6])
        )
    try:
        return handler(df, q)
    except Exception as exc:
        return None, f"Chart generation failed ({intent}): {exc}"
