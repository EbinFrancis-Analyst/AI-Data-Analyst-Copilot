"""
dashboard_generator.py
Executive Analytics Dashboard Generator
Produces a Power BI / Tableau style interactive dashboard using Plotly.
Dynamically adapts to any dataset structure.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict, Any


# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────

BRAND_PRIMARY   = "#4F8BF9"
BRAND_SECONDARY = "#A259FF"
BRAND_SUCCESS   = "#43D9AD"
BRAND_WARNING   = "#FFB347"
BRAND_DANGER    = "#FF6B6B"
BRAND_GOLD      = "#F5C842"

COLOR_SEQ = [
    "#4F8BF9", "#A259FF", "#43D9AD", "#FFB347",
    "#FF6B6B", "#87CEEB", "#FF69B4", "#98FB98",
    "#DDA0DD", "#F0E68C", "#20B2AA", "#FF7F50",
]

CHART_BG    = "rgba(0,0,0,0)"
PAPER_BG    = "rgba(0,0,0,0)"
GRID_COLOR  = "rgba(255,255,255,0.07)"
FONT_COLOR  = "#E0E0E0"
AXIS_COLOR  = "rgba(255,255,255,0.3)"

BASE_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1E1E2E",
        bordercolor="#444",
        font=dict(size=12, color="#fff"),
    ),
)

AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=GRID_COLOR,
    zeroline=False,
    linecolor=AXIS_COLOR,
    tickfont=dict(size=11, color=FONT_COLOR),
    title_font=dict(size=12, color=FONT_COLOR),
)

# ─────────────────────────────────────────────
# COLUMN DETECTION HELPERS
# ─────────────────────────────────────────────

def _find_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Find the best-matching column from a keyword list (exact → prefix → contains)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in keywords:
        if kw in cols_lower:
            return cols_lower[kw]
    for kw in keywords:
        for cl, c in cols_lower.items():
            if cl.startswith(kw):
                return c
    for kw in keywords:
        for cl, c in cols_lower.items():
            if kw in cl:
                return c
    return None


def _numeric_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    col = _find_col(df, hints)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        return col
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None


def _cat_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    col = _find_col(df, hints)
    if col and not pd.api.types.is_numeric_dtype(df[col]):
        return col
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cats[0] if cats else None


def _date_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(kw in col.lower() for kw in ["date", "time", "month", "year", "day"]):
            try:
                pd.to_datetime(df[col], errors="raise")
                return col
            except Exception:
                pass
    return None


def _fmt(val: float) -> str:
    """Format a number for KPI display."""
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:,.2f}"


def _apply_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply the standard dark theme layout to any Plotly figure."""
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color=FONT_COLOR), x=0.02),
        height=height,
        xaxis=AXIS_STYLE,
        yaxis=AXIS_STYLE,
    )
    return fig


def _section(label: str):
    """Render a styled dashboard section header."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {BRAND_PRIMARY}22, transparent);
        border-left: 4px solid {BRAND_PRIMARY};
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 1rem;
        margin: 1.5rem 0 0.75rem 0;
    ">
        <span style="font-size:1.1rem; font-weight:700; color:{BRAND_PRIMARY};">{label}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — KPI CARDS
# ─────────────────────────────────────────────

def _render_kpis(df: pd.DataFrame):
    """Render the top-row KPI metric cards."""
    _section("📌 Executive KPI Overview")

    sales_col   = _numeric_col(df, ["price", "sales", "revenue", "amount", "total", "value"])
    qty_col     = _numeric_col(df, ["quantity", "qty", "units", "count", "volume"])
    cust_col    = _cat_col(df, ["customer", "client", "buyer", "user", "name"])
    prod_col    = _cat_col(df, ["product", "item", "goods", "sku", "category"])
    order_col   = _cat_col(df, ["order", "order_id", "invoice", "transaction"])

    total_rev   = df[sales_col].sum()         if sales_col else 0
    total_orders= df[order_col].nunique()     if order_col else len(df)
    aov         = total_rev / max(total_orders, 1)
    total_custs = df[cust_col].nunique()      if cust_col else 0
    total_prods = df[prod_col].nunique()      if prod_col else 0
    avg_qty     = df[qty_col].mean()          if qty_col else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def _kpi(container, icon, label, value, delta_label=""):
        container.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1E1E2E 60%, #2a2a3e);
            border: 1px solid rgba(79,139,249,0.25);
            border-radius: 12px;
            padding: 1rem 0.8rem;
            text-align: center;
        ">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-size:1.4rem; font-weight:800; color:{BRAND_PRIMARY};
                        margin: 0.2rem 0;">{value}</div>
            <div style="font-size:0.72rem; color:#888; text-transform:uppercase;
                        letter-spacing:0.05em;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    _kpi(c1, "💰", "Total Revenue",    _fmt(total_rev))
    _kpi(c2, "🧾", "Total Orders",     f"{total_orders:,}")
    _kpi(c3, "📊", "Avg Order Value",  _fmt(aov))
    _kpi(c4, "👥", "Unique Customers", f"{total_custs:,}")
    _kpi(c5, "📦", "Unique Products",  f"{total_prods:,}")
    _kpi(c6, "🔢", "Avg Qty / Order",  _fmt(avg_qty) if avg_qty else f"{len(df):,} rows")

    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 2 — REVENUE & SALES PERFORMANCE
# ─────────────────────────────────────────────

def _render_revenue(df: pd.DataFrame):
    """Revenue breakdown: city bar + product pie."""
    _section("💹 Revenue & Sales Performance")

    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount", "total"])
    city_col  = _cat_col(df, ["city", "region", "location", "area", "zone", "branch"])
    prod_col  = _cat_col(df, ["product", "item", "goods", "sku", "category"])

    col_left, col_right = st.columns([3, 2])

    # ── Bar: revenue by city ──
    with col_left:
        if city_col and sales_col:
            grp = (df.groupby(city_col)[sales_col]
                     .sum()
                     .sort_values(ascending=True)
                     .tail(15))
            fig = px.bar(
                grp.reset_index(),
                x=sales_col, y=city_col,
                orientation="h",
                color=sales_col,
                color_continuous_scale=[[0, "#1a2a4a"], [0.5, BRAND_PRIMARY], [1, BRAND_SECONDARY]],
                text=grp.values,
                labels={sales_col: "Revenue", city_col: ""},
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}",
                textposition="outside",
                marker_line_width=0,
            )
            fig.update_coloraxes(showscale=False)
            _apply_layout(fig, f"Revenue by {city_col}", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No city/region and numeric column found.")

    # ── Pie: revenue share by product ──
    with col_right:
        if prod_col and sales_col:
            grp = (df.groupby(prod_col)[sales_col]
                     .sum()
                     .sort_values(ascending=False)
                     .head(8))
            fig = px.pie(
                grp.reset_index(),
                names=prod_col,
                values=sales_col,
                color_discrete_sequence=COLOR_SEQ,
                hole=0.45,
            )
            fig.update_traces(
                textinfo="percent+label",
                textfont_size=11,
                pull=[0.04] + [0] * (len(grp) - 1),
            )
            _apply_layout(fig, f"Revenue Share by {prod_col}", height=400)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product and numeric column found.")

    # ── Bar: top 10 products by revenue ──
    if prod_col and sales_col:
        top10 = (df.groupby(prod_col)[sales_col]
                   .sum()
                   .sort_values(ascending=False)
                   .head(10)
                   .reset_index())
        fig = px.bar(
            top10,
            x=prod_col, y=sales_col,
            color=sales_col,
            color_continuous_scale=[[0, "#1a2a4a"], [0.5, BRAND_SECONDARY], [1, BRAND_PRIMARY]],
            text=top10[sales_col].apply(lambda v: _fmt(v)),
            labels={sales_col: "Revenue", prod_col: ""},
        )
        fig.update_traces(
            textposition="outside",
            marker_line_width=0,
        )
        fig.update_coloraxes(showscale=False)
        _apply_layout(fig, f"Top 10 {prod_col}s by Revenue", height=340)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 3 — CUSTOMER ANALYSIS
# ─────────────────────────────────────────────

def _render_customers(df: pd.DataFrame):
    """Top customers and purchase frequency."""
    _section("👥 Customer Analysis")

    cust_col  = _cat_col(df, ["customer", "client", "buyer", "user"])
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount", "total"])
    order_col = _cat_col(df, ["order", "order_id", "invoice", "transaction"])

    if not cust_col:
        st.info("No customer column detected in this dataset.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        if sales_col:
            top_custs = (df.groupby(cust_col)[sales_col]
                           .sum()
                           .sort_values(ascending=False)
                           .head(10)
                           .reset_index())
            fig = px.bar(
                top_custs,
                x=cust_col, y=sales_col,
                color=sales_col,
                color_continuous_scale=[[0, "#1a2a4a"], [0.5, BRAND_SUCCESS], [1, BRAND_PRIMARY]],
                text=top_custs[sales_col].apply(_fmt),
                labels={sales_col: "Total Spend", cust_col: "Customer"},
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            _apply_layout(fig, f"Top 10 Customers by Spend", height=360)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        freq = df[cust_col].value_counts().head(10).reset_index()
        freq.columns = [cust_col, "Order Count"]
        fig = px.bar(
            freq,
            x=cust_col, y="Order Count",
            color="Order Count",
            color_continuous_scale=[[0, "#1a2a4a"], [0.5, BRAND_WARNING], [1, BRAND_SECONDARY]],
            text="Order Count",
            labels={"Order Count": "# Orders", cust_col: "Customer"},
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        _apply_layout(fig, "Customer Purchase Frequency (Top 10)", height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ── Spend distribution via box plot ──
    if sales_col:
        cust_totals = df.groupby(cust_col)[sales_col].sum().reset_index()
        fig = px.box(
            cust_totals,
            y=sales_col,
            points="all",
            color_discrete_sequence=[BRAND_PRIMARY],
            labels={sales_col: "Total Customer Spend"},
        )
        _apply_layout(fig, "Customer Spend Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 4 — PRODUCT PERFORMANCE
# ─────────────────────────────────────────────

def _render_products(df: pd.DataFrame):
    """Product performance: ranking, revenue, frequency."""
    _section("📦 Product Performance")

    prod_col  = _cat_col(df, ["product", "item", "goods", "sku", "category"])
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount", "total"])
    qty_col   = _numeric_col(df, ["quantity", "qty", "units", "volume"])

    if not prod_col:
        st.info("No product column detected.")
        return

    col_left, col_right = st.columns([3, 2])

    with col_left:
        if sales_col:
            prod_rev = (df.groupby(prod_col)[sales_col]
                          .sum()
                          .sort_values(ascending=True)
                          .tail(12)
                          .reset_index())
            fig = px.bar(
                prod_rev,
                x=sales_col, y=prod_col,
                orientation="h",
                color=sales_col,
                color_continuous_scale=[[0, "#1a1a2e"], [0.5, BRAND_SECONDARY], [1, BRAND_SUCCESS]],
                text=prod_rev[sales_col].apply(_fmt),
                labels={sales_col: "Revenue", prod_col: ""},
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            _apply_layout(fig, "Product Revenue Ranking", height=420)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        freq = df[prod_col].value_counts().head(10)
        fig = px.pie(
            freq.reset_index(),
            names=prod_col,
            values="count",
            color_discrete_sequence=COLOR_SEQ,
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=10)
        _apply_layout(fig, "Product Sales Frequency Share", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Quantity by product if available ──
    if qty_col and sales_col:
        prod_summary = (df.groupby(prod_col)
                          .agg(Revenue=(sales_col, "sum"), Quantity=(qty_col, "sum"))
                          .sort_values("Revenue", ascending=False)
                          .head(10)
                          .reset_index())

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=prod_summary[prod_col],
                y=prod_summary["Revenue"],
                name="Revenue",
                marker_color=BRAND_PRIMARY,
                opacity=0.85,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=prod_summary[prod_col],
                y=prod_summary["Quantity"],
                name="Quantity",
                mode="lines+markers",
                line=dict(color=BRAND_WARNING, width=2.5),
                marker=dict(size=7, color=BRAND_WARNING),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            **BASE_LAYOUT,
            title=dict(text="<b>Revenue vs Quantity by Product (Top 10)</b>",
                       font=dict(size=14, color=FONT_COLOR), x=0.02),
            height=360,
            xaxis=AXIS_STYLE,
            yaxis=dict(**AXIS_STYLE, title="Revenue"),
            yaxis2=dict(**AXIS_STYLE, title="Quantity", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 5 — REGIONAL PERFORMANCE
# ─────────────────────────────────────────────

def _render_regional(df: pd.DataFrame):
    """Regional breakdown with treemap and ranked bar."""
    _section("🗺️ Regional Performance")

    city_col  = _cat_col(df, ["city", "region", "location", "area", "zone", "state", "district"])
    prod_col  = _cat_col(df, ["product", "item", "goods", "sku", "category"])
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount", "total"])

    if not city_col or not sales_col:
        st.info("No regional column detected.")
        return

    col_left, col_right = st.columns([3, 2])

    with col_left:
        grp = (df.groupby(city_col)[sales_col]
                 .sum()
                 .sort_values(ascending=False)
                 .reset_index())
        total = grp[sales_col].sum()
        grp["Share %"] = (grp[sales_col] / total * 100).round(1)

        fig = px.bar(
            grp,
            x=city_col, y=sales_col,
            color=sales_col,
            color_continuous_scale=[[0, "#0d1b2a"], [0.5, BRAND_PRIMARY], [1, BRAND_GOLD]],
            text=grp["Share %"].apply(lambda v: f"{v}%"),
            labels={sales_col: "Revenue", city_col: "Region"},
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        _apply_layout(fig, f"Revenue by {city_col}", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Treemap if product col exists too
        if prod_col:
            treemap_df = (df.groupby([city_col, prod_col])[sales_col]
                            .sum()
                            .reset_index())
            fig = px.treemap(
                treemap_df,
                path=[city_col, prod_col],
                values=sales_col,
                color=sales_col,
                color_continuous_scale=[[0, "#0d1b2a"], [0.5, BRAND_SECONDARY], [1, BRAND_PRIMARY]],
            )
            fig.update_coloraxes(showscale=False)
            _apply_layout(fig, f"{city_col} × {prod_col} Treemap", height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Funnel chart as alternative
            fig = px.funnel(
                grp.head(10),
                x=sales_col, y=city_col,
                color_discrete_sequence=[BRAND_PRIMARY],
            )
            _apply_layout(fig, "Regional Revenue Funnel", height=380)
            st.plotly_chart(fig, use_container_width=True)

    # ── Region × Product heatmap ──
    if prod_col:
        try:
            pivot = df.pivot_table(index=city_col, columns=prod_col,
                                   values=sales_col, aggfunc="sum", fill_value=0)
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(10).index]
            pivot = pivot[pivot.sum().sort_values(ascending=False).head(8).index]

            fig = px.imshow(
                pivot,
                color_continuous_scale=[[0, "#0d1b2a"], [0.5, BRAND_PRIMARY], [1, BRAND_GOLD]],
                aspect="auto",
                text_auto=".0f",
            )
            fig.update_traces(textfont_size=9)
            _apply_layout(fig, f"{city_col} × {prod_col} Revenue Heatmap", height=360)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ─────────────────────────────────────────────
# SECTION 6 — TREND ANALYSIS
# ─────────────────────────────────────────────

def _render_trends(df: pd.DataFrame):
    """Monthly time-series trends with moving average."""
    _section("📅 Trend Analysis")

    date_col  = _date_col(df)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount", "total"])
    city_col  = _cat_col(df, ["city", "region", "location", "area", "zone"])
    prod_col  = _cat_col(df, ["product", "item", "goods", "sku", "category"])

    if not date_col or not sales_col:
        st.info("No date and numeric columns found — trend analysis requires a datetime column.")
        return

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])

    monthly = (tmp.resample("ME", on=date_col)[sales_col]
                  .sum()
                  .reset_index())
    monthly.columns = ["Month", "Revenue"]
    monthly["3M MA"] = monthly["Revenue"].rolling(3, min_periods=1).mean()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Revenue"],
            name="Monthly Revenue",
            marker_color=BRAND_PRIMARY,
            opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["3M MA"],
            name="3-Month MA",
            mode="lines+markers",
            line=dict(color=BRAND_WARNING, width=2.5, dash="dot"),
            marker=dict(size=5),
        ))
        _apply_layout(fig, "Monthly Revenue Trend", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # MoM growth
        monthly["MoM Growth %"] = monthly["Revenue"].pct_change() * 100
        fig = px.bar(
            monthly.dropna(subset=["MoM Growth %"]),
            x="Month", y="MoM Growth %",
            color="MoM Growth %",
            color_continuous_scale=[[0, BRAND_DANGER], [0.5, "#333"], [1, BRAND_SUCCESS]],
            labels={"MoM Growth %": "Growth %"},
        )
        fig.update_coloraxes(showscale=False)
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        _apply_layout(fig, "Month-over-Month Growth %", height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ── Multi-line trend by category ──
    if city_col and len(tmp[city_col].dropna().unique()) <= 15:
        try:
            city_monthly = (tmp.groupby([pd.Grouper(key=date_col, freq="ME"), city_col])[sales_col]
                               .sum()
                               .reset_index())
            city_monthly.columns = ["Month", city_col, "Revenue"]
            fig = px.line(
                city_monthly,
                x="Month", y="Revenue",
                color=city_col,
                markers=True,
                color_discrete_sequence=COLOR_SEQ,
                labels={"Revenue": "Revenue", "Month": ""},
            )
            _apply_layout(fig, f"Monthly Revenue by {city_col}", height=360)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ─────────────────────────────────────────────
# SECTION 7 — CORRELATION INSIGHTS
# ─────────────────────────────────────────────

def _render_correlations(df: pd.DataFrame):
    """Interactive Plotly correlation heatmap + scatter matrix."""
    _section("🔗 Correlation Insights")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
        return

    col_left, col_right = st.columns([2, 3])

    with col_left:
        corr = df[num_cols].corr().round(2)
        fig  = px.imshow(
            corr,
            color_continuous_scale=[[0, BRAND_DANGER], [0.5, "#222"], [1, BRAND_SUCCESS]],
            zmin=-1, zmax=1,
            text_auto=True,
            aspect="auto",
        )
        fig.update_traces(textfont_size=11)
        fig.update_coloraxes(colorbar=dict(
            title="r",
            tickfont=dict(color=FONT_COLOR),
            titlefont=dict(color=FONT_COLOR),
        ))
        _apply_layout(fig, "Correlation Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Ranked correlation pairs
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    pairs.append({
                        "Column A": num_cols[i],
                        "Column B": num_cols[j],
                        "Correlation": round(float(val), 3),
                        "Abs":         abs(float(val)),
                    })
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values("Abs", ascending=False).drop("Abs", axis=1)
            fig = px.bar(
                pairs_df.head(12),
                x="Correlation",
                y=pairs_df.head(12).apply(lambda r: f"{r['Column A']} ↔ {r['Column B']}", axis=1),
                orientation="h",
                color="Correlation",
                color_continuous_scale=[[0, BRAND_DANGER], [0.5, "#333"], [1, BRAND_SUCCESS]],
                range_x=[-1, 1],
            )
            fig.update_coloraxes(showscale=False)
            fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            _apply_layout(fig, "Ranked Correlation Pairs", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Scatter plot for top correlated pair ──
    if pairs:
        top = pairs_df.iloc[0]
        cat_col_scatter = _cat_col(df, ["city", "region", "product", "category"])
        scatter_color   = cat_col_scatter if cat_col_scatter else None

        fig = px.scatter(
            df,
            x=top["Column A"],
            y=top["Column B"],
            color=scatter_color,
            trendline="ols",
            trendline_color_override=BRAND_WARNING,
            opacity=0.65,
            color_discrete_sequence=COLOR_SEQ,
            labels={top["Column A"]: top["Column A"], top["Column B"]: top["Column B"]},
        )
        _apply_layout(
            fig,
            f"Scatter: {top['Column A']} vs {top['Column B']} (r={top['Correlation']})",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 8 — DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────

def _render_distributions(df: pd.DataFrame):
    """Histograms + box plots for all numeric columns."""
    _section("📊 Distribution Analysis")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns available.")
        return

    cat_color_col = _cat_col(df, ["city", "region", "product", "category"])
    ncols = 3
    rows_needed = (len(num_cols) + ncols - 1) // ncols

    for row in range(rows_needed):
        cols_batch = num_cols[row * ncols: row * ncols + ncols]
        st_cols = st.columns(ncols)
        for j, col in enumerate(cols_batch):
            with st_cols[j]:
                try:
                    fig = px.histogram(
                        df,
                        x=col,
                        color=cat_color_col if cat_color_col else None,
                        marginal="box",
                        nbins=25,
                        opacity=0.8,
                        color_discrete_sequence=COLOR_SEQ,
                        labels={col: col},
                    )
                    _apply_layout(fig, f"Distribution: {col}", height=310)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot `{col}`: {e}")

    # ── Violin plot for all numeric cols together ──
    if len(num_cols) >= 2:
        try:
            melted = df[num_cols].copy()
            # Normalize for comparable scale
            melted_norm = (melted - melted.min()) / (melted.max() - melted.min() + 1e-9)
            melted_long = melted_norm.melt(var_name="Column", value_name="Normalized Value")

            fig = px.violin(
                melted_long,
                x="Column", y="Normalized Value",
                box=True,
                points=False,
                color="Column",
                color_discrete_sequence=COLOR_SEQ,
            )
            _apply_layout(fig, "Normalized Distribution Comparison (All Numeric Columns)", height=360)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ─────────────────────────────────────────────
# BUSINESS INSIGHTS PANEL
# ─────────────────────────────────────────────

def _render_business_insights(df: pd.DataFrame):
    """Auto-generated plain-English key business insights panel."""
    _section("💡 Key Business Insights")

    insights = []

    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount", "total"])
    city_col  = _cat_col(df, ["city", "region", "location", "area", "zone"])
    prod_col  = _cat_col(df, ["product", "item", "goods", "sku", "category"])
    cust_col  = _cat_col(df, ["customer", "client", "buyer", "user"])
    qty_col   = _numeric_col(df, ["quantity", "qty", "units", "volume"])
    date_col  = _date_col(df)

    # Total revenue
    if sales_col:
        total = df[sales_col].sum()
        insights.append(("💰", "Total Revenue", f"**{_fmt(total)}** across {len(df):,} records"))

    # Top city
    if city_col and sales_col:
        grp = df.groupby(city_col)[sales_col].sum()
        top_city  = grp.idxmax()
        top_val   = grp.max()
        share_pct = top_val / grp.sum() * 100
        insights.append(("🏆", "Top Revenue Region",
                          f"**{top_city}** — {_fmt(top_val)} ({share_pct:.1f}% of total)"))

    # Top product
    if prod_col and sales_col:
        grp     = df.groupby(prod_col)[sales_col].sum()
        top_p   = grp.idxmax()
        top_pv  = grp.max()
        insights.append(("🥇", "Best-Selling Product",
                          f"**{top_p}** — {_fmt(top_pv)} revenue"))

    # Top customer
    if cust_col and sales_col:
        grp    = df.groupby(cust_col)[sales_col].sum()
        top_c  = grp.idxmax()
        top_cv = grp.max()
        insights.append(("⭐", "Highest-Value Customer",
                          f"**{top_c}** — {_fmt(top_cv)} total spend"))

    # AOV
    if sales_col:
        aov = df[sales_col].mean()
        insights.append(("🧾", "Average Order Value", f"**{_fmt(aov)}** per transaction"))

    # Total quantity
    if qty_col:
        total_qty = df[qty_col].sum()
        insights.append(("📦", "Total Units Sold", f"**{_fmt(total_qty)}** units"))

    # Worst city
    if city_col and sales_col:
        grp       = df.groupby(city_col)[sales_col].sum()
        low_city  = grp.idxmin()
        low_val   = grp.min()
        insights.append(("📉", "Lowest Revenue Region",
                          f"**{low_city}** — {_fmt(low_val)} (attention needed)"))

    # Trend direction
    if date_col and sales_col:
        try:
            tmp = df.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            monthly = tmp.resample("ME", on=date_col)[sales_col].sum()
            if len(monthly) >= 2:
                delta = monthly.iloc[-1] - monthly.iloc[-2]
                arrow = "📈" if delta > 0 else "📉"
                trend_label = "Growing" if delta > 0 else "Declining"
                insights.append((arrow, "Latest Month Trend",
                                  f"**{trend_label}** — {_fmt(abs(delta))} vs previous month"))
        except Exception:
            pass

    # Render insight cards in a 2-column grid
    if not insights:
        st.info("Not enough data to generate business insights.")
        return

    cols = st.columns(2)
    for i, (icon, title, body) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border: 1px solid rgba(162,89,255,0.2);
                border-radius: 10px;
                padding: 0.85rem 1rem;
                margin-bottom: 0.6rem;
            ">
                <div style="font-size:1.3rem; margin-bottom:0.15rem;">{icon}
                    <span style="font-size:0.75rem; color:#888;
                                 text-transform:uppercase; letter-spacing:0.06em;
                                 margin-left:0.4rem;">{title}</span>
                </div>
                <div style="font-size:0.95rem; color:#ddd;">{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────

def _apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render interactive sidebar filters and return a filtered DataFrame.
    Filters are shown only when relevant columns exist.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ Dashboard Filters")

    filtered = df.copy()

    city_col  = _cat_col(df, ["city", "region", "location", "area", "zone"])
    prod_col  = _cat_col(df, ["product", "item", "goods", "sku", "category"])
    date_col  = _date_col(df)
    sales_col = _numeric_col(df, ["price", "sales", "revenue", "amount"])

    # City / Region filter
    if city_col:
        cities = sorted(df[city_col].dropna().astype(str).unique().tolist())
        selected_cities = st.sidebar.multiselect(
            f"📍 Filter by {city_col}",
            options=cities,
            default=[],
            placeholder="All regions",
        )
        if selected_cities:
            filtered = filtered[filtered[city_col].isin(selected_cities)]

    # Product filter
    if prod_col:
        products = sorted(df[prod_col].dropna().astype(str).unique().tolist())
        selected_prods = st.sidebar.multiselect(
            f"📦 Filter by {prod_col}",
            options=products,
            default=[],
            placeholder="All products",
        )
        if selected_prods:
            filtered = filtered[filtered[prod_col].isin(selected_prods)]

    # Date range filter
    if date_col:
        try:
            tmp_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not tmp_dates.empty:
                min_date = tmp_dates.min().date()
                max_date = tmp_dates.max().date()
                date_range = st.sidebar.date_input(
                    "📅 Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    start, end = date_range
                    tmp_col = pd.to_datetime(filtered[date_col], errors="coerce")
                    filtered = filtered[
                        (tmp_col.dt.date >= start) & (tmp_col.dt.date <= end)
                    ]
        except Exception:
            pass

    # Revenue range slider
    if sales_col:
        try:
            min_val = float(df[sales_col].min())
            max_val = float(df[sales_col].max())
            if min_val < max_val:
                rev_range = st.sidebar.slider(
                    f"💰 {sales_col} Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    format="%.0f",
                )
                filtered = filtered[
                    (filtered[sales_col] >= rev_range[0]) &
                    (filtered[sales_col] <= rev_range[1])
                ]
        except Exception:
            pass

    # Filter status badge
    pct_shown = len(filtered) / max(len(df), 1) * 100
    st.sidebar.markdown(f"""
    <div style="
        background: rgba(79,139,249,0.12);
        border: 1px solid rgba(79,139,249,0.3);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.82rem;
        color: #aaa;
    ">
        Showing <b style="color:{BRAND_PRIMARY};">{len(filtered):,}</b> of
        <b>{len(df):,}</b> rows ({pct_shown:.1f}%)
    </div>
    """, unsafe_allow_html=True)

    return filtered


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_dashboard(df: pd.DataFrame):
    """
    Render the full executive analytics dashboard for a given DataFrame.

    Sections rendered:
        1. Sidebar filters (interactive — returns filtered df)
        2. KPI Overview cards
        3. Revenue & Sales Performance
        4. Customer Analysis
        5. Product Performance
        6. Regional Performance
        7. Trend Analysis
        8. Correlation Insights
        9. Distribution Analysis
        10. Key Business Insights panel

    Args:
        df: The cleaned DataFrame to visualise.
    """
    if df is None or df.empty:
        st.warning("Dashboard requires a non-empty dataset.")
        return

    # Apply sidebar filters — all charts use filtered df
    df_filtered = _apply_sidebar_filters(df)

    if df_filtered.empty:
        st.warning("⚠️ Current filters result in an empty dataset. Adjust the filters in the sidebar.")
        return

    _render_kpis(df_filtered)
    _render_business_insights(df_filtered)
    _render_revenue(df_filtered)
    _render_customers(df_filtered)
    _render_products(df_filtered)
    _render_regional(df_filtered)
    _render_trends(df_filtered)
    _render_correlations(df_filtered)
    _render_distributions(df_filtered)
