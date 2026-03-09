"""
dashboard_generator.py
Executive Analytics Dashboard — tab-based lazy Plotly rendering.
No statsmodels or heavy deps. All charts are lightweight and fast.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics_engine import (
    group_by_col, trend_analysis, correlation_matrix,
    distribution_metrics, _num, _cat, _date, _fmt,
)

# ── Design tokens ──────────────────────────────────────────────────────────────

COLORS = ["#4F8BF9", "#A259FF", "#43D9AD", "#FFB347", "#FF6B6B",
          "#87CEEB", "#FF69B4", "#98FB98", "#DDA0DD", "#F0E68C"]

_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter,sans-serif", color="#E0E0E0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#1E1E2E", bordercolor="#444", font=dict(size=12, color="#fff")),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)
_AX = dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
           zeroline=False, linecolor="rgba(255,255,255,0.3)",
           tickfont=dict(size=11, color="#E0E0E0"))


def _lay(fig: go.Figure, title: str = "", h: int = 390) -> go.Figure:
    fig.update_layout(
        **_BASE, height=h,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#E0E0E0"), x=0.02),
        xaxis=_AX, yaxis=_AX,
    )
    return fig


def _hdr(label: str):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,#4F8BF922,transparent);'
        f'border-left:4px solid #4F8BF9;border-radius:0 8px 8px 0;'
        f'padding:.4rem 1rem;margin:.5rem 0 .75rem 0;">'
        f'<span style="font-size:1rem;font-weight:700;color:#4F8BF9;">{label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Sidebar filters ────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, kpis: Dict) -> pd.DataFrame:
    """Render sidebar filter widgets and return filtered df."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ Dashboard Filters")
    out = df

    city_col  = kpis.get("city_col")
    prod_col  = kpis.get("prod_col")
    date_col  = kpis.get("date_col")
    sales_col = kpis.get("sales_col")

    if city_col and city_col in df.columns:
        opts = sorted(df[city_col].dropna().astype(str).unique().tolist())
        sel  = st.sidebar.multiselect(f"📍 {city_col}", opts, default=[], placeholder="All")
        if sel:
            out = out[out[city_col].isin(sel)]

    if prod_col and prod_col in df.columns:
        opts = sorted(df[prod_col].dropna().astype(str).unique().tolist())
        sel  = st.sidebar.multiselect(f"📦 {prod_col}", opts, default=[], placeholder="All")
        if sel:
            out = out[out[prod_col].isin(sel)]

    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                rng = st.sidebar.date_input(
                    "📅 Date Range",
                    value=(dates.min().date(), dates.max().date()),
                    min_value=dates.min().date(),
                    max_value=dates.max().date(),
                )
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    s, e = rng
                    tmp = pd.to_datetime(out[date_col], errors="coerce")
                    out = out[(tmp.dt.date >= s) & (tmp.dt.date <= e)]
        except Exception:
            pass

    if sales_col and sales_col in df.columns:
        try:
            lo, hi = float(df[sales_col].min()), float(df[sales_col].max())
            if lo < hi:
                rv = st.sidebar.slider(f"💰 {sales_col}", lo, hi, (lo, hi), format="%.0f")
                out = out[(out[sales_col] >= rv[0]) & (out[sales_col] <= rv[1])]
        except Exception:
            pass

    pct = len(out) / max(len(df), 1) * 100
    st.sidebar.markdown(
        f'<div style="background:rgba(79,139,249,.1);border:1px solid rgba(79,139,249,.3);'
        f'border-radius:8px;padding:.5rem .75rem;font-size:.82rem;color:#aaa;">'
        f'Showing <b style="color:#4F8BF9">{len(out):,}</b> of <b>{len(df):,}</b> rows ({pct:.1f}%)'
        f"</div>",
        unsafe_allow_html=True,
    )
    return out


# ── KPI cards ──────────────────────────────────────────────────────────────────

def render_kpis(kpis: Dict):
    _hdr("📌 Executive KPIs")
    cards = [
        ("💰", "Total Revenue",    kpis.get("total_revenue",    "—")),
        ("🧾", "Total Orders",     kpis.get("total_orders",     "—")),
        ("📊", "Avg Order Value",  kpis.get("aov",              "—")),
        ("👥", "Customers",        kpis.get("unique_customers", "—")),
        ("📦", "Products",         kpis.get("unique_products",  "—")),
        ("🗺️","Regions",          kpis.get("unique_regions",   "—")),
    ]
    for col, (icon, label, value) in zip(st.columns(6), cards):
        col.markdown(
            f'<div style="background:linear-gradient(135deg,#1E1E2E,#2a2a3e);'
            f'border:1px solid rgba(79,139,249,.25);border-radius:12px;'
            f'padding:.9rem .6rem;text-align:center;">'
            f'<div style="font-size:1.4rem">{icon}</div>'
            f'<div style="font-size:1.3rem;font-weight:800;color:#4F8BF9;margin:.15rem 0">{value}</div>'
            f'<div style="font-size:.68rem;color:#888;text-transform:uppercase;letter-spacing:.05em">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)


# ── Insight cards ──────────────────────────────────────────────────────────────

def _insights_panel(df: pd.DataFrame, kpis: Dict):
    _hdr("💡 Key Business Insights")
    sales_col = kpis.get("sales_col")
    city_col  = kpis.get("city_col")
    prod_col  = kpis.get("prod_col")
    cust_col  = kpis.get("cust_col")
    qty_col   = kpis.get("qty_col")
    date_col  = kpis.get("date_col")
    cards: List[tuple] = []

    if sales_col:
        cards.append(("💰","Total Revenue",
                      f"<b>{kpis.get('total_revenue','—')}</b> across {len(df):,} records"))
    if city_col and sales_col:
        grp = df.groupby(city_col, observed=True)[sales_col].sum()
        top = grp.idxmax(); tv = grp.max(); sh = tv/grp.sum()*100
        low = grp.idxmin(); lv = grp.min()
        cards.append(("🏆","Top Region",    f"<b>{top}</b> — {_fmt(tv)} ({sh:.0f}% of total)"))
        cards.append(("📉","Lowest Region", f"<b>{low}</b> — {_fmt(lv)}"))
    if prod_col and sales_col:
        grp = df.groupby(prod_col, observed=True)[sales_col].sum()
        top = grp.idxmax(); tv = grp.max()
        cards.append(("🥇","Best Product", f"<b>{top}</b> — {_fmt(tv)} revenue"))
    if cust_col and sales_col:
        grp = df.groupby(cust_col, observed=True)[sales_col].sum()
        top = grp.idxmax(); tv = grp.max()
        cards.append(("⭐","Top Customer", f"<b>{top}</b> — {_fmt(tv)} spend"))
    if qty_col:
        cards.append(("📦","Total Units", f"<b>{_fmt(float(df[qty_col].sum()))}</b> units sold"))
    if date_col and sales_col:
        try:
            tmp = df[[date_col, sales_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            monthly = tmp.resample("ME", on=date_col)[sales_col].sum()
            if len(monthly) >= 2:
                d = monthly.iloc[-1] - monthly.iloc[-2]
                arrow = "📈" if d > 0 else "📉"
                cards.append((arrow, "Latest Trend",
                              f"{'Growing' if d > 0 else 'Declining'} — {_fmt(abs(d))} vs prior month"))
        except Exception:
            pass

    cols2 = st.columns(2)
    for i, (icon, title, body) in enumerate(cards):
        with cols2[i % 2]:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
                f'border:1px solid rgba(162,89,255,.2);border-radius:10px;'
                f'padding:.8rem 1rem;margin-bottom:.6rem;">'
                f'<span style="font-size:1.2rem">{icon}</span>'
                f'<span style="font-size:.72rem;color:#888;text-transform:uppercase;'
                f'letter-spacing:.06em;margin-left:.4rem">{title}</span>'
                f'<div style="font-size:.92rem;color:#ddd;margin-top:.2rem">{body}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Tab renderers ──────────────────────────────────────────────────────────────

def _tab_overview(df: pd.DataFrame, kpis: Dict):
    sc = kpis.get("sales_col"); cc = kpis.get("city_col"); pc = kpis.get("prod_col")
    c1, c2 = st.columns(2)
    with c1:
        if cc and sc:
            d = group_by_col(df, cc, sc, "sum", 10).sort_values(sc, ascending=True)
            fig = px.bar(d, x=sc, y=cc, orientation="h", color=sc,
                         color_continuous_scale=[[0,"#1a2a4a"],[.5,"#4F8BF9"],[1,"#A259FF"]],
                         text=d[sc].apply(_fmt))
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            _lay(fig, f"Revenue by {cc}"); st.plotly_chart(fig, use_container_width=True)
    with c2:
        if pc and sc:
            d = group_by_col(df, pc, sc, "sum", 8)
            fig = px.pie(d, names=pc, values=sc, hole=.42, color_discrete_sequence=COLORS)
            fig.update_traces(textinfo="percent+label", textfont_size=11,
                              pull=[.04]+[0]*(len(d)-1))
            _lay(fig, f"Revenue Share by {pc}"); st.plotly_chart(fig, use_container_width=True)


def _tab_sales(df: pd.DataFrame, kpis: Dict):
    sc = kpis.get("sales_col"); cc = kpis.get("city_col"); pc = kpis.get("prod_col")
    if not sc:
        st.info("No numeric sales column detected."); return
    if pc:
        d   = group_by_col(df, pc, sc, "sum", 10)
        fig = px.bar(d, x=pc, y=sc, color=sc,
                     color_continuous_scale=[[0,"#1a2a4a"],[.5,"#A259FF"],[1,"#4F8BF9"]],
                     text=d[sc].apply(_fmt))
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        _lay(fig, f"Top 10 {pc}s by Revenue"); st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if cc and sc:
            d   = df.groupby(cc, observed=True)[sc].sum().sort_values(ascending=False).head(8).reset_index()
            fig = px.funnel(d, x=sc, y=cc, color_discrete_sequence=["#4F8BF9"])
            _lay(fig, "Revenue Funnel by Region", h=360); st.plotly_chart(fig, use_container_width=True)
    with c2:
        if pc and cc and sc:
            try:
                piv = df.pivot_table(index=cc, columns=pc, values=sc,
                                     aggfunc="sum", fill_value=0, observed=True)
                piv = piv.loc[piv.sum(1).sort_values(ascending=False).head(8).index]
                piv = piv[piv.sum().sort_values(ascending=False).head(6).index]
                fig = px.imshow(piv, aspect="auto", text_auto=".0f",
                                color_continuous_scale=[[0,"#0d1b2a"],[.5,"#4F8BF9"],[1,"#FFB347"]])
                fig.update_traces(textfont_size=9)
                _lay(fig, f"{cc} × {pc} Heatmap", h=360); st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass


def _tab_customers(df: pd.DataFrame, kpis: Dict):
    cu = kpis.get("cust_col"); sc = kpis.get("sales_col")
    if not cu:
        st.info("No customer column detected."); return
    c1, c2 = st.columns(2)
    with c1:
        if sc:
            d   = group_by_col(df, cu, sc, "sum", 10).sort_values(sc, ascending=True)
            fig = px.bar(d, x=sc, y=cu, orientation="h", color=sc,
                         color_continuous_scale=[[0,"#1a2a4a"],[.5,"#43D9AD"],[1,"#4F8BF9"]],
                         text=d[sc].apply(_fmt))
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            _lay(fig, "Top 10 Customers by Revenue"); st.plotly_chart(fig, use_container_width=True)
    with c2:
        freq = df[cu].value_counts().head(10).reset_index()
        freq.columns = [cu, "Orders"]
        freq = freq.sort_values("Orders", ascending=True)
        fig  = px.bar(freq, x="Orders", y=cu, orientation="h", color="Orders",
                      color_continuous_scale=[[0,"#1a2a4a"],[.5,"#FFB347"],[1,"#A259FF"]],
                      text="Orders")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        _lay(fig, "Customer Purchase Frequency"); st.plotly_chart(fig, use_container_width=True)
    if sc:
        totals = df.groupby(cu, observed=True)[sc].sum().reset_index()
        fig    = px.box(totals, y=sc, points="outliers", color_discrete_sequence=["#4F8BF9"])
        _lay(fig, "Customer Spend Distribution", h=300); st.plotly_chart(fig, use_container_width=True)


def _tab_products(df: pd.DataFrame, kpis: Dict):
    pc = kpis.get("prod_col"); sc = kpis.get("sales_col"); qc = kpis.get("qty_col")
    if not pc:
        st.info("No product column detected."); return
    c1, c2 = st.columns([3, 2])
    with c1:
        if sc:
            d   = group_by_col(df, pc, sc, "sum", 12).sort_values(sc, ascending=True)
            fig = px.bar(d, x=sc, y=pc, orientation="h", color=sc,
                         color_continuous_scale=[[0,"#1a1a2e"],[.5,"#A259FF"],[1,"#43D9AD"]],
                         text=d[sc].apply(_fmt))
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            _lay(fig, "Product Revenue Ranking", h=420); st.plotly_chart(fig, use_container_width=True)
    with c2:
        freq = df[pc].value_counts().head(10)
        fig  = px.pie(freq.reset_index(), names=pc, values="count",
                      hole=.4, color_discrete_sequence=COLORS)
        fig.update_traces(textinfo="percent+label", textfont_size=10)
        _lay(fig, "Product Frequency Share", h=420); st.plotly_chart(fig, use_container_width=True)
    if qc and sc:
        summ = (df.groupby(pc, observed=True)
                  .agg(Revenue=(sc,"sum"), Quantity=(qc,"sum"))
                  .sort_values("Revenue", ascending=False).head(10).reset_index())
        fig  = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=summ[pc], y=summ["Revenue"], name="Revenue",
                             marker_color="#4F8BF9", opacity=.85), secondary_y=False)
        fig.add_trace(go.Scatter(x=summ[pc], y=summ["Quantity"], name="Quantity",
                                 mode="lines+markers",
                                 line=dict(color="#FFB347", width=2.5),
                                 marker=dict(size=7, color="#FFB347")), secondary_y=True)
        fig.update_layout(**_BASE, height=340,
                          title=dict(text="<b>Revenue vs Quantity (Top 10)</b>",
                                     font=dict(size=14, color="#E0E0E0"), x=.02),
                          xaxis=_AX,
                          yaxis=dict(**_AX, title="Revenue"),
                          yaxis2=dict(**_AX, title="Quantity", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)


def _tab_regional(df: pd.DataFrame, kpis: Dict):
    cc = kpis.get("city_col"); sc = kpis.get("sales_col"); pc = kpis.get("prod_col")
    if not cc or not sc:
        st.info("No regional column detected."); return
    grp   = df.groupby(cc, observed=True)[sc].sum().sort_values(ascending=False).reset_index()
    total = grp[sc].sum()
    grp["Share %"] = (grp[sc] / total * 100).round(1)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(grp, x=cc, y=sc, color=sc,
                     color_continuous_scale=[[0,"#0d1b2a"],[.5,"#4F8BF9"],[1,"#F5C842"]],
                     text=grp["Share %"].apply(lambda v: f"{v}%"))
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        _lay(fig, f"Revenue by {cc}"); st.plotly_chart(fig, use_container_width=True)
    with c2:
        if pc:
            tm = df.groupby([cc, pc], observed=True)[sc].sum().reset_index()
            fig = px.treemap(tm, path=[cc, pc], values=sc, color=sc,
                             color_continuous_scale=[[0,"#0d1b2a"],[.5,"#A259FF"],[1,"#4F8BF9"]])
            fig.update_coloraxes(showscale=False)
            _lay(fig, f"{cc} × {pc} Treemap"); st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.funnel(grp.head(10), x=sc, y=cc, color_discrete_sequence=["#4F8BF9"])
            _lay(fig, "Regional Funnel"); st.plotly_chart(fig, use_container_width=True)


def _tab_trends(df: pd.DataFrame, kpis: Dict):
    result = trend_analysis(df)
    if result is None:
        st.info("No date + numeric column found for trend analysis."); return
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=result["month"], y=result["revenue"],
                             name="Revenue", marker_color="#4F8BF9", opacity=.7))
        fig.add_trace(go.Scatter(x=result["month"], y=result["ma3"],
                                 name="3M MA", mode="lines+markers",
                                 line=dict(color="#FFB347", width=2.5, dash="dot"),
                                 marker=dict(size=5)))
        _lay(fig, "Monthly Revenue + 3-Month MA"); st.plotly_chart(fig, use_container_width=True)
    with c2:
        mom = result.dropna(subset=["mom_growth"]).copy()
        fig = px.bar(mom, x="month", y="mom_growth", color="mom_growth",
                     color_continuous_scale=[[0,"#FF6B6B"],[.5,"#333"],[1,"#43D9AD"]])
        fig.update_coloraxes(showscale=False)
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,.3)")
        _lay(fig, "Month-over-Month Growth %"); st.plotly_chart(fig, use_container_width=True)


def _tab_correlations(df: pd.DataFrame):
    corr = correlation_matrix(df)
    if corr is None:
        st.info("Need ≥ 2 numeric columns for correlation analysis."); return
    c1, c2 = st.columns([2, 3])
    with c1:
        fig = px.imshow(corr,
                        color_continuous_scale=[[0,"#FF6B6B"],[.5,"#222"],[1,"#43D9AD"]],
                        zmin=-1, zmax=1, text_auto=True, aspect="auto")
        fig.update_traces(textfont_size=11)
        _lay(fig, "Correlation Matrix", h=420); st.plotly_chart(fig, use_container_width=True)
    with c2:
        num_cols = corr.columns.tolist()
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    pairs.append({"Pair": f"{num_cols[i]} ↔ {num_cols[j]}",
                                  "r": float(v), "_abs": abs(float(v))})
        if pairs:
            pf  = pd.DataFrame(pairs).sort_values("_abs", ascending=False).head(12)
            pf2 = pf[["Pair","r"]].copy()
            fig = px.bar(pf2, x="r", y="Pair", orientation="h", color="r",
                         color_continuous_scale=[[0,"#FF6B6B"],[.5,"#333"],[1,"#43D9AD"]],
                         range_x=[-1,1])
            fig.update_coloraxes(showscale=False)
            fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,.3)")
            _lay(fig, "Ranked Correlation Pairs", h=420); st.plotly_chart(fig, use_container_width=True)


def _tab_distributions(df: pd.DataFrame):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if not nums:
        st.info("No numeric columns for distribution analysis."); return
    cat_col = None
    for col in df.select_dtypes(include=["object","category"]).columns:
        if df[col].nunique() <= 15:
            cat_col = col; break
    for i in range(0, len(nums), 3):
        batch = nums[i:i+3]
        st_c  = st.columns(3)
        for j, col in enumerate(batch):
            with st_c[j]:
                try:
                    # sample for large datasets to keep rendering fast
                    plot_df = df if len(df) <= 50_000 else df.sample(20_000, random_state=42)
                    fig = px.histogram(plot_df, x=col, color=cat_col, nbins=25,
                                       opacity=.8, marginal="box",
                                       color_discrete_sequence=COLORS)
                    _lay(fig, f"Distribution: {col}", h=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Cannot plot `{col}`: {e}")
    if len(nums) >= 2:
        try:
            sample_df = df[nums] if len(df) <= 50_000 else df[nums].sample(20_000, random_state=42)
            melted = sample_df.copy()
            melted = (melted - melted.min()) / (melted.max() - melted.min() + 1e-9)
            long   = melted.melt(var_name="Column", value_name="Normalised")
            fig    = px.violin(long, x="Column", y="Normalised", box=True,
                               points=False, color="Column",
                               color_discrete_sequence=COLORS)
            _lay(fig, "Normalised Distribution Comparison", h=340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_dashboard(df: pd.DataFrame, kpis: Dict[str, Any]):
    """
    Render the full executive dashboard with lazy tab rendering.
    Charts only render when the tab is active — keeps initial load fast.
    """
    if df is None or df.empty:
        st.warning("Dashboard requires a non-empty dataset."); return

    render_kpis(kpis)
    _insights_panel(df, kpis)
    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "🌐 Overview", "💹 Sales", "👥 Customers", "📦 Products",
        "🗺️ Regional", "📅 Trends", "🔗 Correlations", "📊 Distributions",
    ])
    with tabs[0]: _tab_overview(df, kpis)
    with tabs[1]: _tab_sales(df, kpis)
    with tabs[2]: _tab_customers(df, kpis)
    with tabs[3]: _tab_products(df, kpis)
    with tabs[4]: _tab_regional(df, kpis)
    with tabs[5]: _tab_trends(df, kpis)
    with tabs[6]: _tab_correlations(df)
    with tabs[7]: _tab_distributions(df)
