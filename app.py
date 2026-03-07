"""
app.py
AI Data Analyst Copilot — Streamlit Application
Extends the data cleaning tool with insights, AI summaries, and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

from analyzer import profile_dataset, detect_issues, generate_suggestions
from cleaner import apply_cleaning
from insights import generate_insights
from ai_insights import generate_ai_summary
from query_engine_v2 import answer_query
from utils import (
    load_file, compute_quality_score,
    df_to_csv_bytes, df_to_excel_bytes,
    format_profile_table, format_issues_table,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Analyst Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #4F8BF9, #A259FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #A259FF;
        margin-top: 1.5rem; border-bottom: 2px solid #A259FF22;
        padding-bottom: 0.3rem;
    }
    .insight-card {
        background: #1a1a2e; border-left: 4px solid #4F8BF9;
        border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
        font-size: 0.93rem;
    }
    .ai-card {
        background: #0f2027; border-left: 4px solid #A259FF;
        border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
        font-size: 0.93rem;
    }
    .severity-high   { color: #FF4B4B; font-weight: 700; }
    .severity-medium { color: #FFA500; font-weight: 600; }
    .severity-low    { color: #2196F3; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#0E1117",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#ccc",
    "xtick.color":       "#aaa",
    "ytick.color":       "#aaa",
    "text.color":        "#ddd",
    "grid.color":        "#333",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})
PALETTE = ["#4F8BF9", "#A259FF", "#FF6B6B", "#43D9AD", "#FFB347",
           "#87CEEB", "#FF69B4", "#98FB98", "#DDA0DD", "#F0E68C"]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">🤖 AI Data Analyst Copilot</div>', unsafe_allow_html=True)
st.markdown("Automated cleaning · intelligent insights · natural language analysis · instant visualizations")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Supported: CSV, XLSX, XLS",
        type=["csv", "xlsx", "xls"],
    )
    st.divider()
    st.markdown("### ⚙️ Settings")
    preview_rows     = st.slider("Preview rows", 5, 50, 10)
    show_raw_profile = st.checkbox("Show raw column profile", value=False)
    st.divider()
    st.markdown("### 📊 Visualization Settings")
    chart_bins       = st.slider("Histogram bins", 5, 50, 15)
    max_bar_cats     = st.slider("Max bar chart categories", 5, 20, 10)

# ─────────────────────────────────────────────
# NO FILE STATE
# ─────────────────────────────────────────────

if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file from the sidebar to get started.")
    st.markdown("""
    **This tool will automatically:**
    - 🧹 Detect and fix 14+ categories of data quality issues
    - 📊 Generate numeric & categorical insights
    - 🤖 Write a natural language analysis summary
    - 📈 Produce distribution, bar, and correlation charts
    - 📥 Export clean data as CSV or Excel
    """)
    st.stop()

# ─────────────────────────────────────────────
# LOAD FILE
# ─────────────────────────────────────────────

df_raw, error = load_file(uploaded_file)
if error:
    st.error(f"❌ {error}")
    st.stop()

st.success(f"✅ Loaded **{uploaded_file.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

# ─────────────────────────────────────────────
# DATASET PREVIEW
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(df_raw.head(preview_rows), use_container_width=True)

# ─────────────────────────────────────────────
# PROFILING & ISSUE DETECTION
# ─────────────────────────────────────────────

with st.spinner("🔍 Profiling dataset..."):
    profile     = profile_dataset(df_raw)
    issues      = detect_issues(df_raw, profile)
    suggestions = generate_suggestions(df_raw, profile, issues)
    score, score_label, score_colour = compute_quality_score(profile, issues)

# ─────────────────────────────────────────────
# QUALITY DASHBOARD
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Data Quality Dashboard</div>', unsafe_allow_html=True)

rows, cols_count = profile["shape"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Rows",      f"{rows:,}")
c2.metric("Columns",         cols_count)
c3.metric("Duplicate Rows",  profile["duplicate_rows"])
c4.metric("Missing Values",  profile["total_missing"])
c5.metric("Issues Found",    len(issues))
c6.metric("Quality Score",   f"{score} / 100")

bar_colour = {"green": "#4CAF50", "orange": "#FFA500",
              "darkorange": "#FF5722", "red": "#FF1744"}.get(score_colour, "#888")
st.markdown(f"""
<div style="margin-top:0.5rem">
  <div style="background:#2E2E3E;border-radius:8px;height:18px;width:100%">
    <div style="background:{bar_colour};width:{score}%;height:18px;border-radius:8px;
                display:flex;align-items:center;justify-content:center;
                font-size:0.7rem;color:white;font-weight:700;">
      {score_label}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔬 Column Profile", expanded=False):
    st.dataframe(format_profile_table(profile), use_container_width=True, hide_index=True)

if show_raw_profile:
    with st.expander("🗂️ Raw Profile"):
        st.json(profile)

# ─────────────────────────────────────────────
# DETECTED ISSUES
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">🚨 Detected Issues</div>', unsafe_allow_html=True)
if not issues:
    st.success("No issues detected — dataset looks clean! ✨")
else:
    issues_df = format_issues_table(issues)
    high_n = sum(1 for i in issues if i["severity"] == "High")
    med_n  = sum(1 for i in issues if i["severity"] == "Medium")
    low_n  = sum(1 for i in issues if i["severity"] == "Low")
    ih, im, il = st.columns(3)
    ih.markdown(f'<span class="severity-high">🔴 High: {high_n}</span>', unsafe_allow_html=True)
    im.markdown(f'<span class="severity-medium">🟠 Medium: {med_n}</span>', unsafe_allow_html=True)
    il.markdown(f'<span class="severity-low">🔵 Low: {low_n}</span>', unsafe_allow_html=True)
    st.dataframe(issues_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# CLEANING SUGGESTIONS CHECKLIST
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">💡 Intelligent Cleaning Suggestions</div>', unsafe_allow_html=True)

selected_actions = []
if suggestions:
    grouped = defaultdict(list)
    for s in suggestions:
        grouped[s["column"]].append(s)

    for group_col, group_suggestions in grouped.items():
        label = "📁 Dataset" if group_col == "(dataset)" else f"📌 Column: **{group_col}**"
        with st.expander(label, expanded=True):
            for s in group_suggestions:
                key = f"chk_{s['column']}_{s['action']}"
                if st.checkbox(s["description"], value=s["enabled"], key=key):
                    selected_actions.append(s)
else:
    st.info("No cleaning suggestions — data may already be clean.")

st.divider()

col_btn, col_count = st.columns([2, 3])
with col_btn:
    apply_btn = st.button(
        "🚀 Apply & Analyse",
        type="primary",
        disabled=len(selected_actions) == 0 and not suggestions,
    )
with col_count:
    st.markdown(f"**{len(selected_actions)}** action(s) selected out of {len(suggestions)}")

if not apply_btn:
    st.stop()

# ─────────────────────────────────────────────
# APPLY CLEANING
# ─────────────────────────────────────────────

prog = st.progress(0, text="Cleaning in progress…")
with st.spinner("Applying cleaning actions..."):
    df_clean, cleaning_report = apply_cleaning(df_raw, selected_actions)
prog.progress(100, text="✅ Cleaning complete!")

# ── Cleaned Preview ───────────────────────────────────────────────────────────

st.markdown('<div class="section-header">✨ Cleaned Dataset</div>', unsafe_allow_html=True)
st.dataframe(df_clean.head(preview_rows), use_container_width=True)

# ── Cleaning Report ───────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📑 Cleaning Report</div>', unsafe_allow_html=True)
r1, r2, r3, r4, r5, r6 = st.columns(6)
r1.metric("Rows Before",          f"{cleaning_report['rows_before']:,}")
r2.metric("Rows After",           f"{cleaning_report['rows_after']:,}",
          delta=cleaning_report['rows_after'] - cleaning_report['rows_before'])
r3.metric("Duplicates Removed",   cleaning_report["duplicates_removed"])
r4.metric("Missing Fixed",        cleaning_report["missing_fixed"])
r5.metric("Columns Standardised", cleaning_report["columns_standardized"])
r6.metric("Outliers Handled",     cleaning_report["outliers_handled"])

with st.expander("📋 Full Action Log", expanded=False):
    log_df = pd.DataFrame(cleaning_report["actions_log"])
    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No actions logged.")

# ─────────────────────────────────────────────
# GENERATE INSIGHTS & AI SUMMARY
# ─────────────────────────────────────────────

with st.spinner("🤖 Generating insights and AI analysis..."):
    insights   = generate_insights(df_clean)
    ai_summary = generate_ai_summary(df_clean, insights)

# ═════════════════════════════════════════════
# SECTION: AUTOMATIC DATA INSIGHTS
# ═════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-header">🔍 Automatic Data Insights</div>', unsafe_allow_html=True)

s = insights["summary"]
if s:
    si1, si2, si3, si4 = st.columns(4)
    si1.metric("Rows",          f"{s['rows']:,}")
    si2.metric("Numeric Cols",  s["numeric_count"])
    si3.metric("Category Cols", s["categorical_count"])
    si4.metric("Completeness",  f"{s['completeness_pct']}%")

# ── Numeric Stats Table ───────────────────────────────────────────────────────

if insights["numeric"]:
    st.markdown("#### 📐 Numeric Column Statistics")
    num_data = []
    for n in insights["numeric"]:
        if n.get("empty"):
            continue
        num_data.append({
            "Column":       n["column"],
            "Count":        n["count"],
            "Mean":         n["mean"],
            "Median":       n["median"],
            "Std Dev":      n["std"],
            "Min":          n["min"],
            "Max":          n["max"],
            "Outliers":     n["outliers"],
            "Distribution": n["skew_label"],
        })
    if num_data:
        st.dataframe(pd.DataFrame(num_data), use_container_width=True, hide_index=True)

# ── Categorical Top Values ────────────────────────────────────────────────────

if insights["categorical"]:
    st.markdown("#### 🏷️ Categorical Column Insights")
    n_cat = len([c for c in insights["categorical"] if not c.get("empty")])
    if n_cat > 0:
        cat_display_cols = st.columns(min(3, n_cat))
        display_idx = 0
        for c in insights["categorical"]:
            if c.get("empty"):
                continue
            with cat_display_cols[display_idx % len(cat_display_cols)]:
                st.markdown(f"**{c['column']}** · {c['unique_count']} unique · {c['concentration']}")
                top10 = c.get("top_10", {})
                if top10:
                    cat_df = pd.DataFrame(
                        list(top10.items())[:max_bar_cats],
                        columns=["Value", "Count"]
                    )
                    st.dataframe(cat_df, use_container_width=True, hide_index=True, height=180)
            display_idx += 1

# ── Correlation Insights ──────────────────────────────────────────────────────

if insights["correlations"]:
    st.markdown("#### 🔗 Notable Correlations")
    for corr_str in insights["correlations"]:
        st.markdown(f'<div class="insight-card">• {corr_str}</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════
# SECTION: AI ANALYSIS SUMMARY
# ═════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-header">🤖 AI Analysis Summary</div>', unsafe_allow_html=True)

tabs = st.tabs(["📋 Overview", "📐 Numeric", "🏷️ Categorical",
                "🔗 Correlations", "⚠️ Anomalies", "💡 Recommendations"])

section_keys = ["overview", "numeric_obs", "category_obs",
                "correlation_obs", "anomalies", "recommendations"]

for tab, key in zip(tabs, section_keys):
    with tab:
        lines = ai_summary.get(key, [])
        if lines:
            for line in lines:
                st.markdown(f'<div class="ai-card">{line}</div>', unsafe_allow_html=True)
        else:
            st.info("No observations for this section.")

# ═════════════════════════════════════════════
# SECTION: DATA VISUALIZATION DASHBOARD
# ═════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-header">📈 Data Visualization Dashboard</div>', unsafe_allow_html=True)

numeric_cols     = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
ncols_per_row    = 3

# ── Histograms ────────────────────────────────────────────────────────────────

if numeric_cols:
    st.markdown("#### 📊 Numeric Distributions")
    for i in range(0, len(numeric_cols), ncols_per_row):
        row_cols = numeric_cols[i: i + ncols_per_row]
        fig_cols = st.columns(ncols_per_row)
        for j, col in enumerate(row_cols):
            with fig_cols[j]:
                try:
                    s = pd.to_numeric(df_clean[col], errors="coerce").dropna()
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.hist(s, bins=chart_bins,
                            color=PALETTE[j % len(PALETTE)],
                            edgecolor="#0E1117", alpha=0.9)
                    ax.set_title(col, fontsize=10, color="#ddd")
                    ax.set_xlabel("Value", fontsize=8)
                    ax.set_ylabel("Frequency", fontsize=8)
                    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                    ax.grid(True, axis="y")
                    med = float(s.median())
                    ax.axvline(med, color="#FFD700", linestyle="--",
                               linewidth=1.2, label=f"Median: {med:,.2f}")
                    ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not plot '{col}': {e}")

# ── Bar Charts ────────────────────────────────────────────────────────────────

if categorical_cols:
    st.markdown("#### 🏷️ Top Category Distributions")
    for i in range(0, len(categorical_cols), ncols_per_row):
        row_cols = categorical_cols[i: i + ncols_per_row]
        fig_cols = st.columns(ncols_per_row)
        for j, col in enumerate(row_cols):
            with fig_cols[j]:
                try:
                    vc = (df_clean[col].dropna()
                                       .astype(str).str.strip()
                                       .value_counts()
                                       .head(max_bar_cats))
                    if vc.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(4, 3))
                    colors = [PALETTE[k % len(PALETTE)] for k in range(len(vc))]
                    bars = ax.barh(vc.index[::-1], vc.values[::-1],
                                   color=colors[::-1],
                                   edgecolor="#0E1117", alpha=0.9)
                    ax.set_title(col, fontsize=10, color="#ddd")
                    ax.set_xlabel("Count", fontsize=8)
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                    ax.grid(True, axis="x")
                    for bar, val in zip(bars, vc.values[::-1]):
                        ax.text(val + max(vc.values) * 0.01,
                                bar.get_y() + bar.get_height() / 2,
                                str(val), va="center", fontsize=7, color="#ddd")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not plot '{col}': {e}")

# ── Correlation Heatmap ───────────────────────────────────────────────────────

if len(numeric_cols) >= 2:
    st.markdown("#### 🌡️ Correlation Heatmap")
    try:
        corr_df = df_clean[numeric_cols].corr()
        n       = len(numeric_cols)
        fig_w   = max(5, n * 1.0)
        fig_h   = max(4, n * 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im      = ax.imshow(corr_df.values, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_df.columns, fontsize=8)
        for row in range(n):
            for col_idx in range(n):
                val = corr_df.values[row, col_idx]
                tc  = "black" if abs(val) < 0.6 else "white"
                ax.text(col_idx, row, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=max(6, 10 - n),
                        color=tc, fontweight="bold")
        ax.set_title("Correlation Matrix", fontsize=12, color="#ddd", pad=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render heatmap: {e}")

# ═════════════════════════════════════════════
# SECTION: NATURAL LANGUAGE QUERY ENGINE
# ═════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-header">💬 Ask Questions About Your Dataset</div>',
            unsafe_allow_html=True)
st.markdown("Type a plain-English business question and get an instant answer powered by your data.")

# ── Example questions grid ────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "Which city has the highest sales?",
    "What is the total revenue?",
    "Which product sells the most?",
    "What is the average price?",
    "Show sales by city",
    "Who are the top 3 customers?",
    "What is the total quantity sold?",
    "Most popular product?",
    "Which region has the lowest sales?",
    "Revenue per product",
    "Show sales trend over time",
    "Forecast next month revenue",
    "What is the correlation between numeric columns?",
    "Which region is declining?",
    "Show revenue share by category",
    "How many unique customers are there?",
    "Show sales distribution",
    "Top 5 categories by revenue",
    "What is the average order value?",
    "Top 5% customers by revenue",
]

with st.expander("💡 Example Questions — click to copy", expanded=True):
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            st.markdown(f"• _{example}_")

# ── Query Input ───────────────────────────────────────────────────────────────

st.markdown("")
question = st.text_input(
    "🔍 Your Question",
    placeholder="e.g. Which city has the highest sales?",
    key="nl_query_input",
)

col_ask, col_clear = st.columns([2, 1])
with col_ask:
    ask_btn = st.button("📊 Get Answer", type="primary", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("Analysing your question..."):
        answer = answer_query(df_clean, question)
    st.success("**Answer:**")
    st.markdown(answer)
    # Save to history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    st.session_state.query_history.insert(0, {"q": question, "a": answer})
    st.session_state.query_history = st.session_state.query_history[:10]

elif ask_btn and not question.strip():
    st.warning("Please type a question first.")

# ── Query History (session state) ────────────────────────────────────────────

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if st.session_state.query_history:
    with st.expander("🕓 Recent Questions", expanded=False):
        for item in st.session_state.query_history:
            st.markdown(f"**Q:** {item['q']}")
            st.markdown(f"**A:** {item['a']}")
            st.divider()

# ═════════════════════════════════════════════
# DOWNLOAD
# ═════════════════════════════════════════════

st.divider()
st.markdown('<div class="section-header">📥 Download Cleaned Data</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇️ Download as CSV",
        data=df_to_csv_bytes(df_clean),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "⬇️ Download as Excel",
        data=df_to_excel_bytes(df_clean),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
