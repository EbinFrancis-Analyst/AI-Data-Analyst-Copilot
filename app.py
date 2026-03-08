"""
app.py
AI Data Analyst Copilot — v3 (Performance-Optimised)
Handles datasets up to 200MB / 500k rows smoothly via:
  - chunk-based loading       (data_loader.py)
  - cached heavy analytics    (analytics_engine.py)
  - lazy tab-based dashboard  (dashboard_generator.py)
  - NL chart generator        (query_engine_v3.py)
  - PDF report generator      (report_generator.py)
"""

from __future__ import annotations

import gc
from collections import defaultdict

import pandas as pd
import streamlit as st

# ── Local modules ──────────────────────────────────────────────────────────
from data_loader        import load_file, get_preview, get_sample, memory_report
from analytics_engine   import (
    compute_kpis, column_profile, quality_score,
    detect_column_types, distribution_metrics,
)
from dashboard_generator import generate_dashboard, apply_filters
from query_engine_v3    import generate_chart, EXAMPLE_QUERIES
from report_generator   import generate_pdf_report, is_available as pdf_available

# ── Old cleaning modules (kept for cleaning pipeline) ─────────────────────
from analyzer import profile_dataset, detect_issues, generate_suggestions
from cleaner  import apply_cleaning
from utils    import df_to_csv_bytes, df_to_excel_bytes

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Analyst Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .hero-title {
      font-size: 2.2rem; font-weight: 800;
      background: linear-gradient(90deg, #4F8BF9, #A259FF);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .section-hdr {
      font-size: 1.05rem; font-weight: 700; color: #A259FF;
      margin-top: 1.2rem; border-bottom: 2px solid #A259FF22;
      padding-bottom: 0.3rem;
  }
  .badge-high   { color: #FF4B4B; font-weight: 700; }
  .badge-medium { color: #FFA500; font-weight: 600; }
  .badge-low    { color: #4F8BF9; }
  .ai-card {
      background: #0f2027; border-left: 4px solid #A259FF;
      border-radius: 8px; padding: 0.7rem 1rem;
      margin-bottom: 0.45rem; font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — UPLOAD & SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📂 Upload Dataset")
    uploaded = st.file_uploader(
        "CSV, XLSX, XLS — up to 200 MB",
        type=["csv", "xlsx", "xls"],
        help="Chunk-based loading keeps memory efficient for large files.",
    )
    st.divider()
    st.markdown("### ⚙️ Settings")
    preview_n = st.slider("Preview rows", 10, 200, 100)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🤖 AI Data Analyst Copilot</div>', unsafe_allow_html=True)
st.markdown("Large-dataset ready · Automated cleaning · Executive dashboard · NL charts · PDF report")
st.divider()

if uploaded is None:
    st.info("👆 Upload a CSV or Excel file from the sidebar to begin.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🚀 Performance**
        - Chunk-based CSV reading
        - Memory-optimised dtypes
        - Cached analytics engine
        - Handles up to 200 MB
        """)
    with col2:
        st.markdown("""
        **🧹 Data Cleaning**
        - 14+ issue detectors
        - Intelligent suggestions
        - One-click apply
        - Cleaning audit log
        """)
    with col3:
        st.markdown("""
        **📊 Analytics**
        - Executive dashboard
        - NL chart generator
        - AI analysis summary
        - PDF report export
        """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FILE
# ─────────────────────────────────────────────────────────────────────────────

file_bytes = uploaded.read()
with st.spinner(f"⏳ Loading {uploaded.name} …"):
    df_raw, err = load_file(file_bytes, uploaded.name)

if err:
    st.error(f"❌ {err}")
    st.stop()

rows, cols = df_raw.shape
mem_mb = round(df_raw.memory_usage(deep=True).sum() / 1024**2, 1)
st.success(f"✅ **{uploaded.name}** loaded — {rows:,} rows × {cols} columns — {mem_mb} MB in RAM")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREVIEW  (first N rows only — never render full frame)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📋 Dataset Preview</div>', unsafe_allow_html=True)
st.caption(f"Showing first {min(preview_n, rows):,} of {rows:,} rows")
st.dataframe(get_preview(df_raw, preview_n), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📊 Data Quality Overview</div>', unsafe_allow_html=True)

with st.spinner("Profiling dataset …"):
    score, score_label, score_colour = quality_score(df_raw)
    prof_df  = column_profile(df_raw)
    dupes    = int(df_raw.duplicated().sum())
    missing  = int(df_raw.isna().sum().sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",            f"{rows:,}")
c2.metric("Columns",         cols)
c3.metric("Duplicate Rows",  dupes)
c4.metric("Missing Values",  missing)
c5.metric("Quality Score",   f"{score} / 100")

bar_c = {"green":"#4CAF50","orange":"#FFA500","darkorange":"#FF5722","red":"#FF1744"}.get(score_colour,"#888")
st.markdown(f"""
<div style="background:#2E2E3E;border-radius:8px;height:16px;margin:.4rem 0">
  <div style="background:{bar_c};width:{score}%;height:16px;border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-size:.65rem;color:#fff;font-weight:700;">{score_label}</div>
</div>""", unsafe_allow_html=True)

with st.expander("🔬 Column Profile", expanded=False):
    st.dataframe(prof_df, use_container_width=True, hide_index=True)

# Memory breakdown
with st.expander("🧠 Memory Usage by Column", expanded=False):
    mem = memory_report(df_raw)
    total_mb = mem.pop("__total__")
    st.caption(f"Total in-memory size: **{total_mb}**")
    mem_df = pd.DataFrame(list(mem.items()), columns=["Column", "Memory"])
    st.dataframe(mem_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# ISSUE DETECTION + CLEANING
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">🚨 Issue Detection & Cleaning</div>', unsafe_allow_html=True)

with st.spinner("Detecting issues …"):
    profile     = profile_dataset(df_raw)
    issues      = detect_issues(df_raw, profile)
    suggestions = generate_suggestions(df_raw, profile, issues)

# Issue summary badges
if issues:
    high_n = sum(1 for i in issues if i["severity"] == "High")
    med_n  = sum(1 for i in issues if i["severity"] == "Medium")
    low_n  = sum(1 for i in issues if i["severity"] == "Low")
    ih, im, il = st.columns(3)
    ih.markdown(f'<span class="badge-high">🔴 High severity: {high_n}</span>',   unsafe_allow_html=True)
    im.markdown(f'<span class="badge-medium">🟠 Medium: {med_n}</span>',          unsafe_allow_html=True)
    il.markdown(f'<span class="badge-low">🔵 Low: {low_n}</span>',                unsafe_allow_html=True)

    with st.expander("📋 All Detected Issues", expanded=False):
        issues_df = pd.DataFrame([{
            "Column": i["column"], "Issue": i["issue_type"],
            "Severity": i["severity"], "Description": i["description"],
            "Affected": i["affected_count"],
        } for i in issues])
        st.dataframe(issues_df, use_container_width=True, hide_index=True)
else:
    st.success("✨ No issues detected — dataset looks clean!")

# Cleaning suggestions checklist
st.markdown("#### 💡 Cleaning Suggestions")
selected_actions = []
if suggestions:
    grouped = defaultdict(list)
    for s in suggestions:
        grouped[s["column"]].append(s)

    for group_col, group_sug in grouped.items():
        label = "📁 Dataset" if group_col == "(dataset)" else f"📌 **{group_col}**"
        with st.expander(label, expanded=False):
            for s in group_sug:
                key = f"chk_{s['column']}_{s['action']}"
                if st.checkbox(s["description"], value=s["enabled"], key=key):
                    selected_actions.append(s)
else:
    st.info("No cleaning suggestions.")

btn_col, cnt_col = st.columns([2, 3])
with btn_col:
    apply_btn = st.button("🚀 Apply & Analyse",
                          type="primary",
                          disabled=len(selected_actions) == 0)
with cnt_col:
    st.markdown(f"**{len(selected_actions)}** of {len(suggestions)} suggestions selected")

if not apply_btn:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RUN CLEANING
# ─────────────────────────────────────────────────────────────────────────────

prog = st.progress(0, text="Cleaning …")
with st.spinner("Applying cleaning actions …"):
    df_clean, cleaning_report = apply_cleaning(df_raw, selected_actions)
gc.collect()
prog.progress(100, text="✅ Cleaning done")

# Cleaning report metrics
st.markdown('<div class="section-hdr">📑 Cleaning Report</div>', unsafe_allow_html=True)
r1,r2,r3,r4,r5,r6 = st.columns(6)
r1.metric("Rows Before",      f"{cleaning_report['rows_before']:,}")
r2.metric("Rows After",       f"{cleaning_report['rows_after']:,}",
          delta=cleaning_report['rows_after']-cleaning_report['rows_before'])
r3.metric("Duplicates Removed",   cleaning_report["duplicates_removed"])
r4.metric("Missing Fixed",        cleaning_report["missing_fixed"])
r5.metric("Cols Standardised",    cleaning_report["columns_standardized"])
r6.metric("Outliers Handled",     cleaning_report["outliers_handled"])

with st.expander("📋 Action Log", expanded=False):
    log_df = pd.DataFrame(cleaning_report["actions_log"])
    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No actions logged.")

# Cleaned preview
st.markdown('<div class="section-hdr">✨ Cleaned Dataset Preview</div>', unsafe_allow_html=True)
st.caption(f"Showing first {min(preview_n, len(df_clean)):,} of {len(df_clean):,} rows")
st.dataframe(get_preview(df_clean, preview_n), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE ANALYTICS (cached — runs only once per unique df)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔬 Computing analytics …"):
    kpis = compute_kpis(df_clean)
    dist = distribution_metrics(df_clean)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="background:linear-gradient(90deg,#4F8BF9,#A259FF);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:1.8rem;font-weight:800;margin-bottom:.2rem;">
    📊 Executive Analytics Dashboard
</div>
<div style="color:#888;font-size:0.85rem;margin-bottom:0.8rem;">
    Interactive Power BI–style dashboard · Sidebar filters update all charts
</div>
""", unsafe_allow_html=True)

df_filtered = apply_filters(df_clean, kpis)

if df_filtered.empty:
    st.warning("⚠️ Current filters return an empty dataset — adjust sidebar filters.")
else:
    with st.spinner("🖼️ Rendering dashboard …"):
        generate_dashboard(df_filtered, compute_kpis(df_filtered))

# ─────────────────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE CHART GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-hdr">💬 Natural Language Chart Generator</div>',
            unsafe_allow_html=True)
st.markdown("Describe a chart in plain English and it appears instantly.")

with st.expander("💡 Example prompts", expanded=True):
    ex_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLE_QUERIES):
        with ex_cols[i % 2]:
            st.markdown(f"• *{ex}*")

nl_q = st.text_input("🔍 Describe your chart",
                     placeholder="e.g. top 10 products by revenue",
                     key="nl_chart_input")
if st.button("📈 Generate Chart", type="primary") and nl_q.strip():
    with st.spinner("Generating …"):
        fig, label = generate_chart(df_filtered, nl_q)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Chart type detected: `{label}`")
    else:
        st.warning(label)

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD & PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-hdr">📥 Export</div>', unsafe_allow_html=True)

dl1, dl2, dl3 = st.columns(3)

with dl1:
    st.download_button(
        "⬇️ Download CSV",
        data=df_to_csv_bytes(df_clean),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dl2:
    st.download_button(
        "⬇️ Download Excel",
        data=df_to_excel_bytes(df_clean),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with dl3:
    if pdf_available():
        if st.button("📄 Generate AI PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Building PDF report …"):
                try:
                    pdf_bytes = generate_pdf_report(
                        df_clean, kpis,
                        dataset_name=uploaded.name,
                    )
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name="ai_business_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("✅ PDF report ready!")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
    else:
        st.info("Install `reportlab` to enable PDF reports:\n`pip install reportlab`")
