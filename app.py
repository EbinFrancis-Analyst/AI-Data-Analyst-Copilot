"""
app.py — AI Data Analyst Copilot  v5
=====================================================
New in v5:
  FEATURE  Professional social profile buttons in page header
  BUG-1    Ambiguous column → disambiguation widget in NL chart generator
  BUG-2    get_dataset_schema() guards all column references
  BUG-3    clean_numeric_columns() applied on file load (before any analytics)
"""

from __future__ import annotations

import gc
from collections import defaultdict

import pandas as pd
import streamlit as st

# ── Local modules ──────────────────────────────────────────────────────────────
from data_loader      import load_file, get_preview, memory_report
from analytics_engine import (
    compute_kpis, column_profile, quality_score, distribution_metrics,
    get_dataset_schema, clean_numeric_columns,
    resolve_column_ambiguity, ambiguity_message,
)
from dashboard_generator import generate_dashboard, apply_filters
from query_engine_v3     import generate_chart, EXAMPLE_QUERIES
from report_generator    import generate_pdf_report, is_available as pdf_available

from analyzer import profile_dataset, detect_issues, generate_suggestions
from cleaner  import apply_cleaning
from utils    import df_to_csv_bytes, df_to_excel_bytes


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Analyst Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── App chrome ────────────────────────────────────────────────────────── */
.hero-title {
    font-size:2.2rem; font-weight:800; line-height:1.15;
    background:linear-gradient(90deg,#4F8BF9,#A259FF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.section-hdr {
    font-size:1.05rem; font-weight:700; color:#A259FF;
    margin-top:1.2rem; border-bottom:2px solid #A259FF22; padding-bottom:.3rem;
}
.badge-high   { color:#FF4B4B; font-weight:700; }
.badge-medium { color:#FFA500; font-weight:600; }
.badge-low    { color:#4F8BF9; }
.schema-box {
    background:#0d1117; border:1px solid #30363d; border-radius:8px;
    padding:.75rem 1rem; font-family:monospace; font-size:.78rem;
    color:#c9d1d9; white-space:pre-wrap; line-height:1.6;
}
div[data-testid="stDownloadButton"] button { width:100%; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL HEADER  — rendered via components.v1.html so Streamlit's
# markdown sanitiser NEVER touches it (fixes raw-HTML-visible-on-page bug).
# ─────────────────────────────────────────────────────────────────────────────

import streamlit.components.v1 as _components

_components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Inter", "Segoe UI", sans-serif;
    background: transparent;
    padding: 4px 0 6px 0;
  }
  .header-wrap {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }
  .title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4F8BF9, #A259FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
  }
  .subtitle {
    font-size: 0.82rem;
    color: #888;
    margin-top: 4px;
  }
  .btns { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 16px;
    border-radius: 22px;
    font-size: 0.8rem;
    font-weight: 600;
    text-decoration: none;
    color: #fff;
    border: 1.5px solid transparent;
    cursor: pointer;
    transition: transform 0.18s, box-shadow 0.18s, filter 0.18s;
    white-space: nowrap;
  }
  .btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    filter: brightness(1.15);
    text-decoration: none;
  }
  .btn-li  { background: #0A66C2; border-color: #0A66C2; }
  .btn-gh  { background: #24292E; border-color: #555; }
  .btn-em  { background: #EA4335; border-color: #EA4335; }
  svg { flex-shrink: 0; }
</style>
</head>
<body>
<div class="header-wrap">

  <div>
    <div class="title">🤖 AI Data Analyst Copilot</div>
    <div class="subtitle">
      Large-dataset ready &nbsp;·&nbsp; Automated cleaning &nbsp;·&nbsp;
      Executive dashboard &nbsp;·&nbsp; NL charts &nbsp;·&nbsp; PDF report
    </div>
  </div>

  <div class="btns">

    <a class="btn btn-li"
       href="https://www.linkedin.com/in/ebin-francis-30b7b4273/"
       target="_blank" rel="noopener noreferrer">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="white">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853
        0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85
        3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0
        1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225
        0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24
        23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
      </svg>
      LinkedIn
    </a>

    <a class="btn btn-gh"
       href="https://github.com/EbinFrancis-Analyst"
       target="_blank" rel="noopener noreferrer">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="white">
        <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577
        0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633
        17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809
        1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38
        1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405
        1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12
        3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81
        2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592
        24 12.297c0-6.627-5.373-12-12-12"/>
      </svg>
      GitHub
    </a>

    <a class="btn btn-em" href="mailto:ebinfrancis82@gmail.com">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="white">
        <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9
        2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>
      </svg>
      Email
    </a>

  </div>
</div>
</body>
</html>
""", height=80)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  (all persistent state — survives every rerun / button click)
# ─────────────────────────────────────────────────────────────────────────────

_SS_DEFAULTS: dict = {
    "df_raw":          None,
    "df_clean":        None,
    "df_filtered":     None,
    "kpis":            None,
    "cleaning_report": None,
    "pdf_bytes":       None,
    "file_name":       None,
    "analysis_done":   False,
    "profile":         None,
    "issues":          None,
    "suggestions":     None,
    "nl_col_override": None,   # BUG-1: user-chosen column when ambiguous
}

for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📂 Upload Dataset")
    uploaded = st.file_uploader(
        "CSV · XLSX · XLS — up to 200 MB",
        type=["csv", "xlsx", "xls"],
        help="Chunk-based loading for large files.",
    )
    st.divider()
    st.markdown("### ⚙️ Settings")
    preview_n = st.slider("Preview rows", 10, 200, 100, step=10)


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────────────────────────────────────

if uploaded is None and st.session_state["df_raw"] is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **🚀 Performance**
        - Chunk CSV reading (50k rows/chunk)
        - Memory-optimised dtypes
        - `@st.cache_data` on all analytics
        - Handles up to 200 MB / 500k rows
        """)
    with c2:
        st.markdown("""
        **🧹 Data Cleaning**
        - 14+ issue detectors
        - Intelligent suggestions
        - One-click cleaning
        - Cleaning audit log
        """)
    with c3:
        st.markdown("""
        **📊 Analytics**
        - 8-tab executive dashboard
        - NL → Plotly chart generator
        - AI business summary
        - PDF report download
        """)
    st.info("👆 Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOAD  (re-reads only when a new file arrives)
# ─────────────────────────────────────────────────────────────────────────────

if uploaded is not None:
    new_file = uploaded.name != st.session_state.get("file_name")
    if new_file:
        for k, v in _SS_DEFAULTS.items():
            st.session_state[k] = v

        with st.spinner(f"⏳ Loading {uploaded.name} …"):
            file_bytes = uploaded.read()
            df_raw, err = load_file(file_bytes, uploaded.name)

        if err:
            st.error(f"❌ {err}")
            st.stop()

        # ── BUG-3: fix dirty numeric-as-object columns immediately on load ──
        with st.spinner("🔧 Fixing numeric column types …"):
            df_raw = clean_numeric_columns(df_raw)

        st.session_state["df_raw"]    = df_raw
        st.session_state["file_name"] = uploaded.name

        with st.spinner("🔬 Profiling dataset …"):
            profile     = profile_dataset(df_raw)
            issues      = detect_issues(df_raw, profile)
            suggestions = generate_suggestions(df_raw, profile, issues)

        st.session_state["profile"]     = profile
        st.session_state["issues"]      = issues
        st.session_state["suggestions"] = suggestions
        gc.collect()


df_raw = st.session_state["df_raw"]
if df_raw is None:
    st.warning("Please upload a dataset using the sidebar.")
    st.stop()

rows, cols_n = df_raw.shape
mem_mb = round(df_raw.memory_usage(deep=True).sum() / 1024**2, 1)
st.success(
    f"✅ **{st.session_state['file_name']}** — "
    f"{rows:,} rows × {cols_n} columns — {mem_mb} MB in RAM"
)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET SCHEMA  (BUG-2 — shows validated column list to user)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("🗂️ Dataset Schema (validated column list)", expanded=False):
    schema_str = get_dataset_schema(df_raw)
    st.markdown(
        f'<div class="schema-box">{schema_str}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This schema is used internally to prevent column hallucination. "
        "All chart and analysis operations only reference columns listed here."
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREVIEW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📋 Dataset Preview</div>', unsafe_allow_html=True)
st.caption(f"Showing first {min(preview_n, rows):,} of {rows:,} rows")
st.dataframe(get_preview(df_raw, preview_n), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📊 Data Quality Overview</div>', unsafe_allow_html=True)

score, score_label, bar_colour = quality_score(df_raw)
prof_df = column_profile(df_raw)
dupes   = int(df_raw.duplicated().sum())
missing = int(df_raw.isna().sum().sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",           f"{rows:,}")
c2.metric("Columns",        cols_n)
c3.metric("Duplicate Rows", dupes)
c4.metric("Missing Values", missing)
c5.metric("Quality Score",  f"{score} / 100")

st.markdown(
    f'<div style="background:#2E2E3E;border-radius:8px;height:16px;margin:.4rem 0">'
    f'<div style="background:{bar_colour};width:{score}%;height:16px;border-radius:8px;'
    f'display:flex;align-items:center;justify-content:center;'
    f'font-size:.65rem;color:#fff;font-weight:700;">{score_label}</div>'
    f"</div>",
    unsafe_allow_html=True,
)

with st.expander("🔬 Column Profile", expanded=False):
    st.dataframe(prof_df, use_container_width=True, hide_index=True)

with st.expander("🧠 Memory Usage by Column", expanded=False):
    mem = memory_report(df_raw)
    total_str = mem.pop("__total__")
    st.caption(f"Total in-memory: **{total_str}**")
    st.dataframe(
        pd.DataFrame(list(mem.items()), columns=["Column", "Memory"]),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE DETECTION & CLEANING SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">🚨 Issue Detection & Cleaning</div>', unsafe_allow_html=True)

issues      = st.session_state["issues"]      or []
suggestions = st.session_state["suggestions"] or []

if issues:
    high_n = sum(1 for i in issues if i["severity"] == "High")
    med_n  = sum(1 for i in issues if i["severity"] == "Medium")
    low_n  = sum(1 for i in issues if i["severity"] == "Low")
    ih, im, il = st.columns(3)
    ih.markdown(f'<span class="badge-high">🔴 High: {high_n}</span>',    unsafe_allow_html=True)
    im.markdown(f'<span class="badge-medium">🟠 Medium: {med_n}</span>', unsafe_allow_html=True)
    il.markdown(f'<span class="badge-low">🔵 Low: {low_n}</span>',       unsafe_allow_html=True)
    with st.expander("📋 All Detected Issues", expanded=False):
        st.dataframe(pd.DataFrame([{
            "Column":      i["column"],
            "Issue":       i["issue_type"],
            "Severity":    i["severity"],
            "Description": i["description"],
            "Affected":    i["affected_count"],
        } for i in issues]), use_container_width=True, hide_index=True)
else:
    st.success("✨ No issues detected — dataset looks clean!")

st.markdown("#### 💡 Cleaning Suggestions")
selected_actions = []
if suggestions:
    grouped = defaultdict(list)
    for s in suggestions:
        grouped[s["column"]].append(s)
    for grp_col, grp_sug in grouped.items():
        label = "📁 Dataset" if grp_col == "(dataset)" else f"📌 **{grp_col}**"
        with st.expander(label, expanded=False):
            for s in grp_sug:
                key = f"chk_{s['column']}_{s['action']}"
                if st.checkbox(s["description"], value=s["enabled"], key=key):
                    selected_actions.append(s)
else:
    st.info("No cleaning suggestions available.")

btn_c, cnt_c = st.columns([2, 3])
with btn_c:
    apply_btn = st.button(
        "🚀 Apply & Analyse",
        type="primary",
        disabled=(len(selected_actions) == 0 and not st.session_state["analysis_done"]),
    )
with cnt_c:
    st.markdown(f"**{len(selected_actions)}** of {len(suggestions)} suggestions selected")


# ─────────────────────────────────────────────────────────────────────────────
# APPLY CLEANING
# ─────────────────────────────────────────────────────────────────────────────

if apply_btn:
    prog = st.progress(0, text="Applying cleaning actions …")
    df_clean, report = apply_cleaning(df_raw, selected_actions)

    # BUG-3: re-apply numeric fixer after cleaning (new string values may appear)
    df_clean = clean_numeric_columns(df_clean)
    gc.collect()
    prog.progress(100, text="✅ Done")

    with st.spinner("🔬 Computing analytics …"):
        kpis = compute_kpis(df_clean)

    st.session_state["df_clean"]        = df_clean
    st.session_state["cleaning_report"] = report
    st.session_state["kpis"]            = kpis
    st.session_state["pdf_bytes"]       = None
    st.session_state["analysis_done"]   = True
    st.session_state["nl_col_override"] = None


# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state["analysis_done"]:
    st.info("☝️ Select cleaning suggestions above and click **Apply & Analyse** to continue.")
    st.stop()

df_clean        = st.session_state["df_clean"]
cleaning_report = st.session_state["cleaning_report"]
kpis            = st.session_state["kpis"]


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING REPORT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📑 Cleaning Report</div>', unsafe_allow_html=True)
r1, r2, r3, r4, r5, r6 = st.columns(6)
r1.metric("Rows Before",        f"{cleaning_report['rows_before']:,}")
r2.metric("Rows After",         f"{cleaning_report['rows_after']:,}",
          delta=cleaning_report["rows_after"] - cleaning_report["rows_before"])
r3.metric("Duplicates Removed", cleaning_report["duplicates_removed"])
r4.metric("Missing Fixed",      cleaning_report["missing_fixed"])
r5.metric("Cols Standardised",  cleaning_report["columns_standardized"])
r6.metric("Outliers Handled",   cleaning_report["outliers_handled"])

with st.expander("📋 Full Action Log", expanded=False):
    log_df = pd.DataFrame(cleaning_report["actions_log"])
    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No actions logged.")

st.markdown('<div class="section-hdr">✨ Cleaned Dataset Preview</div>', unsafe_allow_html=True)
st.caption(f"Showing first {min(preview_n, len(df_clean)):,} of {len(df_clean):,} rows")
st.dataframe(get_preview(df_clean, preview_n), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="background:linear-gradient(90deg,#4F8BF9,#A259FF);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:1.8rem;font-weight:800;margin-bottom:.2rem;">
    📊 Executive Analytics Dashboard
</div>
<div style="color:#888;font-size:.85rem;margin-bottom:.8rem;">
    Power BI-style interactive dashboard · Sidebar filters update all charts
</div>
""", unsafe_allow_html=True)

df_filtered = apply_filters(df_clean, kpis)
st.session_state["df_filtered"] = df_filtered

if df_filtered.empty:
    st.warning("⚠️ Filters produce an empty dataset — adjust sidebar filters.")
else:
    kpis_filtered = compute_kpis(df_filtered)
    generate_dashboard(df_filtered, kpis_filtered)


# ─────────────────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE CHART GENERATOR  (BUG-1 disambiguation integrated)
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-hdr">💬 Natural Language Chart Generator</div>',
            unsafe_allow_html=True)
st.markdown("Describe a chart in plain English — ambiguous columns are flagged for you.")

with st.expander("💡 Example prompts", expanded=False):
    ex_c = st.columns(2)
    for i, ex in enumerate(EXAMPLE_QUERIES):
        with ex_c[i % 2]:
            st.markdown(f"• *{ex}*")

# ── BUG-2: show schema so user knows exact column names ──────────────────────
with st.expander("🗂️ Available columns in your dataset", expanded=False):
    st.markdown(
        f'<div class="schema-box">{get_dataset_schema(df_clean)}</div>',
        unsafe_allow_html=True,
    )

nl_q = st.text_input(
    "🔍 Describe your chart",
    placeholder="e.g. top 10 products by revenue",
    key="nl_chart_input",
)

# ── BUG-1: column override selector (shown only when ambiguity was detected) ──
ambiguous_cols = st.session_state.get("nl_ambiguous_cols", [])
if ambiguous_cols:
    st.warning(
        f"⚠️ Multiple columns match your request. Please choose one:",
        icon="⚠️",
    )
    chosen = st.selectbox(
        "Select the column you meant:",
        options=ambiguous_cols,
        key="nl_col_picker",
    )
    if st.button("✅ Use this column", key="nl_col_confirm"):
        st.session_state["nl_col_override"] = chosen
        st.session_state["nl_ambiguous_cols"] = []
        st.rerun()

col1_btn, col2_btn = st.columns([1, 5])
with col1_btn:
    gen_btn = st.button("📈 Generate Chart", type="primary", key="nl_btn")

if gen_btn and nl_q.strip():
    plot_df = st.session_state.get("df_filtered") or df_clean

    # Inject user-chosen column into question if override is set (BUG-1)
    effective_q = nl_q
    override = st.session_state.get("nl_col_override")
    if override:
        effective_q = f"{nl_q} {override}"
        st.session_state["nl_col_override"] = None

    with st.spinner("Generating chart …"):
        fig, label = generate_chart(plot_df, effective_q)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Intent detected: `{label}`")
        st.session_state["nl_ambiguous_cols"] = []
    else:
        # BUG-1: detect whether the error is an ambiguity message
        if "Multiple columns match" in label:
            # Parse the bullet list from the ambiguity message and offer selector
            lines = [
                l.strip().lstrip("•").strip()
                for l in label.split("\n")
                if l.strip().startswith("•")
            ]
            st.session_state["nl_ambiguous_cols"] = lines if lines else []
            st.warning(label)
        else:
            st.session_state["nl_ambiguous_cols"] = []
            st.warning(label)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT — CSV / Excel / PDF
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-hdr">📥 Export</div>', unsafe_allow_html=True)

dl1, dl2, dl3 = st.columns(3)

with dl1:
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=df_to_csv_bytes(df_clean),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_csv",
    )

with dl2:
    st.download_button(
        label="⬇️ Download Cleaned Excel",
        data=df_to_excel_bytes(df_clean),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="btn_xlsx",
    )

with dl3:
    if pdf_available():
        if st.session_state["pdf_bytes"] is None:
            if st.button(
                "📄 Generate AI PDF Report",
                use_container_width=True,
                type="primary",
                key="btn_gen_pdf",
            ):
                with st.spinner("Building PDF report …"):
                    try:
                        st.session_state["pdf_bytes"] = generate_pdf_report(
                            df_clean,
                            kpis,
                            dataset_name=st.session_state.get("file_name", "Dataset"),
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

        # Download button appears in the same script pass (no st.rerun needed)
        if st.session_state["pdf_bytes"] is not None:
            st.download_button(
                label="⬇️ Download AI PDF Report",
                data=st.session_state["pdf_bytes"],
                file_name="ai_business_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="btn_pdf_dl",
            )
            if st.button("🔄 Regenerate PDF", use_container_width=True,
                         key="btn_regen_pdf"):
                st.session_state["pdf_bytes"] = None
                st.rerun()
    else:
        st.info("Install reportlab to enable PDF reports:\n`pip install reportlab`")
