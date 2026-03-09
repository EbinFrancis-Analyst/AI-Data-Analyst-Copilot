"""
app.py — AI Data Analyst Copilot  v4 (Production)
======================================================
Bug fixes in this version:
  BUG-1  Session state: df_clean, kpis, pdf_bytes survive every rerun
  BUG-2  Download buttons: read from session_state, never reset UI
  BUG-3  PDF generation: generated once, stored in session_state
  BUG-4  Large dataset lag: cached analytics, sampled previews, lazy tabs
  BUG-5  Plotly trendline OLS removed (no statsmodels dependency)
"""

from __future__ import annotations

import gc
from collections import defaultdict

import pandas as pd
import streamlit as st

# ── Local modules ──────────────────────────────────────────────────────────────
from data_loader         import load_file, get_preview, memory_report
from analytics_engine    import (
    compute_kpis, column_profile, quality_score, distribution_metrics,
)
from dashboard_generator import generate_dashboard, apply_filters
from query_engine_v3     import generate_chart, EXAMPLE_QUERIES
from report_generator    import generate_pdf_report, is_available as pdf_available

# ── Cleaning pipeline (unchanged from previous version) ───────────────────────
from analyzer import profile_dataset, detect_issues, generate_suggestions
from cleaner  import apply_cleaning
from utils    import df_to_csv_bytes, df_to_excel_bytes

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
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
      font-size:2.2rem; font-weight:800;
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
  div[data-testid="stDownloadButton"] button {
      width:100%; border-radius:8px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# All persistent state is stored here so reruns (button clicks, downloads)
# never lose the cleaned dataframe or computed results.
# ─────────────────────────────────────────────────────────────────────────────

_SS_DEFAULTS: dict = {
    "df_raw":           None,   # raw uploaded dataframe
    "df_clean":         None,   # cleaned dataframe (survives every rerun)
    "df_filtered":      None,   # dashboard-filtered view
    "kpis":             None,   # computed KPIs dict
    "cleaning_report":  None,   # cleaning action summary
    "pdf_bytes":        None,   # generated PDF bytes
    "file_name":        None,   # uploaded filename
    "analysis_done":    False,  # True once Apply & Analyse has run
    "profile":          None,   # dataset profile
    "issues":           None,   # detected issues list
    "suggestions":      None,   # cleaning suggestions list
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
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🤖 AI Data Analyst Copilot</div>', unsafe_allow_html=True)
st.markdown("Large-dataset ready · Automated cleaning · Executive dashboard · NL charts · PDF report")
st.divider()

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
# FILE LOAD — only re-reads when a new file is uploaded
# ─────────────────────────────────────────────────────────────────────────────

if uploaded is not None:
    # Detect whether a new file has been uploaded
    new_file = uploaded.name != st.session_state.get("file_name")
    if new_file:
        # Reset all derived state when a different file is uploaded
        for k, v in _SS_DEFAULTS.items():
            st.session_state[k] = v

        with st.spinner(f"⏳ Loading {uploaded.name} …"):
            file_bytes = uploaded.read()
            df_raw, err = load_file(file_bytes, uploaded.name)

        if err:
            st.error(f"❌ {err}")
            st.stop()

        st.session_state["df_raw"]     = df_raw
        st.session_state["file_name"]  = uploaded.name

        # Pre-compute profile & issues once on load
        with st.spinner("🔬 Profiling dataset …"):
            profile     = profile_dataset(df_raw)
            issues      = detect_issues(df_raw, profile)
            suggestions = generate_suggestions(df_raw, profile, issues)
        st.session_state["profile"]     = profile
        st.session_state["issues"]      = issues
        st.session_state["suggestions"] = suggestions
        gc.collect()

# Retrieve raw df from session_state
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
# DATASET PREVIEW (first N rows only)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📋 Dataset Preview</div>', unsafe_allow_html=True)
st.caption(f"Showing first {min(preview_n, rows):,} of {rows:,} rows")
st.dataframe(get_preview(df_raw, preview_n), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📊 Data Quality Overview</div>', unsafe_allow_html=True)

score, score_label, bar_colour = quality_score(df_raw)
prof_df  = column_profile(df_raw)
dupes    = int(df_raw.duplicated().sum())
missing  = int(df_raw.isna().sum().sum())

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
    ih.markdown(f'<span class="badge-high">🔴 High: {high_n}</span>',   unsafe_allow_html=True)
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

# Cleaning suggestions checklist
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
# APPLY CLEANING — only runs when button clicked
# Result stored in session_state so reruns (downloads, PDF) don't re-execute
# ─────────────────────────────────────────────────────────────────────────────

if apply_btn:
    prog = st.progress(0, text="Applying cleaning actions …")
    df_clean, report = apply_cleaning(df_raw, selected_actions)
    gc.collect()
    prog.progress(100, text="✅ Done")

    # Compute analytics immediately after cleaning
    with st.spinner("🔬 Computing analytics …"):
        kpis = compute_kpis(df_clean)

    # Persist everything in session_state
    st.session_state["df_clean"]        = df_clean
    st.session_state["cleaning_report"] = report
    st.session_state["kpis"]            = kpis
    st.session_state["pdf_bytes"]       = None   # reset old PDF when re-cleaning
    st.session_state["analysis_done"]   = True

# ─────────────────────────────────────────────────────────────────────────────
# GUARD — nothing below runs until analysis_done
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state["analysis_done"]:
    st.info("☝️ Select cleaning suggestions above and click **Apply & Analyse** to continue.")
    st.stop()

# Retrieve from session_state (survives every rerun / button click)
df_clean       = st.session_state["df_clean"]
cleaning_report= st.session_state["cleaning_report"]
kpis           = st.session_state["kpis"]

# ─────────────────────────────────────────────────────────────────────────────
# CLEANING REPORT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-hdr">📑 Cleaning Report</div>', unsafe_allow_html=True)
r1,r2,r3,r4,r5,r6 = st.columns(6)
r1.metric("Rows Before",        f"{cleaning_report['rows_before']:,}")
r2.metric("Rows After",         f"{cleaning_report['rows_after']:,}",
          delta=cleaning_report['rows_after'] - cleaning_report['rows_before'])
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

# Cleaned preview
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

# Apply sidebar filters — stores filtered df in session_state too
df_filtered = apply_filters(df_clean, kpis)
st.session_state["df_filtered"] = df_filtered

if df_filtered.empty:
    st.warning("⚠️ Filters produce an empty dataset — adjust sidebar filters.")
else:
    # Re-compute kpis for filtered view (cached — instant if same data)
    kpis_filtered = compute_kpis(df_filtered)
    generate_dashboard(df_filtered, kpis_filtered)

# ─────────────────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE CHART GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-hdr">💬 Natural Language Chart Generator</div>',
            unsafe_allow_html=True)
st.markdown("Describe a chart in plain English and it appears instantly.")

with st.expander("💡 Example prompts", expanded=False):
    ex_c = st.columns(2)
    for i, ex in enumerate(EXAMPLE_QUERIES):
        with ex_c[i % 2]:
            st.markdown(f"• *{ex}*")

nl_q = st.text_input("🔍 Describe your chart",
                     placeholder="e.g. top 10 products by revenue",
                     key="nl_chart_input")

if st.button("📈 Generate Chart", type="primary", key="nl_btn") and nl_q.strip():
    # Use filtered df for charts
    plot_df = st.session_state.get("df_filtered") or df_clean
    with st.spinner("Generating chart …"):
        fig, label = generate_chart(plot_df, nl_q)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Intent detected: `{label}`")
    else:
        st.warning(label)

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT — CSV / Excel / PDF
# All download buttons read from session_state and do NOT trigger reruns that
# lose the cleaned dataframe, because the df lives in session_state.
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-hdr">📥 Export</div>', unsafe_allow_html=True)

dl1, dl2, dl3 = st.columns(3)

# ── CSV download ───────────────────────────────────────────────────────────
with dl1:
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=df_to_csv_bytes(df_clean),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_csv",
    )

# ── Excel download ─────────────────────────────────────────────────────────
with dl2:
    st.download_button(
        label="⬇️ Download Cleaned Excel",
        data=df_to_excel_bytes(df_clean),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="btn_xlsx",
    )

# ── PDF report ─────────────────────────────────────────────────────────────
with dl3:
    if pdf_available():
        # ── Step 1: generate button ────────────────────────────────────────
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

        # ── Step 2: download button renders on the SAME rerun as generation
        #    No st.rerun() needed — session_state["pdf_bytes"] is now set,
        #    so this block executes immediately in the same script pass.
        if st.session_state["pdf_bytes"] is not None:
            st.download_button(
                label="⬇️ Download AI PDF Report",
                data=st.session_state["pdf_bytes"],
                file_name="ai_business_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="btn_pdf_dl",
            )
            if st.button(
                "🔄 Regenerate PDF",
                use_container_width=True,
                key="btn_regen_pdf",
            ):
                st.session_state["pdf_bytes"] = None
                st.rerun()
    else:
        st.info("Install reportlab to enable PDF reports:\n`pip install reportlab`")
