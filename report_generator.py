"""
report_generator.py
AI Business Report Generator — produces a professional PDF via ReportLab.
No external API required. Fully self-contained.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        BaseDocTemplate, Flowable, Frame, HRFlowable,
        PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    _OK = True
except ImportError:
    _OK = False


def is_available() -> bool:
    return _OK


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"{v/1_000:.1f}K"
    return f"{v:,.2f}"

def _find(df: pd.DataFrame, kws: List[str]) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for kw in kws:
        for nl, nc in cl.items():
            if kw in nl: return nc
    return None

def _num(df, hints):
    c = _find(df, hints)
    return c if c and pd.api.types.is_numeric_dtype(df[c]) else \
           (df.select_dtypes(include=[np.number]).columns.tolist() or [None])[0]

def _cat(df, hints):
    c = _find(df, hints)
    return c if c and not pd.api.types.is_numeric_dtype(df[c]) else \
           (df.select_dtypes(include=["object","category"]).columns.tolist() or [None])[0]


# ── Colours ────────────────────────────────────────────────────────────────────

if _OK:
    C_NAVY  = colors.HexColor("#0D1B2A")
    C_BLUE  = colors.HexColor("#4F8BF9")
    C_PUR   = colors.HexColor("#A259FF")
    C_GREEN = colors.HexColor("#43D9AD")
    C_WARN  = colors.HexColor("#FFB347")
    C_RED   = colors.HexColor("#FF6B6B")
    C_LIGHT = colors.HexColor("#E0E0E0")
    C_GREY  = colors.HexColor("#888888")
    C_WHITE = colors.white
    C_BLACK = colors.black
    C_BG    = colors.HexColor("#F8F9FC")
    C_RULE  = colors.HexColor("#D0D5E8")


# ── Styles ─────────────────────────────────────────────────────────────────────

def _styles() -> Dict[str, ParagraphStyle]:
    s = {}
    s["cover_title"] = ParagraphStyle("cover_title", fontSize=26, fontName="Helvetica-Bold",
                                       textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=8)
    s["cover_sub"]   = ParagraphStyle("cover_sub",   fontSize=13, fontName="Helvetica",
                                       textColor=C_LIGHT, alignment=TA_CENTER, spaceAfter=4)
    s["h1"]          = ParagraphStyle("h1",  fontSize=15, fontName="Helvetica-Bold",
                                       textColor=C_BLUE,  spaceBefore=14, spaceAfter=6)
    s["h2"]          = ParagraphStyle("h2",  fontSize=12, fontName="Helvetica-Bold",
                                       textColor=C_PUR,   spaceBefore=10, spaceAfter=4)
    s["body"]        = ParagraphStyle("body",fontSize=10, fontName="Helvetica",
                                       textColor=C_BLACK, leading=14, spaceAfter=4)
    s["bullet"]      = ParagraphStyle("bullet",fontSize=10, fontName="Helvetica",
                                       textColor=C_BLACK, leading=14,
                                       leftIndent=14, bulletIndent=6, spaceAfter=3)
    s["kpi_label"]   = ParagraphStyle("kpi_label",fontSize=8,  fontName="Helvetica",
                                       textColor=C_GREY, alignment=TA_CENTER)
    s["kpi_value"]   = ParagraphStyle("kpi_value",fontSize=17, fontName="Helvetica-Bold",
                                       textColor=C_BLUE, alignment=TA_CENTER)
    s["footer"]      = ParagraphStyle("footer",fontSize=8, fontName="Helvetica",
                                       textColor=C_GREY, alignment=TA_RIGHT)
    return s


# ── Doc template ───────────────────────────────────────────────────────────────

def _build_doc(buf: io.BytesIO) -> BaseDocTemplate:
    doc = BaseDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                          leftMargin=2*cm, rightMargin=2*cm, title="AI Business Report")
    w, h = A4

    def _hf(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(C_GREY)
            canvas.drawString(2*cm, h-1.3*cm, "AI Data Analyst Copilot  |  Confidential")
            canvas.drawRightString(w-2*cm, h-1.3*cm,
                                   f"Page {doc.page}  |  {datetime.now().strftime('%B %d, %Y')}")
            canvas.setStrokeColor(C_RULE)
            canvas.setLineWidth(.5)
            canvas.line(2*cm, h-1.5*cm, w-2*cm, h-1.5*cm)
            canvas.line(2*cm, 1.5*cm, w-2*cm, 1.5*cm)
        canvas.restoreState()

    frame = Frame(2*cm, 2*cm, A4[0]-4*cm, A4[1]-4*cm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_hf)])
    return doc


# ── Table helpers ──────────────────────────────────────────────────────────────

def _kpi_table(items: List[Tuple[str,str]], S: Dict) -> Table:
    rows = []; row = []
    for i, (label, value) in enumerate(items):
        row.append([Paragraph(value, S["kpi_value"]), Paragraph(label, S["kpi_label"])])
        if len(row) == 3 or i == len(items)-1:
            while len(row) < 3: row.append("")
            rows.append(row); row = []
    cw = (A4[0]-4*cm)/3
    t  = Table(rows, colWidths=[cw]*3, rowHeights=[2*cm]*len(rows))
    t.setStyle(TableStyle([
        ("BOX",          (0,0),(-1,-1),.5, C_RULE),
        ("INNERGRID",    (0,0),(-1,-1),.5, C_RULE),
        ("BACKGROUND",   (0,0),(-1,-1),    C_BG),
        ("VALIGN",       (0,0),(-1,-1),    "MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1),    8),
        ("BOTTOMPADDING",(0,0),(-1,-1),    8),
    ]))
    return t

def _data_table(df_t: pd.DataFrame, S: Dict, max_rows: int = 12) -> Table:
    df_t = df_t.head(max_rows)
    hdrs = [Paragraph(f"<b>{c}</b>", S["body"]) for c in df_t.columns]
    body = [[Paragraph(str(v), S["body"]) for v in row] for row in df_t.values]
    cw   = (A4[0]-4*cm)/len(df_t.columns)
    t    = Table([hdrs]+body, colWidths=[cw]*len(df_t.columns), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),        C_BLUE),
        ("TEXTCOLOR",     (0,0),(-1,0),        C_WHITE),
        ("FONTNAME",      (0,0),(-1,0),        "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1),        9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),       [C_WHITE, C_BG]),
        ("BOX",           (0,0),(-1,-1),.5,    C_RULE),
        ("INNERGRID",     (0,0),(-1,-1),.3,    C_RULE),
        ("TOPPADDING",    (0,0),(-1,-1),        4),
        ("BOTTOMPADDING", (0,0),(-1,-1),        4),
        ("LEFTPADDING",   (0,0),(-1,-1),        6),
    ]))
    return t


# ── Sections ───────────────────────────────────────────────────────────────────

def _cover(story, S, name):
    w, h = A4

    class _BG(Flowable):
        def draw(self):
            self.canv.setFillColor(C_NAVY)
            self.canv.rect(-2*cm, -h+2*cm, w, h, fill=1, stroke=0)
            self.canv.setFillColor(C_BLUE)
            self.canv.rect(-2*cm, h*.28-2*cm, w, .6*cm, fill=1, stroke=0)
            self.canv.setFillColor(C_PUR)
            self.canv.rect(-2*cm, h*.28-2*cm-.7*cm, w, .3*cm, fill=1, stroke=0)

    story.append(_BG())
    story.append(Spacer(1, 5*cm))
    story.append(Paragraph("AI Business Intelligence Report", S["cover_title"]))
    story.append(Spacer(1, .4*cm))
    story.append(Paragraph(f"Dataset: {name}", S["cover_sub"]))
    story.append(Spacer(1, .2*cm))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}", S["cover_sub"]))
    story.append(Spacer(1, .2*cm))
    story.append(Paragraph("Powered by AI Data Analyst Copilot", S["cover_sub"]))
    story.append(PageBreak())


def _exec_summary(story, S, df, kpis):
    story.append(Paragraph("1. Executive Summary", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    rows, cols = df.shape
    comp = round((1-df.isna().sum().sum()/max(df.size,1))*100, 1)
    story.append(Paragraph(
        f"This report presents an automated analysis of a dataset containing "
        f"<b>{rows:,} records</b> across <b>{cols} columns</b> with data "
        f"completeness of <b>{comp}%</b>.", S["body"]))
    story.append(Spacer(1, .3*cm))
    items = []
    for k, label in [("total_revenue","Total Revenue"),("total_orders","Total Orders"),
                     ("aov","Avg Order Value"),("unique_customers","Unique Customers"),
                     ("unique_products","Unique Products"),("unique_regions","Regions")]:
        v = kpis.get(k)
        if v and v not in ("N/A", "—"):
            items.append((label, str(v)))
    if items:
        story.append(Paragraph("Key Performance Indicators", S["h2"]))
        story.append(_kpi_table(items, S))
    story.append(Spacer(1, .4*cm))


def _revenue_section(story, S, df, kpis):
    sc = kpis.get("sales_col"); cc = kpis.get("city_col"); pc = kpis.get("prod_col")
    if not sc: return
    story.append(Paragraph("2. Revenue & Sales Performance", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    total = df[sc].sum(); mean = df[sc].mean(); med = df[sc].median()
    story.append(Paragraph(
        f"Total revenue is <b>{_fmt(total)}</b>. "
        f"Average transaction value is <b>{_fmt(mean)}</b> (median: {_fmt(med)}).", S["body"]))
    if cc:
        grp = df.groupby(cc, observed=True)[sc].sum().sort_values(ascending=False)
        tv  = grp.iloc[0]; tci = grp.index[0]
        story.append(Paragraph(
            f"The highest-revenue region is <b>{tci}</b>, contributing "
            f"<b>{_fmt(tv)}</b> ({tv/total*100:.1f}% of total).", S["body"]))
        top5 = grp.head(5).reset_index()
        top5.columns = [cc, "Revenue"]
        top5["Revenue"] = top5["Revenue"].apply(_fmt)
        story.append(Spacer(1, .2*cm))
        story.append(Paragraph("Top 5 Regions by Revenue", S["h2"]))
        story.append(_data_table(top5, S))
    if pc:
        grp2 = df.groupby(pc, observed=True)[sc].sum().sort_values(ascending=False)
        top5p = grp2.head(5).reset_index()
        top5p.columns = [pc, "Revenue"]
        top5p["Revenue"] = top5p["Revenue"].apply(_fmt)
        story.append(Spacer(1, .3*cm))
        story.append(Paragraph("Top 5 Products by Revenue", S["h2"]))
        story.append(_data_table(top5p, S))
    story.append(Spacer(1, .4*cm))


def _customer_section(story, S, df, kpis):
    cu = kpis.get("cust_col"); sc = kpis.get("sales_col")
    if not cu: return
    story.append(Paragraph("3. Customer Analysis", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    story.append(Paragraph(f"Dataset contains <b>{df[cu].nunique():,} unique customers</b>.", S["body"]))
    if sc:
        top = df.groupby(cu, observed=True)[sc].sum().sort_values(ascending=False)
        top5 = top.head(5).reset_index()
        top5.columns = [cu, "Total Spend"]
        sh   = top.head(5).sum() / top.sum() * 100
        story.append(Paragraph(
            f"Top 5 customers account for <b>{sh:.1f}%</b> of total revenue.", S["body"]))
        top5["Total Spend"] = top5["Total Spend"].apply(_fmt)
        story.append(Spacer(1, .2*cm))
        story.append(Paragraph("Top 5 Customers by Spend", S["h2"]))
        story.append(_data_table(top5, S))
    story.append(Spacer(1, .4*cm))


def _product_section(story, S, df, kpis):
    pc = kpis.get("prod_col"); sc = kpis.get("sales_col"); qc = kpis.get("qty_col")
    if not pc: return
    story.append(Paragraph("4. Product Performance", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    freq = df[pc].value_counts()
    story.append(Paragraph(
        f"There are <b>{df[pc].nunique():,} unique products</b>. "
        f"Most frequent: <b>{freq.index[0]}</b> ({freq.iloc[0]:,} orders).", S["body"]))
    if qc:
        tq = df.groupby(pc, observed=True)[qc].sum().sort_values(ascending=False).head(5).reset_index()
        tq.columns = [pc, "Units Sold"]
        story.append(Spacer(1, .2*cm))
        story.append(Paragraph("Top 5 Products by Units Sold", S["h2"]))
        story.append(_data_table(tq, S))
    story.append(Spacer(1, .4*cm))


def _regional_section(story, S, df, kpis):
    cc = kpis.get("city_col"); sc = kpis.get("sales_col")
    if not cc or not sc: return
    story.append(Paragraph("5. Regional Performance", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    grp   = df.groupby(cc, observed=True)[sc].sum().sort_values(ascending=False)
    total = grp.sum()
    top3  = ", ".join(f"{r} ({v/total*100:.0f}%)" for r, v in grp.head(3).items())
    story.append(Paragraph(f"Top 3 regions: <b>{top3}</b>.", S["body"]))
    story.append(Paragraph(
        f"Lowest: <b>{grp.index[-1]}</b> — {_fmt(grp.iloc[-1])}.", S["body"]))
    full = grp.reset_index()
    full.columns = [cc, "Revenue"]
    full["Revenue"] = full["Revenue"].apply(_fmt)
    full["Share %"] = (grp/total*100).round(1).values
    story.append(Spacer(1, .2*cm))
    story.append(Paragraph("Regional Revenue Breakdown", S["h2"]))
    story.append(_data_table(full, S, max_rows=15))
    story.append(Spacer(1, .4*cm))


def _trends_section(story, S, df, kpis):
    dc = kpis.get("date_col"); sc = kpis.get("sales_col")
    if not dc or not sc: return
    story.append(Paragraph("6. Trend Analysis", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    try:
        tmp = df[[dc, sc]].copy()
        tmp[dc] = pd.to_datetime(tmp[dc], errors="coerce")
        monthly = tmp.dropna(subset=[dc]).resample("ME", on=dc)[sc].sum()
        peak    = monthly.idxmax().strftime("%B %Y")
        trough  = monthly.idxmin().strftime("%B %Y")
        if len(monthly) >= 2:
            d = monthly.iloc[-1] - monthly.iloc[-2]
            dir_str = "upward 📈" if d > 0 else "downward 📉"
            story.append(Paragraph(
                f"Revenue peaked in <b>{peak}</b> and was lowest in <b>{trough}</b>. "
                f"Most recent month shows a <b>{dir_str}</b> trend "
                f"({_fmt(abs(d))} vs prior month).", S["body"]))
        tbl = monthly.reset_index()
        tbl.columns = ["Month", "Revenue"]
        tbl["Month"]   = tbl["Month"].dt.strftime("%b %Y")
        tbl["Revenue"] = tbl["Revenue"].apply(_fmt)
        story.append(Spacer(1, .2*cm))
        story.append(Paragraph("Monthly Revenue Summary", S["h2"]))
        story.append(_data_table(tbl, S, max_rows=14))
    except Exception:
        story.append(Paragraph("Trend data could not be computed.", S["body"]))
    story.append(Spacer(1, .4*cm))


def _corr_section(story, S, df):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(nums) < 2: return
    story.append(Paragraph("7. Correlation Findings", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    corr  = df[nums].corr().round(3)
    pairs = []
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((nums[i], nums[j], float(v)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for a, b, v in pairs[:6]:
        lbl = ("very strong" if abs(v)>.8 else "strong" if abs(v)>.6
               else "moderate" if abs(v)>.4 else "weak")
        dir_s = "positive" if v > 0 else "negative"
        story.append(Paragraph(
            f"• <b>{a}</b> ↔ <b>{b}</b>: {lbl} {dir_s} correlation (r = {v:.3f})", S["bullet"]))
    story.append(Spacer(1, .4*cm))


def _recs_section(story, S, df, kpis):
    story.append(Paragraph("8. Strategic Recommendations", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, .3*cm))
    sc = kpis.get("sales_col"); cc = kpis.get("city_col")
    pc = kpis.get("prod_col");  cu = kpis.get("cust_col")
    recs = []
    if cc and sc:
        grp = df.groupby(cc, observed=True)[sc].sum().sort_values(ascending=False)
        top = grp.index[0]; sh = grp.iloc[0]/grp.sum()*100
        recs.append(f"Prioritise <b>{top}</b> — it drives {sh:.0f}% of revenue. "
                    "Consider dedicated marketing and inventory allocation.")
        recs.append(f"Investigate underperformance in <b>{grp.index[-1]}</b>. "
                    "A targeted pricing or outreach strategy is recommended.")
    if pc and sc:
        grp = df.groupby(pc, observed=True)[sc].sum().sort_values(ascending=False)
        recs.append(f"Protect supply of <b>{grp.index[0]}</b>, the highest-revenue category.")
    if cu and sc:
        grp = df.groupby(cu, observed=True)[sc].sum()
        sh  = grp.sort_values(ascending=False).head(5).sum()/grp.sum()*100
        if sh > 20:
            recs.append(f"Top 5 customers represent {sh:.0f}% of revenue — "
                        "introduce a VIP retention programme.")
    comp = round((1-df.isna().sum().sum()/max(df.size,1))*100,1)
    if comp < 95:
        recs.append(f"Data completeness is {comp}%. Improve data collection processes.")
    recs.append("Schedule monthly automated report runs to track KPI trends.")
    recs.append("Integrate with live data sources for real-time dashboard monitoring.")
    for r in recs:
        story.append(Paragraph(f"• {r}", S["bullet"]))
    story.append(Spacer(1, .4*cm))


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_pdf_report(
    df: pd.DataFrame,
    kpis: Dict[str, Any],
    dataset_name: str = "Dataset",
) -> bytes:
    """
    Generate a multi-section professional PDF report.

    Args:
        df:           Cleaned DataFrame
        kpis:         Pre-computed KPIs from analytics_engine.compute_kpis()
        dataset_name: Shown on cover page

    Returns:
        Raw PDF bytes for st.download_button()

    Raises:
        ImportError: if reportlab is not installed
    """
    if not _OK:
        raise ImportError("reportlab is required: pip install reportlab")

    buf   = io.BytesIO()
    doc   = _build_doc(buf)
    S     = _styles()
    story = []

    _cover(story, S, dataset_name)
    _exec_summary(story, S, df, kpis)
    _revenue_section(story, S, df, kpis)
    _customer_section(story, S, df, kpis)
    _product_section(story, S, df, kpis)
    _regional_section(story, S, df, kpis)
    _trends_section(story, S, df, kpis)
    _corr_section(story, S, df)
    _recs_section(story, S, df, kpis)

    doc.build(story)
    return buf.getvalue()
