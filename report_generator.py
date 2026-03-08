"""
report_generator.py
AI Business Report Generator
Produces a professional multi-page PDF using ReportLab.
No external API required — all content is rule-based.
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
        BaseDocTemplate, Flowable, Frame, HRFlowable, Image,
        PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

if REPORTLAB_OK:
    C_NAVY   = colors.HexColor("#0D1B2A")
    C_BLUE   = colors.HexColor("#4F8BF9")
    C_PURPLE = colors.HexColor("#A259FF")
    C_GREEN  = colors.HexColor("#43D9AD")
    C_ORANGE = colors.HexColor("#FFB347")
    C_RED    = colors.HexColor("#FF6B6B")
    C_LIGHT  = colors.HexColor("#E0E0E0")
    C_GREY   = colors.HexColor("#888888")
    C_WHITE  = colors.white
    C_BLACK  = colors.black
    C_BG     = colors.HexColor("#F8F9FC")
    C_RULE   = colors.HexColor("#D0D5E8")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:,.2f}"


def _find(df: pd.DataFrame, kws: List[str]) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for kw in kws:
        for nl, nc in cl.items():
            if kw in nl:
                return nc
    return None


def _num(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    c = _find(df, hints)
    return c if c and pd.api.types.is_numeric_dtype(df[c]) else \
           (df.select_dtypes(include=[np.number]).columns.tolist() or [None])[0]


def _cat(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    c = _find(df, hints)
    return c if c and not pd.api.types.is_numeric_dtype(df[c]) else \
           (df.select_dtypes(include=["object","category"]).columns.tolist() or [None])[0]


# ─────────────────────────────────────────────────────────────────────────────
# STYLE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {}

    s["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=28, fontName="Helvetica-Bold",
        textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=8,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=14, fontName="Helvetica",
        textColor=C_LIGHT, alignment=TA_CENTER, spaceAfter=4,
    )
    s["h1"] = ParagraphStyle(
        "h1", fontSize=16, fontName="Helvetica-Bold",
        textColor=C_BLUE, spaceBefore=14, spaceAfter=6,
    )
    s["h2"] = ParagraphStyle(
        "h2", fontSize=13, fontName="Helvetica-Bold",
        textColor=C_PURPLE, spaceBefore=10, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body", fontSize=10, fontName="Helvetica",
        textColor=C_BLACK, leading=14, spaceAfter=4,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", fontSize=10, fontName="Helvetica",
        textColor=C_BLACK, leading=14, leftIndent=14,
        bulletIndent=6, spaceAfter=3,
    )
    s["kpi_label"] = ParagraphStyle(
        "kpi_label", fontSize=8, fontName="Helvetica",
        textColor=C_GREY, alignment=TA_CENTER,
    )
    s["kpi_value"] = ParagraphStyle(
        "kpi_value", fontSize=18, fontName="Helvetica-Bold",
        textColor=C_BLUE, alignment=TA_CENTER,
    )
    s["footer"] = ParagraphStyle(
        "footer", fontSize=8, fontName="Helvetica",
        textColor=C_GREY, alignment=TA_RIGHT,
    )
    return s


# ─────────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _build_doc(buffer: io.BytesIO) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        buffer, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="AI Business Report",
    )

    def _header_footer(canvas, doc):
        canvas.saveState()
        w, h = A4
        if doc.page > 1:
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(C_GREY)
            canvas.drawString(2*cm, h - 1.3*cm, "AI Data Analyst Copilot  |  Confidential")
            canvas.drawRightString(w - 2*cm, h - 1.3*cm,
                                   f"Page {doc.page}  |  {datetime.now().strftime('%B %d, %Y')}")
            canvas.setStrokeColor(C_RULE)
            canvas.setLineWidth(0.5)
            canvas.line(2*cm, h - 1.5*cm, w - 2*cm, h - 1.5*cm)
            canvas.line(2*cm, 1.5*cm, w - 2*cm, 1.5*cm)
        canvas.restoreState()

    frame = Frame(2*cm, 2*cm, A4[0]-4*cm, A4[1]-4*cm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=_header_footer)])
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# KPI TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_table(kpi_list: List[Tuple[str, str]], S: Dict) -> Table:
    """Render KPIs as a styled 3-column grid table."""
    rows = []
    row: List = []
    for i, (label, value) in enumerate(kpi_list):
        cell = [Paragraph(value, S["kpi_value"]),
                Paragraph(label, S["kpi_label"])]
        row.append(cell)
        if len(row) == 3 or i == len(kpi_list) - 1:
            while len(row) < 3:
                row.append("")
            rows.append(row)
            row = []

    col_w = (A4[0] - 4*cm) / 3
    tbl = Table(rows, colWidths=[col_w]*3, rowHeights=[2*cm]*len(rows))
    tbl.setStyle(TableStyle([
        ("BOX",         (0,0), (-1,-1), 0.5,  C_RULE),
        ("INNERGRID",   (0,0), (-1,-1), 0.5,  C_RULE),
        ("BACKGROUND",  (0,0), (-1,-1),       C_BG),
        ("VALIGN",      (0,0), (-1,-1),       "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1),       8),
        ("BOTTOMPADDING",(0,0),(-1,-1),       8),
    ]))
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _data_table(df_table: pd.DataFrame, S: Dict,
                max_rows: int = 12) -> Table:
    """Convert a small DataFrame into a styled ReportLab table."""
    df_t = df_table.head(max_rows)
    headers = [Paragraph(f"<b>{c}</b>", S["body"]) for c in df_t.columns]
    body    = [[Paragraph(str(v), S["body"]) for v in row]
               for row in df_t.values]

    col_w = (A4[0] - 4*cm) / len(df_t.columns)
    tbl   = Table([headers] + body,
                  colWidths=[col_w]*len(df_t.columns),
                  repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),        C_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0),        C_WHITE),
        ("FONTNAME",    (0,0), (-1,0),        "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1),        9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),      [C_WHITE, C_BG]),
        ("BOX",         (0,0), (-1,-1), 0.5,  C_RULE),
        ("INNERGRID",   (0,0), (-1,-1), 0.3,  C_RULE),
        ("TOPPADDING",  (0,0), (-1,-1),        4),
        ("BOTTOMPADDING",(0,0),(-1,-1),        4),
        ("LEFTPADDING", (0,0), (-1,-1),        6),
    ]))
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# COVER PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _cover_page(story: List, S: Dict, dataset_name: str):
    """Build a dark gradient cover page."""
    w, h = A4

    class CoverBackground(Flowable):
        def draw(self):
            self.canv.setFillColor(C_NAVY)
            self.canv.rect(-2*cm, -h+2*cm, w, h, fill=1, stroke=0)
            # accent stripe
            self.canv.setFillColor(C_BLUE)
            self.canv.rect(-2*cm, h*0.28-2*cm, w, 0.6*cm, fill=1, stroke=0)
            self.canv.setFillColor(C_PURPLE)
            self.canv.rect(-2*cm, h*0.28-2*cm - 0.7*cm, w, 0.3*cm, fill=1, stroke=0)

    story.append(CoverBackground())
    story.append(Spacer(1, 5*cm))
    story.append(Paragraph("AI Business Intelligence Report", S["cover_title"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Dataset: {dataset_name}", S["cover_sub"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        S["cover_sub"],
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Powered by AI Data Analyst Copilot", S["cover_sub"]))
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────────────────
# REPORT SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _section_executive_summary(story, S, df, kpis):
    story.append(Paragraph("1. Executive Summary", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    rows, cols  = df.shape
    comp        = round((1 - df.isna().sum().sum() / max(df.size,1)) * 100, 1)
    date_range  = ""
    date_col = _find(df, ["date","time","month","year"])
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            date_range = f"{dates.min().strftime('%b %Y')} – {dates.max().strftime('%b %Y')}"
        except Exception:
            pass

    story.append(Paragraph(
        f"This report presents an automated analysis of a dataset containing "
        f"<b>{rows:,} records</b> across <b>{cols} columns</b> with a data "
        f"completeness of <b>{comp}%</b>."
        + (f" The data spans <b>{date_range}</b>." if date_range else ""),
        S["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    kpi_items = []
    for k in ["total_revenue","total_orders","aov","unique_customers",
              "unique_products","unique_regions"]:
        if kpis.get(k) and kpis[k] != "N/A":
            label = k.replace("_"," ").title()
            kpi_items.append((label, str(kpis[k])))
    if kpi_items:
        story.append(Paragraph("Key Performance Indicators", S["h2"]))
        story.append(_kpi_table(kpi_items, S))
    story.append(Spacer(1, 0.4*cm))


def _section_revenue(story, S, df, kpis):
    sales_col = kpis.get("sales_col")
    city_col  = kpis.get("city_col")
    prod_col  = kpis.get("prod_col")

    if not sales_col:
        return

    story.append(Paragraph("2. Revenue & Sales Performance", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    total = df[sales_col].sum()
    mean  = df[sales_col].mean()
    med   = df[sales_col].median()
    story.append(Paragraph(
        f"Total revenue is <b>{_fmt(total)}</b>. "
        f"Average transaction value is <b>{_fmt(mean)}</b> (median: {_fmt(med)}). ",
        S["body"],
    ))

    if city_col:
        grp   = df.groupby(city_col, observed=True)[sales_col].sum().sort_values(ascending=False)
        top_c = grp.index[0];  top_v = grp.iloc[0]
        story.append(Paragraph(
            f"The highest-revenue region is <b>{top_c}</b> contributing "
            f"<b>{_fmt(top_v)}</b> ({top_v/total*100:.1f}% of total).",
            S["body"],
        ))
        top5 = grp.head(5).reset_index()
        top5.columns = [city_col, "Revenue"]
        top5["Revenue"] = top5["Revenue"].apply(_fmt)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Top 5 Regions by Revenue", S["h2"]))
        story.append(_data_table(top5, S))

    if prod_col:
        grp2  = df.groupby(prod_col, observed=True)[sales_col].sum().sort_values(ascending=False)
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Top 5 Products by Revenue", S["h2"]))
        top5p = grp2.head(5).reset_index()
        top5p.columns = [prod_col, "Revenue"]
        top5p["Revenue"] = top5p["Revenue"].apply(_fmt)
        story.append(_data_table(top5p, S))
    story.append(Spacer(1, 0.4*cm))


def _section_customers(story, S, df, kpis):
    cust_col  = kpis.get("cust_col")
    sales_col = kpis.get("sales_col")
    if not cust_col:
        return

    story.append(Paragraph("3. Customer Analysis", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    n_cust = df[cust_col].nunique()
    story.append(Paragraph(
        f"The dataset contains <b>{n_cust:,} unique customers</b>.",
        S["body"],
    ))

    if sales_col:
        top_custs = (df.groupby(cust_col, observed=True)[sales_col]
                       .sum().sort_values(ascending=False).head(5).reset_index())
        top_custs.columns = [cust_col, "Total Spend"]
        top_custs["Total Spend"] = top_custs["Total Spend"].apply(_fmt)
        top5_total = df.groupby(cust_col, observed=True)[sales_col].sum().head(5).sum()
        total_rev  = df[sales_col].sum()
        story.append(Paragraph(
            f"The top 5 customers account for "
            f"<b>{top5_total/total_rev*100:.1f}%</b> of total revenue.",
            S["body"],
        ))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Top 5 Customers by Spend", S["h2"]))
        story.append(_data_table(top_custs, S))
    story.append(Spacer(1, 0.4*cm))


def _section_products(story, S, df, kpis):
    prod_col  = kpis.get("prod_col")
    sales_col = kpis.get("sales_col")
    qty_col   = kpis.get("qty_col")
    if not prod_col:
        return

    story.append(Paragraph("4. Product Performance", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    freq = df[prod_col].value_counts()
    story.append(Paragraph(
        f"There are <b>{df[prod_col].nunique():,} unique products</b>. "
        f"The most frequently ordered is <b>{freq.index[0]}</b> "
        f"({freq.iloc[0]:,} orders).",
        S["body"],
    ))

    if qty_col:
        top_qty = (df.groupby(prod_col, observed=True)[qty_col]
                     .sum().sort_values(ascending=False).head(5).reset_index())
        top_qty.columns = [prod_col, "Units Sold"]
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Top 5 Products by Units Sold", S["h2"]))
        story.append(_data_table(top_qty, S))
    story.append(Spacer(1, 0.4*cm))


def _section_regional(story, S, df, kpis):
    city_col  = kpis.get("city_col")
    sales_col = kpis.get("sales_col")
    if not city_col or not sales_col:
        return

    story.append(Paragraph("5. Regional Performance", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    grp   = df.groupby(city_col, observed=True)[sales_col].sum().sort_values(ascending=False)
    total = grp.sum()
    top3  = ", ".join(f"{r} ({v/total*100:.0f}%)" for r, v in grp.head(3).items())
    bot   = grp.index[-1];  bot_v = grp.iloc[-1]

    story.append(Paragraph(
        f"Top 3 revenue regions: <b>{top3}</b>.", S["body"]))
    story.append(Paragraph(
        f"The lowest-performing region is <b>{bot}</b> with {_fmt(bot_v)} revenue.",
        S["body"],
    ))
    full_tbl = grp.reset_index()
    full_tbl.columns = [city_col, "Revenue"]
    full_tbl["Revenue"] = full_tbl["Revenue"].apply(_fmt)
    full_tbl["Share %"] = (grp / total * 100).round(1).values
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Full Regional Revenue Breakdown", S["h2"]))
    story.append(_data_table(full_tbl, S, max_rows=15))
    story.append(Spacer(1, 0.4*cm))


def _section_trends(story, S, df, kpis):
    date_col  = kpis.get("date_col")
    sales_col = kpis.get("sales_col")
    if not date_col or not sales_col:
        return

    story.append(Paragraph("6. Trend Analysis", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    try:
        tmp = df[[date_col, sales_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        monthly = tmp.resample("ME", on=date_col)[sales_col].sum()
        peak    = monthly.idxmax().strftime("%B %Y")
        trough  = monthly.idxmin().strftime("%B %Y")
        delta   = monthly.iloc[-1] - monthly.iloc[-2] if len(monthly) >= 2 else 0
        direction = "upward 📈" if delta > 0 else "downward 📉"

        story.append(Paragraph(
            f"Monthly revenue peaked in <b>{peak}</b> and was lowest in <b>{trough}</b>. "
            f"The most recent month shows a <b>{direction}</b> trend "
            f"with a change of <b>{_fmt(abs(delta))}</b> vs the prior month.",
            S["body"],
        ))
        tbl_data = monthly.reset_index()
        tbl_data.columns = ["Month", "Revenue"]
        tbl_data["Month"]   = tbl_data["Month"].dt.strftime("%b %Y")
        tbl_data["Revenue"] = tbl_data["Revenue"].apply(_fmt)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Monthly Revenue Summary", S["h2"]))
        story.append(_data_table(tbl_data, S, max_rows=14))
    except Exception:
        story.append(Paragraph("Trend data could not be computed.", S["body"]))
    story.append(Spacer(1, 0.4*cm))


def _section_correlations(story, S, df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return

    story.append(Paragraph("7. Correlation Findings", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    corr = df[num_cols].corr().round(3)
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((num_cols[i], num_cols[j], float(v)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    for a, b, v in pairs[:5]:
        label = ("very strong" if abs(v) > 0.8 else "strong" if abs(v) > 0.6
                 else "moderate" if abs(v) > 0.4 else "weak")
        direction = "positive" if v > 0 else "negative"
        story.append(Paragraph(
            f"• <b>{a}</b> and <b>{b}</b>: {label} {direction} correlation (r = {v:.3f})",
            S["bullet"],
        ))

    if not pairs:
        story.append(Paragraph("No notable correlations found.", S["body"]))
    story.append(Spacer(1, 0.4*cm))


def _section_recommendations(story, S, df, kpis):
    story.append(Paragraph("8. Strategic Recommendations", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_RULE))
    story.append(Spacer(1, 0.3*cm))

    recs = _auto_recommendations(df, kpis)
    for rec in recs:
        story.append(Paragraph(f"• {rec}", S["bullet"]))
    story.append(Spacer(1, 0.4*cm))


def _auto_recommendations(df: pd.DataFrame, kpis: Dict) -> List[str]:
    recs = []
    sales_col = kpis.get("sales_col")
    city_col  = kpis.get("city_col")
    prod_col  = kpis.get("prod_col")
    cust_col  = kpis.get("cust_col")

    if city_col and sales_col:
        grp   = df.groupby(city_col, observed=True)[sales_col].sum().sort_values(ascending=False)
        top   = grp.index[0]
        bot   = grp.index[-1]
        share = grp.iloc[0] / grp.sum() * 100
        recs.append(f"Double down on <b>{top}</b> — it drives {share:.0f}% of revenue. "
                    "Consider dedicated marketing campaigns for this region.")
        recs.append(f"Investigate underperformance in <b>{bot}</b>. "
                    "A targeted outreach or pricing strategy review is recommended.")

    if prod_col and sales_col:
        grp = df.groupby(prod_col, observed=True)[sales_col].sum().sort_values(ascending=False)
        top = grp.index[0]
        recs.append(f"Ensure sufficient inventory and promotion for <b>{top}</b>, "
                    "the highest-revenue product category.")

    if cust_col and sales_col:
        grp   = df.groupby(cust_col, observed=True)[sales_col].sum()
        top5  = grp.sort_values(ascending=False).head(5).sum()
        share = top5 / grp.sum() * 100
        if share > 20:
            recs.append(f"Top 5 customers generate {share:.0f}% of revenue — "
                        "introduce a VIP retention programme to reduce concentration risk.")

    comp = round((1 - df.isna().sum().sum() / max(df.size,1)) * 100, 1)
    if comp < 95:
        recs.append(f"Data completeness is {comp}%. "
                    "Invest in data collection processes to improve analytical reliability.")

    recs.append("Schedule monthly automated report runs to track KPI trends over time.")
    recs.append("Integrate with live data sources for real-time dashboard monitoring.")
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    df: pd.DataFrame,
    kpis: Dict[str, Any],
    dataset_name: str = "Dataset",
) -> bytes:
    """
    Generate a multi-page professional PDF business report.

    Args:
        df:           The (cleaned) DataFrame to analyse
        kpis:         Pre-computed KPI dict from analytics_engine.compute_kpis()
        dataset_name: Display name shown on the cover page

    Returns:
        Raw PDF bytes ready for st.download_button()

    Raises:
        ImportError: If ReportLab is not installed
    """
    if not REPORTLAB_OK:
        raise ImportError(
            "ReportLab is required. Install with: pip install reportlab"
        )

    buffer = io.BytesIO()
    doc    = _build_doc(buffer)
    S      = _styles()
    story  = []

    _cover_page(story, S, dataset_name)
    _section_executive_summary(story, S, df, kpis)
    _section_revenue(story, S, df, kpis)
    _section_customers(story, S, df, kpis)
    _section_products(story, S, df, kpis)
    _section_regional(story, S, df, kpis)
    _section_trends(story, S, df, kpis)
    _section_correlations(story, S, df)
    _section_recommendations(story, S, df, kpis)

    doc.build(story)
    return buffer.getvalue()


def is_available() -> bool:
    """Return True if ReportLab is installed and PDF generation is possible."""
    return REPORTLAB_OK
