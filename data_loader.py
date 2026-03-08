"""
data_loader.py
Memory-Efficient Data Loading Engine
Handles CSV/Excel up to 200MB with chunk reading, dtype optimization,
and category conversion to minimize RAM usage.
"""

import io
import gc
import logging
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CHUNK_SIZE      = 50_000          # rows per CSV chunk
CATEGORY_THRESH = 0.10            # convert to category if unique ratio < 10%
MAX_CATEGORY_N  = 500             # only categorise if unique count < 500
PREVIEW_ROWS    = 100             # rows shown in UI preview
SAMPLE_ROWS     = 10_000          # rows used for heavy analytics


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=3)
def load_file(file_bytes: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a CSV or Excel file from raw bytes.
    Applies dtype optimisation and category conversion automatically.

    Args:
        file_bytes: Raw bytes from st.file_uploader (.read())
        filename:   Original filename (used to detect format)

    Returns:
        (DataFrame, None) on success | (None, error_message) on failure
    """
    fname = filename.lower()
    try:
        if fname.endswith(".csv"):
            df = _load_csv(file_bytes)
        elif fname.endswith((".xlsx", ".xls")):
            df = _load_excel(file_bytes, fname)
        else:
            return None, f"Unsupported format: '{filename}'. Use CSV or Excel."

        if df is None or df.empty:
            return None, "File is empty or could not be parsed."

        df = _clean_column_names(df)
        df = _optimise_dtypes(df)
        gc.collect()
        return df, None

    except MemoryError:
        return None, "File is too large to load in available memory."
    except Exception as exc:
        logger.exception("load_file failed")
        return None, f"Could not read file: {exc}"


def get_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> pd.DataFrame:
    """Return the first *n* rows for UI display (no copy of the full frame)."""
    return df.head(n)


def get_sample(df: pd.DataFrame, n: int = SAMPLE_ROWS, seed: int = 42) -> pd.DataFrame:
    """
    Return a representative sample for heavy analytics.
    Returns full frame if it has fewer rows than *n*.
    """
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def memory_report(df: pd.DataFrame) -> Dict[str, str]:
    """Return a human-readable memory breakdown by column."""
    mem = df.memory_usage(deep=True)
    total_mb = mem.sum() / 1024 ** 2
    report = {"__total__": f"{total_mb:.2f} MB"}
    for col in df.columns:
        report[col] = f"{mem[col] / 1024:.1f} KB"
    return report


# ── CSV loading ───────────────────────────────────────────────────────────────

def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Read CSV in chunks to avoid peak memory spikes.
    Tries multiple encodings automatically.
    """
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            chunks: List[pd.DataFrame] = []
            reader = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                chunksize=CHUNK_SIZE,
                low_memory=True,
                on_bad_lines="skip",
            )
            for chunk in reader:
                chunks.append(chunk)

            if not chunks:
                return pd.DataFrame()

            df = pd.concat(chunks, ignore_index=True)
            del chunks
            return df

        except UnicodeDecodeError:
            continue
        except Exception:
            raise

    raise ValueError("Could not decode CSV with any supported encoding.")


# ── Excel loading ─────────────────────────────────────────────────────────────

def _load_excel(file_bytes: bytes, fname: str) -> pd.DataFrame:
    """Read Excel file — uses openpyxl for .xlsx, xlrd for .xls."""
    engine = "xlrd" if fname.endswith(".xls") else "openpyxl"
    return pd.read_excel(io.BytesIO(file_bytes), engine=engine)


# ── Column name normalisation ─────────────────────────────────────────────────

def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and drop pure-index leakage columns."""
    df.columns = [str(c).strip() for c in df.columns]
    # Drop columns that look like a leaked RangeIndex
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
    return df


# ── Dtype optimisation ────────────────────────────────────────────────────────

def _optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric dtypes and convert low-cardinality strings to category.
    Can cut memory usage by 40–70 % on typical datasets.
    """
    for col in df.columns:
        col_dtype = df[col].dtype

        # --- numeric downcast ---
        if col_dtype in (np.int64, np.int32):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        elif col_dtype in (np.float64, np.float32):
            df[col] = pd.to_numeric(df[col], downcast="float")

        # --- string → category ---
        elif col_dtype == object or str(col_dtype) == "string":
            n_unique = df[col].nunique()
            ratio    = n_unique / max(len(df), 1)
            if ratio < CATEGORY_THRESH and n_unique < MAX_CATEGORY_N:
                df[col] = df[col].astype("category")

    return df
