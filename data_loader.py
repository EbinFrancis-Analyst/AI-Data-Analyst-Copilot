"""
data_loader.py
Memory-Efficient Data Loading Engine
Handles CSV / XLSX / XLS up to 200 MB using chunk reading,
dtype downcast, and category conversion.
"""

from __future__ import annotations

import io
import gc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE       = 50_000   # rows per CSV chunk
CATEGORY_THRESH  = 0.10     # unique-ratio threshold for category conversion
MAX_CATEGORY_UNQ = 500      # max unique values to still convert to category
PREVIEW_ROWS     = 100
SAMPLE_ROWS      = 10_000


# ── Public API ─────────────────────────────────────────────────────────────────

def load_file(file_bytes: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load CSV or Excel from raw bytes.  Applies dtype optimisation automatically.

    Returns (DataFrame, None) on success | (None, error_string) on failure.
    """
    fname = filename.lower().strip()
    try:
        if fname.endswith(".csv"):
            df = _load_csv(file_bytes)
        elif fname.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        elif fname.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes), engine="xlrd")
        else:
            return None, f"Unsupported format '{filename}'. Upload CSV or Excel."

        if df is None or df.empty:
            return None, "File is empty or could not be parsed."

        df = _clean_column_names(df)
        df = _optimise_dtypes(df)
        gc.collect()
        return df, None

    except MemoryError:
        return None, "File is too large to load. Try a smaller file."
    except Exception as exc:
        logger.exception("load_file failed")
        return None, f"Could not read file: {exc}"


def get_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> pd.DataFrame:
    """Return the first *n* rows for display — never copies the full frame."""
    return df.head(n)


def get_sample(df: pd.DataFrame, n: int = SAMPLE_ROWS, seed: int = 42) -> pd.DataFrame:
    """Return a representative sample for heavy analytics."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def memory_report(df: pd.DataFrame) -> Dict[str, str]:
    """Return human-readable memory usage per column plus total."""
    mem   = df.memory_usage(deep=True)
    total = mem.sum() / 1024 ** 2
    report: Dict[str, str] = {"__total__": f"{total:.2f} MB"}
    for col in df.columns:
        report[col] = f"{mem[col] / 1024:.1f} KB"
    return report


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV in chunks across multiple encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            chunks: List[pd.DataFrame] = []
            reader = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=enc,
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
    raise ValueError("Could not decode CSV with any supported encoding.")


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
    return df


def _optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics and convert low-cardinality strings to category."""
    for col in df.columns:
        dt = df[col].dtype
        if dt in (np.int64, np.int32, np.int16):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif dt in (np.float64, np.float32):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif dt == object or str(dt) == "string":
            n_unique = df[col].nunique()
            ratio    = n_unique / max(len(df), 1)
            if ratio < CATEGORY_THRESH and n_unique < MAX_CATEGORY_UNQ:
                df[col] = df[col].astype("category")
    return df
