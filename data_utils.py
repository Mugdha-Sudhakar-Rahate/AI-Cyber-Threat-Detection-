"""
data_utils.py
-------------
Utility functions for loading, validating, and preprocessing network traffic
CSV datasets for binary cyber-threat classification.

Supports the CIC-IDS2017 column naming convention and any compatible dataset
that contains a 'Label' column with BENIGN / attack class values.
"""

import io
import numpy as np
import pandas as pd

# ── Column names that must NOT be used as model features ──────────────────────
NON_FEATURE_KEYWORDS = [
    "label", "class", "target", "flow id", "source ip", "destination ip",
    "src ip", "dst ip", "timestamp", "src port", "dst port",
]

# ── Labels treated as BENIGN traffic ──────────────────────────────────────────
BENIGN_LABELS = {"benign", "normal", "0", "legitimate"}

# ── Canonical label values used throughout the project ────────────────────────
LABEL_BENIGN = "BENIGN"
LABEL_ATTACK = "ATTACK"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(source) -> pd.DataFrame:
    """
    Load a CSV file from a file path string *or* a Streamlit UploadedFile object.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, str):
        df = pd.read_csv(source, low_memory=False)
    else:
        # Streamlit UploadedFile – read bytes then parse
        content = source.read()
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    return df


def detect_label_column(df: pd.DataFrame) -> str | None:
    """
    Return the name of the label column, or None if it cannot be found.
    Searches column names case-insensitively for 'label', 'class', 'target'.
    """
    for col in df.columns:
        if col.strip().lower() in ("label", "class", "target"):
            return col
    return None


def map_labels(series: pd.Series) -> pd.Series:
    """
    Map raw label strings to the two canonical classes: BENIGN or ATTACK.
    Any value not in BENIGN_LABELS is treated as ATTACK.
    """
    return series.apply(
        lambda v: LABEL_BENIGN if str(v).strip().lower() in BENIGN_LABELS else LABEL_ATTACK
    )


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove common data-quality issues from a network-traffic DataFrame.

    Steps
    -----
    1. Strip whitespace from column names.
    2. Drop fully-duplicate rows.
    3. Replace ±inf with NaN.
    4. Drop rows that are entirely NaN.
    5. Fill remaining NaN numerics with column median.

    Returns
    -------
    cleaned_df : pd.DataFrame
    report     : dict  – summary of changes made
    """
    report = {}
    df = df.copy()

    # 1. Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = n_before - len(df)

    # 3. Infinities → NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    report["inf_values_replaced"] = int(n_inf)

    # 4. All-NaN rows
    n_before = len(df)
    df = df.dropna(how="all")
    report["empty_rows_dropped"] = n_before - len(df)

    # 5. Fill remaining NaN with median
    n_missing = df[numeric_cols].isna().sum().sum()
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if not np.isnan(median) else 0)
    report["missing_values_imputed"] = int(n_missing)

    report["final_rows"] = len(df)
    return df, report


def prepare_features(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a cleaned DataFrame into feature matrix X and label series y.

    * Maps raw labels to BENIGN / ATTACK.
    * Drops any column whose lower-case name matches NON_FEATURE_KEYWORDS.
    * Keeps only numeric columns as features.

    Returns
    -------
    X : pd.DataFrame  – numeric feature matrix
    y : pd.Series     – binary label series (BENIGN / ATTACK)
    """
    df = df.copy()

    # Map labels
    df[label_col] = map_labels(df[label_col])
    y = df[label_col]

    # Identify columns to drop
    drop_cols = set()
    for col in df.columns:
        lower = col.strip().lower()
        for kw in NON_FEATURE_KEYWORDS:
            if kw in lower:
                drop_cols.add(col)
                break

    X = df.drop(columns=list(drop_cols))

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    return X, y


def get_dataset_summary(df: pd.DataFrame, label_col: str | None = None) -> dict:
    """
    Return a summary dictionary suitable for display in the Streamlit dashboard.
    """
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }

    if label_col and label_col in df.columns:
        mapped = map_labels(df[label_col])
        summary["class_distribution"] = mapped.value_counts().to_dict()
    return summary


def validate_prediction_input(df: pd.DataFrame, feature_cols: list[str]) -> tuple[bool, str]:
    """
    Check that a user-uploaded prediction CSV contains the required feature columns.

    Returns
    -------
    (True, "")          if valid
    (False, reason_str) if invalid
    """
    df_cols_lower = {c.strip().lower() for c in df.columns}
    feat_lower = [f.strip().lower() for f in feature_cols]

    missing = [f for f in feat_lower if f not in df_cols_lower]
    if missing:
        msg = (
            f"The uploaded file is missing {len(missing)} required feature column(s). "
            f"First missing: '{missing[0]}'. "
            "Please upload a CSV with the same column layout used during training."
        )
        return False, msg
    return True, ""
