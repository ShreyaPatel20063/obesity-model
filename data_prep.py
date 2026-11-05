"""Shared data loading + preprocessing definitions for the obesity-level project.

Both train_models.py and eda.py import from here, so the cleaning rules and the
feature lists live in exactly one place.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RAW_CSV = Path(__file__).parent / "data_raw" / "ObesityDataSet_raw_and_data_sinthetic.csv"
TARGET = "NObeyesdad"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Height and Weight are dropped on purpose: the target is a BMI band
# (BMI = Weight / Height^2), so keeping them lets any model "read" the label
# straight off the inputs (target leakage). We want lifestyle -> obesity.
LEAKY_COLUMNS = ["Height", "Weight"]

# Real-valued measurements (some are SMOTE-interpolated, so fractional).
NUMERIC_FEATURES = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Frequency answers that have a natural order -> ordinal encode, NOT one-hot.
ORDINAL_FEATURES = ["CAEC", "CALC"]
ORDINAL_ORDER = ["no", "Sometimes", "Frequently", "Always"]

# Unordered categories (binary or nominal) -> one-hot.
NOMINAL_FEATURES = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "MTRANS"]

FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES  # 14 columns


def load_clean() -> pd.DataFrame:
    """Read raw CSV, drop leaky columns, drop exact duplicate rows.

    Duplicates are removed BEFORE splitting; otherwise the same row could land in
    both train and test and quietly inflate the scores.
    """
    df = pd.read_csv(RAW_CSV).drop(columns=LEAKY_COLUMNS)
    return df.drop_duplicates().reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split with a fixed seed (same split for all 5 models)."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """ColumnTransformer that is fit on TRAIN data only (inside a Pipeline).

    scale_numeric=True  -> distance/gradient based models (LogReg, KNN, SVM, NB)
    scale_numeric=False -> decision tree (splits are scale-invariant)
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_ORDER] * len(ORDINAL_FEATURES),
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    # Ordinal codes (0-3) sit next to scaled numerics, so scale them too for
    # distance-based models; otherwise they'd have a different "weight".
    ordinal_step = (
        Pipeline([("encode", ordinal_encoder), ("scale", StandardScaler())])
        if scale_numeric
        else ordinal_encoder
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, NUMERIC_FEATURES),
            ("ord", ordinal_step, ORDINAL_FEATURES),
            (
                "nom",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                NOMINAL_FEATURES,
            ),
        ]
    )
