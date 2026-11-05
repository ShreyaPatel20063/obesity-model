"""Streamlit app: evaluate saved obesity-level classifiers on an uploaded test CSV.

The app only LOADS the fitted pipelines from model/ - nothing is trained here.
Run: streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"

# Feature groups (mirror data_prep.py; kept here so the app has no training deps).
NUMERIC_COLS = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

st.set_page_config(page_title="Obesity Level Classifier Lab", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading saved models...")
def load_artifacts():
    meta = joblib.load(MODEL_DIR / "meta.joblib")
    models = {display: joblib.load(MODEL_DIR / f"{key}.joblib") for key, display in meta["models"].items()}
    return meta, models


@st.cache_data
def load_results():
    return {
        "comparison": pd.read_csv(RESULTS_DIR / "model_comparison.csv"),
        "overfit": pd.read_csv(RESULTS_DIR / "overfit_check.csv"),
        "scaling": pd.read_csv(RESULTS_DIR / "scaling_ablation.csv"),
    }


# ------------------------------------------------------------------------ validation
def validate_upload(df: pd.DataFrame, meta: dict) -> list[str]:
    """Return a list of human-readable problems; empty list means the file is usable."""
    problems = []
    if df.empty:
        return ["The file has no rows."]

    missing = [c for c in meta["features"] if c not in df.columns]
    if missing:
        problems.append(f"Missing feature column(s): **{', '.join(missing)}**.")
    if meta["target"] not in df.columns:
        problems.append(
            f"Target column **{meta['target']}** is missing. It is needed to compute the metrics."
        )
    if problems:
        return problems  # no point checking values if columns are absent

    bad_numeric = [c for c in NUMERIC_COLS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_numeric:
        problems.append(f"Column(s) that must be numeric contain text: **{', '.join(bad_numeric)}**.")

    nulls = df[meta["features"] + [meta["target"]]].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        problems.append("Empty cells found in: " + ", ".join(f"**{c}** ({n})" for c, n in nulls.items()) + ".")

    unknown = sorted(set(df[meta["target"]].dropna().unique()) - set(meta["classes"]))
    if unknown:
        problems.append(
            f"Unknown target label(s): {', '.join(map(str, unknown))}. "
            f"Expected only: {', '.join(meta['classes'])}."
        )
    return problems


# ------------------------------------------------------------------------- evaluation
def compute_metrics(pipe, X, y, classes):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": None,
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    # One-vs-rest AUC needs every class to appear in the uploaded labels.
    if set(y.unique()) == set(classes):
        metrics["AUC"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(classes))
    return pred, metrics


def plot_confusion(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------ UI
meta, models = load_artifacts()
results = load_results()

st.title("🩺 Obesity Level Classifier Lab")
st.caption(
    "Five classical ML models trained on lifestyle and eating-habit data (UCI Obesity Levels). "
    "Upload held-out test data and see how each saved model performs."
)

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Choose a model", list(models.keys()), index=len(models) - 1)
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    st.download_button(
        "Download sample test_data.csv",
        data=(ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    st.markdown("---")
    st.markdown(
        f"**Required columns ({len(meta['features'])} features + target)**\n\n"
        + ", ".join(f"`{c}`" for c in meta["features"])
        + f"\n\nTarget: `{meta['target']}`"
    )

tab_eval, tab_compare, tab_data = st.tabs(["📊 Evaluate a model", "🏆 Model comparison", "📚 About the data"])

# ----------------------------------------------------------------------- tab: evaluate
with tab_eval:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar to begin. Use the sample `test_data.csv` if you don't have one.")
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:  # unreadable / not a CSV
            st.error(f"Could not read that file as a CSV: {exc}")
            st.stop()

        problems = validate_upload(df, meta)
        if problems:
            st.error("This file can't be evaluated yet:")
            for p in problems:
                st.markdown(f"- {p}")
            st.stop()

        X, y = df[meta["features"]], df[meta["target"]]
        pipe = models[model_name]
        pred, metrics = compute_metrics(pipe, X, y, meta["classes"])

        st.subheader(f"{model_name} on {len(df)} uploaded rows")
        # 2 rows x 3 so values never get truncated on narrow screens
        for row in (list(metrics.items())[:3], list(metrics.items())[3:]):
            for col, (name, value) in zip(st.columns(3), row):
                col.metric(name, "n/a" if value is None else f"{value:.4f}")
        if metrics["AUC"] is None:
            st.caption("AUC is n/a: the uploaded file does not contain all 7 classes, so one-vs-rest AUC is undefined.")
        st.caption("Precision, Recall and F1 are macro-averaged across the 7 classes; AUC is one-vs-rest macro.")

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("#### Confusion matrix")
            fig = plot_confusion(y, pred, meta["classes"])
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.markdown("#### Classification report")
            report = classification_report(y, pred, labels=meta["classes"], output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

        out = df.copy()
        out["Predicted"] = pred
        st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv", "text/csv")

# ----------------------------------------------------------------------- tab: compare
with tab_compare:
    st.subheader("All five models on the held-out test split")
    comp = results["comparison"]
    st.dataframe(
        comp.style.format({c: "{:.4f}" for c in comp.columns[1:]}).highlight_max(subset=comp.columns[1:], color="#c8e6c9"),
        width="stretch", hide_index=True,
    )
    metric = st.radio("Chart metric", list(comp.columns[1:]), horizontal=True, index=0)
    st.bar_chart(comp.set_index("ML Model Name")[metric])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Overfitting check (train vs test accuracy)")
        st.dataframe(results["overfit"], width="stretch", hide_index=True)
    with c2:
        st.markdown("#### Does scaling matter? (accuracy)")
        st.dataframe(results["scaling"], width="stretch", hide_index=True)

# -------------------------------------------------------------------------- tab: data
with tab_data:
    st.subheader("Dataset: Estimation of Obesity Levels (UCI)")
    st.markdown(
        """
- **Task:** predict one of 7 obesity levels from eating habits, physical condition and demographics.
- **Size after cleaning:** 2,082 rows and 14 features (80/20 stratified split, `random_state=42`).
- **Height and Weight were removed on purpose.** The label is a BMI band (BMI = Weight / Height²),
  so keeping them leaks the answer. A plain decision tree scored 0.93 with them and 0.75 without.
- **Synthetic data:** about 77 % of the original rows were generated with SMOTE, so some answers
  such as meals per day are fractional.
- **Classes:** """ + ", ".join(f"`{c}`" for c in meta["classes"]) + "."
    )
    with st.expander("Preview of the sample test file"):
        st.dataframe(pd.read_csv(ROOT / "test_data.csv").head(20), width="stretch")
