"""Train the 5 classifiers on the obesity dataset. Run: python train_models.py

Outputs
  model/<name>.joblib          fitted Pipeline (preprocessor + classifier)
  model/meta.joblib            feature list, class labels, target name
  test_data.csv                held-out 20 % split (with target) for the Streamlit uploader
  results/model_comparison.csv the 6 metrics for each model (test set)
  results/overfit_check.csv    train vs test accuracy per model
  results/scaling_ablation.csv KNN / SVM with and without scaling
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data_prep as dp

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = dp.RANDOM_STATE


def model_zoo() -> dict:
    """name -> (display name, unfitted Pipeline). Same seed everywhere."""
    return {
        "logistic_regression": (
            "Logistic Regression",
            Pipeline([
                ("prep", dp.build_preprocessor(scale_numeric=True)),
                ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]),
        ),
        # Trees split on thresholds, so scaling is unnecessary -> encode only.
        "decision_tree": (
            "Decision Tree",
            Pipeline([
                ("prep", dp.build_preprocessor(scale_numeric=False)),
                ("clf", DecisionTreeClassifier(random_state=SEED)),
            ]),
        ),
        "knn": (
            "K-Nearest Neighbors",
            Pipeline([
                ("prep", dp.build_preprocessor(scale_numeric=True)),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]),
        ),
        # GaussianNB: features are mostly continuous (and scaled values are
        # negative, which MultinomialNB cannot accept).
        "naive_bayes": (
            "Naive Bayes",
            Pipeline([
                ("prep", dp.build_preprocessor(scale_numeric=True)),
                ("clf", GaussianNB()),
            ]),
        ),
        # probability=True is required for predict_proba -> AUC.
        "svm": (
            "Support Vector Machine",
            Pipeline([
                ("prep", dp.build_preprocessor(scale_numeric=True)),
                ("clf", SVC(kernel="rbf", probability=True, random_state=SEED)),
            ]),
        ),
    }


def score(pipe: Pipeline, X, y) -> dict:
    """The 6 required metrics. Multi-class: macro precision/recall/F1, OvR macro AUC."""
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=pipe.classes_),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def main() -> None:
    df = dp.load_clean()
    train_df, test_df = dp.split_data(df)
    X_train, y_train = train_df[dp.FEATURES], train_df[dp.TARGET]
    X_test, y_test = test_df[dp.FEATURES], test_df[dp.TARGET]
    print(f"train={len(train_df)}  test={len(test_df)}  features={len(dp.FEATURES)}")

    test_df.to_csv(ROOT / "test_data.csv", index=False)

    rows, overfit = [], []
    for key, (display, pipe) in model_zoo().items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / f"{key}.joblib")
        rows.append({"ML Model Name": display, **score(pipe, X_test, y_test)})
        overfit.append({
            "ML Model Name": display,
            "Train Accuracy": accuracy_score(y_train, pipe.predict(X_train)),
            "Test Accuracy": accuracy_score(y_test, pipe.predict(X_test)),
        })
        print(f"fitted {display}")

    joblib.dump(
        {
            "features": dp.FEATURES,
            "target": dp.TARGET,
            "classes": list(pipe.classes_),
            "models": {k: v[0] for k, v in model_zoo().items()},
        },
        MODEL_DIR / "meta.joblib",
    )

    comparison = pd.DataFrame(rows).round(4)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    og = pd.DataFrame(overfit)
    og["Gap"] = og["Train Accuracy"] - og["Test Accuracy"]
    og.round(4).to_csv(RESULTS_DIR / "overfit_check.csv", index=False)

    # Scaling ablation: same KNN / SVM, numeric scaling switched off.
    ablation = []
    for key in ("knn", "svm"):
        display, scaled = model_zoo()[key]
        # clone(): a fresh classifier, otherwise both pipelines would share (and overwrite) one fitted estimator
        unscaled = Pipeline([
            ("prep", dp.build_preprocessor(scale_numeric=False)),
            ("clf", clone(scaled.named_steps["clf"])),
        ])
        unscaled.fit(X_train, y_train)
        scaled.fit(X_train, y_train)
        ablation.append({
            "ML Model Name": display,
            "Accuracy (scaled)": accuracy_score(y_test, scaled.predict(X_test)),
            "Accuracy (unscaled)": accuracy_score(y_test, unscaled.predict(X_test)),
        })
    pd.DataFrame(ablation).round(4).to_csv(RESULTS_DIR / "scaling_ablation.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nTEST-SET METRICS\n", comparison.to_string(index=False))
    print("\nOVERFIT CHECK\n", og.round(4).to_string(index=False))
    print("\nSCALING ABLATION\n", pd.DataFrame(ablation).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
