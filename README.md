# Obesity Level Prediction from Lifestyle Data

Machine Learning Assignment 2, BITS Pilani WILP M.Tech (AIML). Five classifiers, one dataset, one Streamlit app.

## a. Problem statement

Predict a person's **obesity level (7 classes)** from eating habits, physical activity, transport and basic
demographics, **without** using height or weight. The goal is to compare five classical classifiers
(Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on the same data, the same train/test split and the
same random seed, and to serve the trained models through an interactive Streamlit app that evaluates an uploaded
test CSV.

Height and Weight were removed on purpose. The label is a BMI band (BMI = Weight / Height²), so with those two
columns the answer is almost handed to the model. Evidence from `eda.py` (plain decision tree, 5-fold CV):
**0.928 accuracy with Height and Weight, 0.748 without.** Dropping them turns the task into a real
lifestyle-to-obesity prediction problem.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, *Estimation of obesity levels based on eating habits and physical condition* (Palechor and De la Hoz Manotas, 2019) |
| Raw size | 2,111 rows, 16 features + target |
| After cleaning | **2,082 rows, 14 features** (removed Height, Weight and exact duplicate rows) |
| Target | `NObeyesdad`, 7 classes |
| Train / test | 1,665 / 417, stratified 80/20, `random_state=42` |
| Missing values | None |

**Features (14)**
- Numeric (6): `Age`, `FCVC` (vegetable frequency), `NCP` (main meals per day), `CH2O` (daily water), `FAF` (physical activity frequency), `TUE` (screen time).
- Ordinal (2): `CAEC` (eating between meals), `CALC` (alcohol). Order: no < Sometimes < Frequently < Always.
- Nominal (6): `Gender`, `family_history_with_overweight`, `FAVC` (high-calorie food), `SMOKE`, `SCC` (calorie monitoring), `MTRANS` (transport).

**Class distribution (after cleaning)**

| Class | Rows | Class | Rows |
|---|---|---|---|
| Insufficient_Weight | 265 | Obesity_Type_I | 351 |
| Normal_Weight | 279 | Obesity_Type_II | 297 |
| Overweight_Level_I | 276 | Obesity_Type_III | 324 |
| Overweight_Level_II | 290 | | |

The classes are close to balanced (12.7% to 16.9%), so macro-averaging is a fair choice.

**Caveat:** about 77% of the original rows were generated with SMOTE, so some answers (for example 2.45 meals
per day) are fractional. Scores on this dataset may not carry over to real surveys.

**Preprocessing (fit on training data only, inside each sklearn `Pipeline`)**
- Numeric and ordinal features are standardised for Logistic Regression, KNN, Naive Bayes and SVM. The Decision Tree gets encoding only, since trees are scale-invariant.
- Ordinal features use an explicit order. Nominal features are one-hot encoded with `handle_unknown="ignore"`.
- Duplicates are removed *before* the split so the same row cannot appear in both train and test.

**Metrics.** Multi-class setting: Precision, Recall and F1 are **macro-averaged**; AUC is **one-vs-rest, macro-averaged**
(from `predict_proba`); MCC is the multi-class MCC.

## c. GitHub repository link

`<GITHUB_REPO_LINK_HERE>` (placeholder, to be filled after pushing)

Live app: https://obesity-model.streamlit.app/

## d. Models used

All five models use the same split and `random_state=42`. Test-set results (417 rows), from `results/model_comparison.csv`:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6139 | 0.8817 | 0.6198 | 0.6019 | 0.5782 | 0.5572 |
| Decision Tree | 0.7386 | 0.8458 | 0.7349 | 0.7352 | 0.7335 | 0.6949 |
| K-Nearest Neighbors | 0.7578 | 0.9264 | 0.7499 | 0.7519 | 0.7347 | 0.7212 |
| Naive Bayes (Gaussian) | 0.4221 | 0.8116 | 0.5903 | 0.4119 | 0.3437 | 0.3675 |
| Support Vector Machine (RBF) | **0.7794** | **0.9560** | **0.7712** | **0.7768** | **0.7707** | **0.7434** |

Settings: Logistic Regression `max_iter=2000`; Decision Tree unconstrained (default); KNN `k=5`; GaussianNB defaults;
SVC `kernel="rbf"`, `probability=True`.

**Why Gaussian Naive Bayes:** features are mostly continuous, and after standardisation values go negative,
which MultinomialNB cannot accept.

Supporting experiments (`results/overfit_check.csv`, `results/scaling_ablation.csv`):

| Model | Train acc | Test acc | | Model | Acc scaled | Acc unscaled |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5958 | 0.6139 | | K-Nearest Neighbors | 0.7578 | 0.7410 |
| Decision Tree | 1.0000 | 0.7386 | | Support Vector Machine | 0.7794 | 0.4532 |
| K-Nearest Neighbors | 0.8210 | 0.7578 | | | | |
| Naive Bayes | 0.4306 | 0.4221 | | | | |
| Support Vector Machine | 0.8420 | 0.7794 | | | | |

### Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Weakest of the "reasonable" models (accuracy 0.614). Train and test accuracy are both about 0.60, so it **underfits** rather than overfits. A linear boundary cannot separate neighbouring BMI bands from lifestyle answers: recall is only 0.36 on Overweight_Level_I and 0.16 on Overweight_Level_II, while Obesity_Type_III is perfect (1.00). |
| Decision Tree | **Overfits clearly.** Train accuracy is 1.00 and test accuracy 0.739, a 26-point gap (depth 17, 356 leaves). It still beats Logistic Regression because it captures non-linear interactions, but its AUC (0.846) is the lowest among the four stronger models, since leaf probabilities are coarse. Limiting depth or pruning would likely help (not tried). |
| K-Nearest Neighbors | Second best (accuracy 0.758, AUC 0.926). Gap between train (0.821) and test (0.758) is moderate. Scaling helps but only modestly here (0.741 unscaled vs 0.758 scaled), probably because Age was the only feature with a large raw range and the rest are small integers. |
| Naive Bayes | **Worst model** (accuracy 0.422, MCC 0.368). It recalls only 2% and 3% of the two Overweight classes and predicts Obesity_Type_II for 186 of 417 rows. Likely causes: the independence assumption is violated by related lifestyle features, one-hot and ordinal columns are not Gaussian, and the middle BMI classes overlap so heavily that the wide-variance classes absorb their points. Train and test accuracy are both low (0.43 / 0.42), so this is high bias, not overfitting. Its AUC (0.812) is much better than its accuracy suggests, meaning the probability *ranking* is useful even though the argmax decision is poor. |
| Support Vector Machine | **Best on all six metrics.** RBF kernel models non-linear boundaries smoothly, with a smaller train/test gap (0.842 / 0.779) than the tree. **Scaling matters most here:** accuracy falls from 0.779 to 0.453 without scaling, because the RBF kernel is distance-based and unscaled Age dominates. Still weak on the middle classes (recall about 0.60 to 0.67 for Normal_Weight and both Overweight levels), which is expected without height and weight. |
| **Overall winner for this dataset?** | **Support Vector Machine.** It leads on Accuracy (0.779), AUC (0.956), Precision, Recall, F1 and MCC (0.743). KNN is a close second, and it is much cheaper to train. Because only one split was used, the gap between SVM and KNN (about 2 accuracy points on 417 rows) should be treated as suggestive, not conclusive. |

## Repository layout

```
app.py                 Streamlit app (loads saved models, never retrains)
train_models.py        trains the 5 models, writes model/, results/ and test_data.csv
data_prep.py           cleaning, split and preprocessing shared by all scripts
eda.py                 exploratory plots + leakage check -> eda_plots/
requirements.txt       pinned versions (match the training environment)
test_data.csv          held-out 20% split with the target column, for the app uploader
model/                 5 fitted pipelines + meta.joblib (features, classes)
results/               comparison, overfit and scaling tables (CSV)
data_raw/              original UCI CSV
```

## Run locally

```bash
pip install -r requirements.txt
python train_models.py     # optional: regenerates model/, results/ and test_data.csv
streamlit run app.py
```

The saved models were trained with scikit-learn 1.9.1; keep the pinned versions so the `.joblib` files load correctly.

## App features

- Upload a test CSV (the sample `test_data.csv` can be downloaded from the sidebar).
- Choose any of the 5 models from a dropdown.
- See Accuracy, AUC, Precision, Recall, F1 and MCC, plus the confusion matrix and classification report.
- Friendly validation errors: missing feature or target columns, text in numeric columns, empty cells, unknown labels.
- **Model comparison** tab with the results table, a metric chart, the overfitting check and the scaling experiment.
