"""Exploratory analysis for the obesity dataset. Run: python eda.py

Writes figures to eda_plots/ and prints the numbers quoted in the README.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

import data_prep as dp

OUT = Path(__file__).parent / "eda_plots"
OUT.mkdir(exist_ok=True)

# Natural order of the target, for readable plots.
CLASS_ORDER = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]

raw = pd.read_csv(dp.RAW_CSV)
df = dp.load_clean()
print(f"raw rows={len(raw)}, exact duplicates in raw={raw.duplicated().sum()}")
print(f"after dropping Height/Weight + dedup: rows={len(df)}, features={len(dp.FEATURES)}")
print("nulls:", int(df.isna().sum().sum()))

# 1. Class balance -----------------------------------------------------------
counts = df[dp.TARGET].value_counts().reindex(CLASS_ORDER)
print("\nclass counts:\n", counts.to_string())
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=counts.values, y=counts.index, color="#4c72b0", ax=ax)
ax.set_title("Target class distribution (after cleaning)")
ax.set_xlabel("rows")
fig.tight_layout()
fig.savefig(OUT / "class_distribution.png", dpi=120)
plt.close(fig)

# 2. Numeric features by class -------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.ravel(), dp.NUMERIC_FEATURES):
    sns.boxplot(data=df, x=dp.TARGET, y=col, order=CLASS_ORDER, ax=ax, color="#8fb8de")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=60)
fig.suptitle("Numeric features by obesity class")
fig.tight_layout()
fig.savefig(OUT / "numeric_by_class.png", dpi=120)
plt.close(fig)

# 3. Correlation among numeric/ordinal features -------------------------------
ordinal_codes = df[dp.ORDINAL_FEATURES].apply(lambda s: s.map({v: i for i, v in enumerate(dp.ORDINAL_ORDER)}))
corr = pd.concat([df[dp.NUMERIC_FEATURES], ordinal_codes], axis=1).corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation (numeric + ordinal features)")
fig.tight_layout()
fig.savefig(OUT / "correlation.png", dpi=120)
plt.close(fig)

# 4. How "synthetic" are the rows? -----------------------------------------------
print("\nshare of fractional values (SMOTE fingerprint):")
for col in ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]:
    print(f"  {col}: {(df[col] % 1 != 0).mean():.0%}")

# 5. Leakage proof: same tree, with vs without Height/Weight --------------------
full = raw.drop_duplicates().reset_index(drop=True)
X_leaky = full.drop(columns=[dp.TARGET])
X_leaky = pd.get_dummies(X_leaky, drop_first=True)
X_clean = pd.get_dummies(full.drop(columns=[dp.TARGET] + dp.LEAKY_COLUMNS), drop_first=True)
tree = DecisionTreeClassifier(random_state=dp.RANDOM_STATE)
leaky = cross_val_score(tree, X_leaky, full[dp.TARGET], cv=5).mean()
clean = cross_val_score(tree, X_clean, full[dp.TARGET], cv=5).mean()
print(f"\nleakage check, decision tree 5-fold CV accuracy:")
print(f"  with Height+Weight   : {leaky:.3f}")
print(f"  without Height+Weight: {clean:.3f}")
