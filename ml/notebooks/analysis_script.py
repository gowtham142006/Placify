"""
Comprehensive analysis script for Student Placement Prediction
This script runs all analyses required for the ML mentor report.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, mutual_info_score
)
from sklearn.feature_selection import mutual_info_classif

# ============================================
# 1. DATASET QUALITY
# ============================================
print("=" * 70)
print("1. DATASET QUALITY ANALYSIS")
print("=" * 70)

DATA_PATH = "ml/data/student_placement_prediction_dataset_v2.csv"
df = pd.read_csv(DATA_PATH)

print(f"\nShape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

print(f"\nTarget distribution:")
print(df["placement_status"].value_counts())
print(f"\nTarget proportions:")
print(df["placement_status"].value_counts(normalize=True).round(4))

print(f"\nNumerical feature statistics:")
print(df.describe().round(2))

# ============================================
# 2. TARGET RELATIONSHIPS
# ============================================
print("\n" + "=" * 70)
print("2. TARGET RELATIONSHIPS")
print("=" * 70)

y_binary = (df["placement_status"] == "Placed").astype(int)
numeric_cols = ["age", "cgpa", "college_tier", "internships_count", "projects_count",
                "certifications_count", "coding_skill_score", "communication_skill_score",
                "aptitude_score", "logical_reasoning_score", "mock_interview_score", "backlogs"]

# Spearman correlation
print("\n--- Spearman Correlation with placement_status ---")
for col in numeric_cols:
    corr, pval = stats.spearmanr(df[col], y_binary)
    print(f"  {col:35s}: rho={corr:+.4f}, p={pval:.2e}")

# Mutual information
print("\n--- Mutual Information ---")
X_for_mi = df[numeric_cols]
mi_scores = mutual_info_classif(X_for_mi, y_binary, random_state=42)
for col, mi in sorted(zip(numeric_cols, mi_scores), key=lambda x: -x[1]):
    print(f"  {col:35s}: MI={mi:.6f}")

# Group placement rates for key features
print("\n--- Group Placement Rates ---")

# CGPA bins
df["cgpa_bin"] = pd.cut(df["cgpa"], bins=[0, 6, 7, 8, 10], labels=["<6", "6-7", "7-8", "8+"])
print("\nCGPA Group Placement Rates:")
print(df.groupby("cgpa_bin", observed=True)["placement_status"].apply(
    lambda x: (x == "Placed").mean()
).round(4))

# Coding skill bins
df["coding_bin"] = pd.cut(df["coding_skill_score"], bins=[0, 50, 65, 80, 100],
                          labels=["Low(<50)", "Med(50-65)", "Good(65-80)", "High(80+)"])
print("\nCoding Skill Group Placement Rates:")
print(df.groupby("coding_bin", observed=True)["placement_status"].apply(
    lambda x: (x == "Placed").mean()
).round(4))

# Mock interview bins
df["mock_bin"] = pd.cut(df["mock_interview_score"], bins=[0, 50, 65, 80, 100],
                        labels=["Low(<50)", "Med(50-65)", "Good(65-80)", "High(80+)"])
print("\nMock Interview Group Placement Rates:")
print(df.groupby("mock_bin", observed=True)["placement_status"].apply(
    lambda x: (x == "Placed").mean()
).round(4))

# Backlogs
print("\nBacklogs Group Placement Rates:")
print(df.groupby("backlogs")["placement_status"].apply(
    lambda x: (x == "Placed").mean()
).round(4))

# Categorical features
print("\n--- Categorical Feature Placement Rates ---")
for cat in ["gender", "branch", "college_tier"]:
    print(f"\n{cat}:")
    print(df.groupby(cat)["placement_status"].apply(
        lambda x: (x == "Placed").mean()
    ).round(4))

# ============================================
# 3. MODEL TRAINING AND COMPARISON
# ============================================
print("\n" + "=" * 70)
print("3. MODEL COMPARISON")
print("=" * 70)

# Prepare data
X = df.drop(columns=["placement_status", "cgpa_bin", "coding_bin", "mock_bin"])
y = df["placement_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Further split train into train_final + val
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train target dist: {y_train.value_counts(normalize=True).round(4).to_dict()}")
print(f"Val target dist:   {y_val.value_counts(normalize=True).round(4).to_dict()}")
print(f"Test target dist:  {y_test.value_counts(normalize=True).round(4).to_dict()}")

numeric_features = ["age", "cgpa", "college_tier", "internships_count", "projects_count",
                    "certifications_count", "coding_skill_score", "communication_skill_score",
                    "aptitude_score", "logical_reasoning_score", "mock_interview_score", "backlogs"]
categorical_features = ["gender", "branch"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Train models on full train set
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ))
    ])
}

# Train all models on full train
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate on test
print("\n--- Model Performance on TEST Set ---")
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_bin = (y_test == "Placed").astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Placed")
    rec = recall_score(y_test, y_pred, pos_label="Placed")
    f1 = f1_score(y_test, y_pred, pos_label="Placed")
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    roc = roc_auc_score(y_test_bin, y_prob)
    bal_acc = balanced_accuracy_score(y_test_bin, (y_pred == "Placed").astype(int))

    print(f"\n  {name}:")
    print(f"    Accuracy:          {acc:.4f}")
    print(f"    Precision(Placed): {prec:.4f}")
    print(f"    Recall(Placed):    {rec:.4f}")
    print(f"    F1(Placed):        {f1:.4f}")
    print(f"    Macro F1:          {macro_f1:.4f}")
    print(f"    ROC-AUC:           {roc:.4f}")
    print(f"    Balanced Accuracy: {bal_acc:.4f}")
    print(f"\n    Confusion Matrix (Not Placed, Placed):")
    cm = confusion_matrix(y_test, y_pred, labels=["Not Placed", "Placed"])
    print(f"    {cm}")
    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

# Evaluate on train (to check overfitting)
print("\n--- Model Performance on TRAIN Set (for overfitting check) ---")
for name, model in models.items():
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_train_bin = (y_train == "Placed").astype(int)

    acc = accuracy_score(y_train, y_pred_train)
    roc = roc_auc_score(y_train_bin, y_prob_train)
    macro_f1 = f1_score(y_train, y_pred_train, average="macro")
    bal_acc = balanced_accuracy_score(y_train_bin, (y_pred_train == "Placed").astype(int))

    print(f"\n  {name}:")
    print(f"    Train Accuracy:          {acc:.4f}")
    print(f"    Train ROC-AUC:           {roc:.4f}")
    print(f"    Train Macro F1:          {macro_f1:.4f}")
    print(f"    Train Balanced Accuracy: {bal_acc:.4f}")

# ============================================
# 4. THRESHOLD ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("4. THRESHOLD ANALYSIS (Gradient Boosting on Validation)")
print("=" * 70)

# Retrain models on train_final for proper validation threshold tuning
models_val = {}
for name, _ in models.items():
    if name == "Gradient Boosting":
        m = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
            ))
        ])
        m.fit(X_train_final, y_train_final)
        models_val[name] = m

gb_val = models_val["Gradient Boosting"]
y_val_prob = gb_val.predict_proba(X_val)[:, 1]
y_val_true = (y_val == "Placed").astype(int)

print("\n--- Comprehensive Threshold Analysis on Validation ---")
print(f"{'Thresh':>8}  {'Acc':>7}  {'Prec(P)':>8}  {'Rec(P)':>7}  {'F1(P)':>7}  "
      f"{'Prec(NP)':>8}  {'Rec(NP)':>8}  {'F1(NP)':>7}  {'MacroF1':>8}  {'BalAcc':>7}")

for threshold in np.arange(0.30, 0.76, 0.05):
    y_val_pred = np.where(y_val_prob >= threshold, "Placed", "Not Placed")

    acc = accuracy_score(y_val, y_val_pred)
    prec_p = precision_score(y_val, y_val_pred, pos_label="Placed")
    rec_p = recall_score(y_val, y_val_pred, pos_label="Placed")
    f1_p = f1_score(y_val, y_val_pred, pos_label="Placed")
    prec_np = precision_score(y_val, y_val_pred, pos_label="Not Placed")
    rec_np = recall_score(y_val, y_val_pred, pos_label="Not Placed")
    f1_np = f1_score(y_val, y_val_pred, pos_label="Not Placed")
    macro = f1_score(y_val, y_val_pred, average="macro")
    bal = balanced_accuracy_score(y_val_true, (y_val_pred == "Placed").astype(int))

    print(f"  {threshold:.2f}    {acc:.4f}   {prec_p:.4f}   {rec_p:.4f}  {f1_p:.4f}  "
          f"  {prec_np:.4f}    {rec_np:.4f}   {f1_np:.4f}    {macro:.4f}  {bal:.4f}")


# ============================================
# 5. THRESHOLD ANALYSIS ON ALL 3 MODELS
# ============================================
print("\n" + "=" * 70)
print("5. ALL MODELS - BALANCED THRESHOLD COMPARISON (on Test)")
print("=" * 70)

BALANCED_THRESHOLD = 0.60  # we'll test the best threshold we find

for name, model in models.items():
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_bin = (y_test == "Placed").astype(int)

    print(f"\n  --- {name} (Threshold=0.60) ---")

    for thr in [0.50, 0.55, 0.60, 0.65]:
        y_pred_thr = np.where(y_test_prob >= thr, "Placed", "Not Placed")
        acc = accuracy_score(y_test, y_pred_thr)
        macro = f1_score(y_test, y_pred_thr, average="macro")
        bal = balanced_accuracy_score(y_test_bin, (y_pred_thr == "Placed").astype(int))
        roc = roc_auc_score(y_test_bin, y_test_prob)

        print(f"    Threshold {thr}: Acc={acc:.4f}, MacroF1={macro:.4f}, "
              f"BalAcc={bal:.4f}, ROC-AUC={roc:.4f}")

# ============================================
# 6. FEATURE IMPORTANCE
# ============================================
print("\n" + "=" * 70)
print("6. FEATURE IMPORTANCE / INTERPRETABILITY")
print("=" * 70)

# Gradient Boosting
gb_model = models["Gradient Boosting"].named_steps["model"]
feature_names = (numeric_features +
                 list(models["Gradient Boosting"].named_steps["preprocessor"]
                      .named_transformers_["categorical"]
                      .named_steps["onehot"]
                      .get_feature_names_out(categorical_features)))
gb_importances = gb_model.feature_importances_

print("\n--- Gradient Boosting Feature Importances ---")
for name_f, imp in sorted(zip(feature_names, gb_importances), key=lambda x: -x[1]):
    print(f"  {name_f:35s}: {imp:.4f}")

# Random Forest
rf_model = models["Random Forest"].named_steps["model"]
rf_importances = rf_model.feature_importances_

print("\n--- Random Forest Feature Importances ---")
for name_f, imp in sorted(zip(feature_names, rf_importances), key=lambda x: -x[1]):
    print(f"  {name_f:35s}: {imp:.4f}")

# Logistic Regression coefficients
lr_model = models["Logistic Regression"].named_steps["model"]
lr_coefs = lr_model.coef_[0]

print("\n--- Logistic Regression Coefficients ---")
for name_f, coef in sorted(zip(feature_names, lr_coefs), key=lambda x: -abs(x[1])):
    print(f"  {name_f:35s}: {coef:+.4f}")

# ============================================
# 7. DUMMY BASELINE
# ============================================
print("\n" + "=" * 70)
print("7. DUMMY BASELINE")
print("=" * 70)

from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
print(f"  Majority class accuracy: {accuracy_score(y_test, y_dummy):.4f}")
print(f"  Majority class Macro F1: {f1_score(y_test, y_dummy, average='macro'):.4f}")

dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
dummy_strat.fit(X_train, y_train)
y_dummy_strat = dummy_strat.predict(X_test)
print(f"  Stratified accuracy:    {accuracy_score(y_test, y_dummy_strat):.4f}")
print(f"  Stratified Macro F1:    {f1_score(y_test, y_dummy_strat, average='macro'):.4f}")

print("\n\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
