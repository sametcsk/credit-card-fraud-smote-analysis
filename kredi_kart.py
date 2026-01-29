# kredi_kart.py
# Credit Card Fraud Detection Pipeline
# Clean GitHub-renderable version (converted from notebook)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_recall_curve, auc, recall_score, precision_score
)

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

import warnings
warnings.filterwarnings("ignore")


# -------------------- DATA LOAD --------------------
df = pd.read_csv("creditcard.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())


# -------------------- CLEANING --------------------
df.drop_duplicates(inplace=True)

class_counts = df["Class"].value_counts()
class_percentages = df["Class"].value_counts(normalize=True) * 100

print("\nClass distribution:")
print(class_counts)
print("\nClass percentages:")
print(class_percentages)


# -------------------- FEATURE ENGINEERING --------------------
df["Hour"] = (df["Time"] // 3600) % 24
df.drop(columns=["Time"], inplace=True)

df["Amount_log"] = np.log1p(df["Amount"])
df.drop(columns=["Amount"], inplace=True)


# -------------------- TRAIN / TEST SPLIT --------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------- MODEL PIPELINE --------------------
model_results = {}

def run_model_scenario(pipeline, X_train, y_train, X_test, y_test, name):
    print(f"\n{'='*20} {name} {'='*20}")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1"}

    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1
    )

    print("Mean Recall:", cv_results["test_recall"].mean())
    print("Mean Precision:", cv_results["test_precision"].mean())
    print("Mean F1:", cv_results["test_f1"].mean())

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    model_results[name] = {
        "recall": recall,
        "precision": precision,
        "auc": pr_auc,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# -------------------- SCENARIOS --------------------
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_base = make_pipeline_imb(rf_base)
run_model_scenario(pipe_base, X_train_scaled, y_train, X_test_scaled, y_test, "Baseline RF")

rf_weighted = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
pipe_weighted = make_pipeline_imb(rf_weighted)
run_model_scenario(pipe_weighted, X_train_scaled, y_train, X_test_scaled, y_test, "Class Weight RF")

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_smote = make_pipeline_imb(SMOTE(random_state=42), rf_smote)
run_model_scenario(pipe_smote, X_train_scaled, y_train, X_test_scaled, y_test, "SMOTE RF")

rf_st = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_st = make_pipeline_imb(
    SMOTETomek(tomek=TomekLinks(sampling_strategy="majority"), random_state=42),
    rf_st,
)
run_model_scenario(pipe_st, X_train_scaled, y_train, X_test_scaled, y_test, "SMOTE + Tomek RF")


# -------------------- SUMMARY --------------------
summary = []
for name, metrics in model_results.items():
    y_test = metrics["y_test"]
    y_pred = metrics["y_pred"]

    summary.append([
        name,
        recall_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        metrics["auc"]
    ])

df_summary = pd.DataFrame(
    summary, columns=["Model", "Recall", "Precision", "F1", "AUC_PR"]
).sort_values(by="Recall", ascending=False)

print("\nModel Summary:")
print(df_summary)
