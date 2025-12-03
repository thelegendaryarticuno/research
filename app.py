import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import lightgbm as lgb
warnings.filterwarnings("ignore")


# =========================================================
# Helpers
# =========================================================
def prepare_xy(df: pd.DataFrame):
    """Return (X, y), dropping obvious non-feature columns and using only numeric features."""
    drop_cols = [c for c in ["algorithm", "infile", "outfile"] if c in df.columns]
    y = df["algorithm"].copy()
    X = df.drop(columns=drop_cols)
    # keep only numeric columns
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y

def class_table_from_report(report_dict, label_order):
    """Extract per-class precision/recall/f1 for the given label order (algorithms)."""
    rows = []
    for lab in label_order:
        if lab in report_dict:
            rows.append((
                lab,
                report_dict[lab].get("precision", 0.0),
                report_dict[lab].get("recall", 0.0),
                report_dict[lab].get("f1-score", 0.0),
            ))
        else:
            rows.append((lab, 0.0, 0.0, 0.0))
    out = pd.DataFrame(rows, columns=["algorithm", "precision", "recall", "f1"])
    return out

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


# =========================================================
# Load data
# =========================================================
df = pd.read_csv("metrics/perf_metrics.csv")
if "file_kb" not in df.columns:
    raise RuntimeError("Expected a 'file_kb' column in metrics/perf_metrics.csv")

# Keep a consistent order for algorithms in tables
alg_order = sorted(df["algorithm"].unique())


# =========================================================
# Define models  (SVM & KNN are scaled; SVM lightly tuned)
# =========================================================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ]),
    # Lightly tune SVM RBF for better results
    "SVM": GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight=None, random_state=42))
        ]),
        param_grid={
            "clf__C": [1.0, 5.0, 10.0, 50.0, 100.0],
            "clf__gamma": ["scale", 0.1, 0.01, 0.001],
        },
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        refit=True,
        verbose=0
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
}

# =========================================================
# GLOBAL (all sizes mixed) results
# =========================================================
X, y = prepare_xy(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

overall_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    overall_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    })

overall_df = pd.DataFrame(overall_rows).sort_values("F1", ascending=False)
print("\n===== Overall (all sizes) =====")
print(overall_df.to_string(index=False))

ensure_dir("metrics")
overall_df.to_csv("metrics/model_overall_results.csv", index=False)


# =========================================================
# PER-SIZE tables like the paper (one table per metric)
# Rows = algorithms (cipher), Columns = models
# =========================================================
tables_dir = "metrics/tables_by_size"
ensure_dir(tables_dir)

# unique sizes
sizes = sorted(df["file_kb"].unique())
for sk in sizes:
    block = df[df["file_kb"] == sk].copy()
    # need at least two samples per class for stratify; if not, skip
    if block["algorithm"].value_counts().min() < 2:
        print(f"[WARN] Skipping size {sk}KB: not enough samples per class for stratified split.")
        continue

    Xb, yb = prepare_xy(block)
    Xtr, Xte, ytr, yte = train_test_split(
        Xb, yb, test_size=0.3, stratify=yb, random_state=42
    )

    # metric matrices we will fill and then pivot into “algorithm x model”
    f1_rows, prec_rows, rec_rows = [], [], []
    acc_rows = []  # accuracy is per model (single number for the size)

    for name, model in models.items():
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        # accuracy (per model)
        acc_rows.append({"Model": name, "Accuracy": accuracy_score(yte, yhat)})

        # per-class metrics
        rep = classification_report(yte, yhat, output_dict=True, zero_division=0)
        per_class = class_table_from_report(rep, alg_order)  # precision/recall/f1 per algorithm

        # store rows tagged with model name for later pivot
        tmp = per_class.copy()
        tmp["Model"] = name
        f1_rows.append(tmp[["algorithm", "Model", "f1"]])
        prec_rows.append(tmp[["algorithm", "Model", "precision"]])
        rec_rows.append(tmp[["algorithm", "Model", "recall"]])

    # build tables (rows: algorithm, cols: model)
    f1_tab   = pd.concat(f1_rows).pivot(index="algorithm", columns="Model", values="f1").loc[alg_order]
    prec_tab = pd.concat(prec_rows).pivot(index="algorithm", columns="Model", values="precision").loc[alg_order]
    rec_tab  = pd.concat(rec_rows).pivot(index="algorithm", columns="Model", values="recall").loc[alg_order]
    acc_tab  = pd.DataFrame(acc_rows).set_index("Model").T  # one-row accuracy table

    print(f"\n===== {int(sk)} KB — F1 (rows=ciphers, cols=models) =====")
    print(f1_tab.round(3).to_string())
    print(f"\n===== {int(sk)} KB — Precision =====")
    print(prec_tab.round(3).to_string())
    print(f"\n===== {int(sk)} KB — Recall =====")
    print(rec_tab.round(3).to_string())
    print(f"\n===== {int(sk)} KB — Accuracy (per model) =====")
    print(acc_tab.round(3).to_string())

    # save CSVs
    f1_tab.to_csv(f"{tables_dir}/F1_{int(sk)}KB.csv")
    prec_tab.to_csv(f"{tables_dir}/Precision_{int(sk)}KB.csv")
    rec_tab.to_csv(f"{tables_dir}/Recall_{int(sk)}KB.csv")
    acc_tab.to_csv(f"{tables_dir}/Accuracy_{int(sk)}KB.csv")

print(f"\nSaved overall results to metrics/model_overall_results.csv")
print(f"Per-size tables saved to {tables_dir}/ (F1_*.csv, Precision_*.csv, Recall_*.csv, Accuracy_*.csv)")
