"""
Exercise 8 — Supervised ML pipeline pentru expresie genică (Random Forest)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Config
# --------------------------
HANDLE = "ioanamhl"
DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONFUSION = OUT_DIR / f"confusion_rf_{HANDLE}.png"
OUT_REPORT = OUT_DIR / f"classification_report_{HANDLE}.txt"
OUT_FEATIMP = OUT_DIR / f"feature_importance_{HANDLE}.csv"
OUT_CLUSTER_CROSSTAB = OUT_DIR / f"cluster_crosstab_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Nu am găsit fișierul: {path}")


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    return X, y


def make_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Face X 100% numeric.
    - elimină coloane nenumerice (ex. coloană cu nume de gene gen 'ALDH4A1')
    - convertește restul la float
    """
    # încearcă să convertești tot (ce nu se poate devine NaN)
    X_num = X.apply(pd.to_numeric, errors="coerce")

    # elimină coloanele complet NaN (adică erau text pur)
    all_nan_cols = X_num.columns[X_num.isna().all()].tolist()
    if all_nan_cols:
        print(f"[WARN] Dropping non-numeric columns (all NaN after to_numeric): {all_nan_cols[:10]}"
              + (" ..." if len(all_nan_cols) > 10 else ""))

    X_num = X_num.drop(columns=all_nan_cols)

    # umple NaN-urile rămase (dacă au fost amestec numeric+text) cu mediană
    if X_num.isna().any().any():
        X_num = X_num.fillna(X_num.median(numeric_only=True))

    # dacă mai există NaN (coloane constant NaN/mediană), umple cu 0
    X_num = X_num.fillna(0.0)

    return X_num


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_png: Path,
    out_txt: Path,
) -> None:
    y_pred = model.predict(X_test)

    target_names = [str(cls) for cls in label_encoder.classes_]
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=target_names,
        zero_division=0,
    )
    print(report)
    out_txt.write_text(report, encoding="utf-8")

    cm = confusion_matrix(
        y_test, y_pred, labels=np.arange(len(label_encoder.classes_))
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest — confusion matrix")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def compute_feature_importance(
    model: RandomForestClassifier,
    feature_names: pd.Index,
    out_csv: Path,
) -> pd.DataFrame:
    importances = model.feature_importances_
    df_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    df_imp.to_csv(out_csv, index=False, encoding="utf-8")
    return df_imp


def run_kmeans_and_crosstab(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    n_clusters: int,
    out_csv: Path,
) -> None:
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    clusters = kmeans.fit_predict(X.values)

    y_str = label_encoder.inverse_transform(y)

    df_compare = pd.DataFrame({"True_Label": y_str, "KMeans_Cluster": clusters})
    ctab = pd.crosstab(df_compare["True_Label"], df_compare["KMeans_Cluster"])
    ctab.to_csv(out_csv, encoding="utf-8")
    print("Crosstab True_Label vs KMeans_Cluster:")
    print(ctab)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(DATA_CSV)

    X, y = load_dataset(DATA_CSV)

    # IMPORTANT: fă X numeric ca să nu mai ai strings gen 'ALDH4A1'
    X = make_features_numeric(X)

    y_enc, le = encode_labels(y)

    # distribuție clase
    counts = pd.Series(y_enc).value_counts()
    min_count = int(counts.min()) if len(counts) else 0
    print("[INFO] Class counts:\n", counts.sort_index().to_string())
    print("[INFO] Min class count:", min_count)

    # split robust: fără stratify dacă există clase cu 1 exemplu
    stratify_arg = y_enc if min_count >= 2 else None
    if stratify_arg is None:
        print("[WARN] At least one class has <2 samples. train_test_split will run WITHOUT stratify.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_arg,
    )

    # dacă train are o singură clasă, reîncearcă
    if len(np.unique(y_train)) < 2:
        print("[WARN] Train set ended up with <2 classes. Retrying split without stratify and different random_state.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_enc,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE + 1,
            stratify=None,
        )

    rf = train_random_forest(X_train, y_train, N_ESTIMATORS, RANDOM_STATE)
    evaluate_model(rf, X_test, y_test, le, OUT_CONFUSION, OUT_REPORT)

    _ = compute_feature_importance(rf, X.columns, OUT_FEATIMP)

    n_classes = len(le.classes_)
    if n_classes >= 2:
        run_kmeans_and_crosstab(X, y_enc, le, n_clusters=n_classes, out_csv=OUT_CLUSTER_CROSSTAB)
    else:
        print("[WARN] Only 1 class in dataset -> skipping KMeans.")

    print("[INFO] Done. Outputs saved in:", OUT_DIR)
