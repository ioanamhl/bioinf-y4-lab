"""
Exercise 8b — Logistic Regression vs Random Forest pe expresie genică
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Config
# --------------------------
HANDLE = "ioanamhl"

DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_ITER_LOGREG = 1000

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT_TXT = OUT_DIR / f"rf_vs_logreg_report_{HANDLE}.txt"


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
    Asigură că X este numeric (float).
    - convertește tot la numeric (errors='coerce')
    - elimină coloane care devin complet NaN (erau text pur, ex 'ALDH4A1')
    - umple NaN-uri rămase cu mediană, apoi cu 0
    """
    X_num = X.apply(pd.to_numeric, errors="coerce")

    all_nan_cols = X_num.columns[X_num.isna().all()].tolist()
    if all_nan_cols:
        print(
            f"[WARN] Dropping non-numeric columns (all NaN after to_numeric): {all_nan_cols[:10]}"
            + (" ..." if len(all_nan_cols) > 10 else "")
        )
        X_num = X_num.drop(columns=all_nan_cols)

    if X_num.shape[1] == 0:
        raise ValueError(
            "Nu a rămas nicio coloană numerică în X după conversie. "
            "Verifică formatul CSV-ului (poate genele sunt pe rânduri, nu pe coloane)."
        )

    if X_num.isna().any().any():
        X_num = X_num.fillna(X_num.median(numeric_only=True))

    X_num = X_num.fillna(0.0)

    return X_num


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    logreg = LogisticRegression(
        multi_class="multinomial",
        max_iter=MAX_ITER_LOGREG,
        n_jobs=-1,
        solver="lbfgs",
    )
    logreg.fit(X_train_scaled, y_train)

    return rf, logreg, scaler


def compare_models(
    rf: RandomForestClassifier,
    logreg: LogisticRegression,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_txt: Path,
) -> None:
    X_test_scaled = scaler.transform(X_test.values)

    y_pred_rf = rf.predict(X_test)
    y_pred_logreg = logreg.predict(X_test_scaled)

    target_names = [str(c) for c in label_encoder.classes_]
    all_labels = np.arange(len(label_encoder.classes_))

    report_rf = classification_report(
        y_test,
        y_pred_rf,
        labels=all_labels,
        target_names=target_names,
        zero_division=0,
    )
    report_logreg = classification_report(
        y_test,
        y_pred_logreg,
        labels=all_labels,
        target_names=target_names,
        zero_division=0,
    )

    print("=== Random Forest ===")
    print(report_rf)
    print("\n=== Logistic Regression ===")
    print(report_logreg)

    combined = (
        "=== Random Forest ===\n"
        + report_rf
        + "\n\n=== Logistic Regression ===\n"
        + report_logreg
        + "\n"
    )
    out_txt.write_text(combined, encoding="utf-8")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ensure_exists(DATA_CSV)

    X, y = load_dataset(DATA_CSV)

    # IMPORTANT: fă X numeric (ca să nu ai stringuri)
    X = make_features_numeric(X)

    y_enc, le = encode_labels(y)

    counts = pd.Series(y_enc).value_counts()
    min_count = int(counts.min()) if len(counts) else 0
    print("[INFO] Class counts:\n", counts.sort_index().to_string())
    print("[INFO] Min class count:", min_count)

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

    # dacă train rămâne cu o singură clasă (set mic), reîncearcă
    if len(np.unique(y_train)) < 2:
        print("[WARN] Train set ended up with <2 classes. Retrying split without stratify and different random_state.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_enc,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE + 1,
            stratify=None,
        )

    rf, logreg, scaler = train_models(X_train, y_train)

    compare_models(rf, logreg, scaler, X_test, y_test, le, OUT_REPORT_TXT)

    print("[INFO] Done. Output saved to:", OUT_REPORT_TXT)
