# -*- coding: utf-8 -*-
"""
Old-school NLP baseline adapted for ESG multi-label competition.

What it does
- Reads train.csv, test.csv, sample_submission.csv (paths configurable below)
- Uses TF-IDF word + character n-grams
- Trains one LogisticRegression per label (E, S, G, non_ESG) with CV
- Averages fold predictions and writes submission.csv

Author: adapted from old_school_nlp.py
"""

import os
import re
import string
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss


# =====================
# CONFIG
# =====================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"   # optional (if provided, we will match its column order)
OUT_PATH = "submission.csv"

TEXT_COL = "text"
ID_COL = "id"
LABEL_COLS = ["E", "S", "G", "non_ESG"]

N_SPLITS = 5
SEED = 42

# Vectorizer sizes: adjust up/down depending on RAM
WORD_MAX_FEATURES = 50000
CHAR_MAX_FEATURES = 100000


# =====================
# HELPERS
# =====================
def basic_clean(s: str) -> str:
    """Light cleaning. Char n-grams already capture a lot, so keep it simple."""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_stratify_key(y_df: pd.DataFrame, n_splits: int) -> np.ndarray:
    """
    Create a stratification label for multi-label data by using the label-combination string.
    If rare combinations exist (< n_splits), bucket them into '__RARE__' so StratifiedKFold works.
    """
    combos = y_df.astype(int).astype(str).agg("".join, axis=1)
    vc = combos.value_counts()
    rare = set(vc[vc < n_splits].index)
    combos = combos.where(~combos.isin(rare), "__RARE__")
    return combos.values


def fit_vectorizers(all_text: pd.Series):
    word_vec = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 2),
        max_features=WORD_MAX_FEATURES,
        min_df=2,
    )

    char_vec = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="char",
        ngram_range=(3, 6),
        max_features=CHAR_MAX_FEATURES,
        min_df=2,
    )

    word_vec.fit(all_text)
    char_vec.fit(all_text)
    return word_vec, char_vec


def build_features(text_series: pd.Series, word_vec, char_vec):
    w = word_vec.transform(text_series)
    c = char_vec.transform(text_series)
    return hstack([c, w]).tocsr()


# =====================
# MAIN
# =====================
def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Basic sanity checks
    missing = [c for c in [ID_COL, TEXT_COL] + LABEL_COLS if c not in train.columns]
    if missing:
        raise ValueError(f"Train is missing columns: {missing}")
    for c in [ID_COL, TEXT_COL]:
        if c not in test.columns:
            raise ValueError(f"Test is missing column: {c}")

    train[TEXT_COL] = train[TEXT_COL].fillna("").map(basic_clean)
    test[TEXT_COL] = test[TEXT_COL].fillna("").map(basic_clean)

    # Fit vectorizers on train+test text (standard competition trick)
    all_text = pd.concat([train[TEXT_COL], test[TEXT_COL]], axis=0, ignore_index=True)
    word_vec, char_vec = fit_vectorizers(all_text)

    # Prepare stratification key
    y = train[LABEL_COLS].copy()
    strat_key = make_stratify_key(y, N_SPLITS)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros((len(train), len(LABEL_COLS)), dtype=float)
    test_pred = np.zeros((len(test), len(LABEL_COLS)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, strat_key), 1):
        print(f"\n================ Fold {fold}/{N_SPLITS} ================")
        tr_text = train.loc[tr_idx, TEXT_COL]
        va_text = train.loc[va_idx, TEXT_COL]

        X_tr = build_features(tr_text, word_vec, char_vec)
        X_va = build_features(va_text, word_vec, char_vec)
        X_te = build_features(test[TEXT_COL], word_vec, char_vec)

        for j, label in enumerate(LABEL_COLS):
            y_tr = train.loc[tr_idx, label].astype(int).values
            y_va = train.loc[va_idx, label].astype(int).values

            clf = LogisticRegression(
                C=4.0,
                solver="saga",
                penalty="l2",
                max_iter=4000,
                n_jobs=-1,
                class_weight="balanced",
                verbose=0,
            )

            clf.fit(X_tr, y_tr)

            va_proba = clf.predict_proba(X_va)[:, 1]
            te_proba = clf.predict_proba(X_te)[:, 1]

            oof[va_idx, j] = va_proba
            test_pred[:, j] += te_proba / N_SPLITS

            # quick metric prints (AUC + logloss)
            try:
                auc = roc_auc_score(y_va, va_proba)
            except ValueError:
                auc = np.nan
            ll = log_loss(y_va, np.clip(va_proba, 1e-6, 1 - 1e-6))
            print(f"{label:8s} | AUC={auc:.4f} | logloss={ll:.4f}")

    # Overall OOF metrics
    print("\n================ OOF Summary ================")
    for j, label in enumerate(LABEL_COLS):
        y_true = train[label].astype(int).values
        y_pred = oof[:, j]
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan
        ll = log_loss(y_true, np.clip(y_pred, 1e-6, 1 - 1e-6))
        print(f"{label:8s} | AUC={auc:.4f} | logloss={ll:.4f}")

    # Build submission
    sub = pd.DataFrame({ID_COL: test[ID_COL].values})
    for j, label in enumerate(LABEL_COLS):
        sub[label] = test_pred[:, j]

    # If sample submission exists, match column order
    if os.path.exists(SAMPLE_SUB_PATH):
        sample = pd.read_csv(SAMPLE_SUB_PATH)
        keep_cols = [c for c in sample.columns if c in sub.columns]
        sub = sub[keep_cols]

    sub.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}  shape={sub.shape}")


if __name__ == "__main__":
    main()
