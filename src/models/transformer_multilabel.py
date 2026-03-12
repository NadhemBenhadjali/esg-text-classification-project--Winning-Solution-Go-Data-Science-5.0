# -*- coding: utf-8 -*-
"""
Transformer baseline adapted from gods2025_firsty.py to the ESG multi-label competition.

Dataset expected:
- train.csv with columns: id, text, E, S, G, non_ESG   (labels are 0/1)
- test.csv with columns: id, text
(Optional) sample_submission.csv to enforce column order.

Outputs:
- submission_probs.csv (probabilities)
- submission_binary.csv (0/1 with threshold)

Key changes vs the original GODS code:
- Multi-label head (num_labels=4) with sigmoid
- BCEWithLogitsLoss (+ optional pos_weight for imbalance)
- No GPT2 augmentation (slow + not needed for a strong baseline)
- Simple multi-label stratification via label-combo buckets
"""

import os
import re
import math
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


# =====================
# CONFIG
# =====================
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"  # optional

OUT_PROBS = "submission_probs.csv"
OUT_BIN   = "submission_binary.csv"

ID_COL = "id"
TEXT_COL = "text"
LABEL_COLS = ["E", "S", "G", "non_ESG"]

MODEL_NAME = "microsoft/deberta-v3-base"   # good default
MAX_LEN = 256                               # 512 is slower; start with 256
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 2                                  # start small; increase later
N_SPLITS = 5
SEED = 42
NUM_WORKERS = os.cpu_count() or 2
THRESH = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def basic_clean(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_stratify_key(y_df: pd.DataFrame, n_splits: int) -> np.ndarray:
    """
    Stratify on multi-label combination strings; bucket rare combos to avoid fold errors.
    """
    combos = y_df.astype(int).astype(str).agg("".join, axis=1)
    vc = combos.value_counts()
    rare = set(vc[vc < n_splits].index)
    combos = combos.where(~combos.isin(rare), "__RARE__")
    return combos.values


class ESGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels  # can be None for test
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, num_labels)
        nn.init.xavier_normal_(self.head.weight)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation (works for DeBERTa/RoBERTa/BERT-family)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(self.dropout(cls))
        return logits


def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    """
    pos_weight for BCEWithLogitsLoss: (N - pos) / pos per label.
    Helps with imbalance. Safe-guard division by zero.
    """
    y = y.astype(np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pw = (neg / np.clip(pos, 1.0, None)).astype(np.float32)
    return torch.tensor(pw, device=DEVICE)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attn)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def train_one_fold(fold, train_idx, val_idx, df, tokenizer, pos_weight):
    train_texts = df.loc[train_idx, TEXT_COL].tolist()
    val_texts   = df.loc[val_idx, TEXT_COL].tolist()
    y_train = df.loc[train_idx, LABEL_COLS].values.astype(np.float32)
    y_val   = df.loc[val_idx, LABEL_COLS].values.astype(np.float32)

    train_ds = ESGDataset(train_texts, y_train, tokenizer, MAX_LEN)
    val_ds   = ESGDataset(val_texts,   y_val,   tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = MultiLabelClassifier(MODEL_NAME, num_labels=len(LABEL_COLS)).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # validation
        val_probs = predict(model, val_loader)
        val_pred = (val_probs >= THRESH).astype(int)
        f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

        print(f"[fold {fold}] epoch {epoch}/{EPOCHS} | train_loss={total_loss/len(train_loader):.4f} | val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # load best
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return model, best_f1


def main():
    seed_everything(SEED)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # Basic checks
    need_train = [ID_COL, TEXT_COL] + LABEL_COLS
    missing_train = [c for c in need_train if c not in train_df.columns]
    if missing_train:
        raise ValueError(f"Train missing columns: {missing_train}")

    need_test = [ID_COL, TEXT_COL]
    missing_test = [c for c in need_test if c not in test_df.columns]
    if missing_test:
        raise ValueError(f"Test missing columns: {missing_test}")

    train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").map(basic_clean)
    test_df[TEXT_COL]  = test_df[TEXT_COL].fillna("").map(basic_clean)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    y_all = train_df[LABEL_COLS].values.astype(np.float32)
    pos_weight = compute_pos_weight(y_all)

    strat_key = make_stratify_key(train_df[LABEL_COLS], N_SPLITS)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros((len(train_df), len(LABEL_COLS)), dtype=np.float32)
    test_probs = np.zeros((len(test_df), len(LABEL_COLS)), dtype=np.float32)

    fold_scores = []

    # Prebuild test loader once
    test_ds = ESGDataset(test_df[TEXT_COL].tolist(), labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_key), 1):
        model, best_f1 = train_one_fold(fold, tr_idx, va_idx, train_df, tokenizer, pos_weight)
        fold_scores.append(best_f1)

        # oof preds
        val_texts = train_df.loc[va_idx, TEXT_COL].tolist()
        val_ds = ESGDataset(val_texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
        oof[va_idx] = predict(model, val_loader)

        # test preds (average across folds)
        test_probs += predict(model, test_loader) / N_SPLITS

        del model
        torch.cuda.empty_cache()

    print("\nFold macro-F1:", [round(x, 4) for x in fold_scores], " | mean=", round(float(np.mean(fold_scores)), 4))

    # Save probability submission
    sub_probs = pd.DataFrame({ID_COL: test_df[ID_COL].values})
    for j, c in enumerate(LABEL_COLS):
        sub_probs[c] = test_probs[:, j]
    # Align columns to sample submission if present
    if os.path.exists(SAMPLE_SUB_PATH):
        sample = pd.read_csv(SAMPLE_SUB_PATH)
        keep = [c for c in sample.columns if c in sub_probs.columns]
        sub_probs = sub_probs[keep]
    sub_probs.to_csv(OUT_PROBS, index=False)
    print(f"Saved {OUT_PROBS} shape={sub_probs.shape}")

    # Save binary submission
    sub_bin = sub_probs.copy()
    for c in LABEL_COLS:
        if c in sub_bin.columns:
            sub_bin[c] = (sub_bin[c].astype(float) >= THRESH).astype(int)
    sub_bin.to_csv(OUT_BIN, index=False)
    print(f"Saved {OUT_BIN} shape={sub_bin.shape}")


if __name__ == "__main__":
    main()
