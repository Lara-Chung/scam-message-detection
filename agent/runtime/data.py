"""Canonical export access. No raw downloads and no split regeneration."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data import REPO_ROOT
from src.splits import leakage_counts

CHANNEL_DATASETS = {"sms": "mendeley", "email": "email_fraud", "phone": "bothbosu"}
EXPORT = REPO_ROOT / "data/processed/export"
LABEL_DEFINITION = "label_fraud: 1=fraud/phishing/scam, 0=legitimate; NA excluded"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def membership_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def load_channel(channel: str, export_dir: Path = EXPORT, seed: int = 0) -> pd.DataFrame:
    if channel not in CHANNEL_DATASETS:
        raise ValueError("unsupported_channel")
    path = export_dir / f"{CHANNEL_DATASETS[channel]}.csv.gz"
    df = pd.read_csv(path, dtype={"label_fraud": "Int64", "label_spam": "Int64"}, keep_default_na=False,
                     na_values={"label_fraud": [""], "label_spam": [""]})
    split = f"split_seed{seed}"
    required = {"id", "text", "template", "label_orig", "label_fraud", "dataset", "channel", split}
    if not required.issubset(df.columns):
        raise ValueError("export_columns_missing")
    if not df.id.is_unique or df[["id", "text", "template", split]].isna().any().any():
        raise ValueError("invalid_export_rows")
    if set(df[split]) != {"train", "val", "test"}:
        raise ValueError("invalid_splits")
    if set(df.channel) != {channel} or set(df.dataset) != {CHANNEL_DATASETS[channel]}:
        raise ValueError("export_identity_mismatch")
    expected = df.label_orig.map({"ham": 0, "smishing": 1, "fraud": 1,
                                  "legitimate": 0, "scam": 1, "non-scam": 0}).astype("Int64")
    if channel == "phone":
        expected = expected.mask(df.subtype.isin(["insurance", "telemarketing"]))
    if not expected.equals(df.label_fraud):
        raise ValueError("label_mapping_mismatch")
    if any(leakage_counts(df, df[split]).values()):
        raise ValueError("canonical_split_leakage")
    return df


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """NA is inapplicable, never the negative class."""
    return df.loc[df.label_fraud.notna()].copy()
