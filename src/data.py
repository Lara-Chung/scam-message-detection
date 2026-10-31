"""Load and clean the scam message datasets into one common format.

Every loader returns a DataFrame with these columns:
    id, text, label_orig, label_spam, label_fraud, subtype,
    dataset, source, channel, template

label_spam and label_fraud use pandas' nullable Int64, so "not applicable"
is <NA> and never 0. The number of rows removed by each cleaning step is
stored in df.attrs["cleaning_log"].
"""

import html
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

COLUMNS = [
    "id", "text", "label_orig", "label_spam", "label_fraud", "subtype",
    "dataset", "source", "channel", "template",
]

# Also matches links without a scheme, such as "smsg.io/fCVbD" or "jamster.co.uk".
# The lookbehind skips e-mail domains ("john@gmail.com"); only a "/path" may follow.
URL_RE = re.compile(
    r"https?://\S+|www\.\S+"
    r"|(?<![@\w.])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|info|biz|io|ly|uk)\b(?:/\S*)?",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^<>]{0,200}>")
DIGITS_RE = re.compile(r"\d+")
SPACES_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9<>]")


def dedup_key(text: str) -> str:
    """Lower-case and collapse white space; used to find duplicate texts."""
    return SPACES_RE.sub(" ", text).strip().lower()


def make_template(text: str) -> str:
    """Grouping key for splits: one scam template, or one e-mail in two formats, maps to one group.

    Removes <...> tags and stray angle brackets, replaces URLs and digit runs, then drops everything except
    letters, digits and the placeholders (spaces and punctuation included).
    """
    text = TAG_RE.sub(" ", dedup_key(text)).replace("<", " ").replace(">", " ")
    text = URL_RE.sub("<url>", text)
    text = DIGITS_RE.sub("<num>", text)
    return NON_ALNUM_RE.sub("", text)


def drop_empty(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Remove rows whose text is missing or blank."""
    keep = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
    log["empty_text"] = int((~keep).sum())
    return df[keep].copy()


def drop_conflicting_labels(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Remove every copy of a text that appears with more than one label."""
    key = df["text"].map(dedup_key)
    n_labels = df.groupby(key)["label_orig"].transform("nunique")
    keep = n_labels == 1
    log["conflicting_labels"] = int((~keep).sum())
    return df[keep].copy()


def drop_duplicates(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Keep the first copy of each text. Run this after all text cleaning."""
    key = df["text"].map(dedup_key)
    keep = ~key.duplicated()
    log["duplicates"] = int((~keep).sum())
    return df[keep].copy()


def clean(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Cleaning steps shared by all loaders, in order."""
    # Unescape first, so a text that is only an entity (e.g. "&nbsp;") counts as empty.
    df["text"] = df["text"].map(html.unescape, na_action="ignore")
    df = drop_empty(df, log)
    df = drop_conflicting_labels(df, log)
    return drop_duplicates(df, log)


def finalise(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Add the template column, fix column order and dtypes, attach the log."""
    log["final_rows"] = len(df)
    df["template"] = df["text"].map(make_template)
    for col in ["label_spam", "label_fraud"]:
        df[col] = df[col].astype("Int64")
    df = df[COLUMNS].reset_index(drop=True)
    df.attrs["cleaning_log"] = log
    return df


def load_mendeley(path: Path = RAW_DIR / "mendeley" / "Dataset_5971.csv") -> pd.DataFrame:
    """SMS Phishing Dataset (Mishra and Soni, 2022): ham / spam / smishing.

    label_fraud: smishing = 1, ham = 0, spam = <NA> (left out of the fraud task).
    label_spam: <NA>; the spam task (E4) uses only CEAS_08 and SpamAssassin.
    """
    raw = pd.read_csv(path)
    log = {"raw_rows": len(raw)}

    df = pd.DataFrame({
        "id": [f"mendeley_{i}" for i in raw.index],
        "text": raw["TEXT"],
        # The raw file mixes "Smishing"/"smishing" and "Spam"/"spam".
        "label_orig": raw["LABEL"].str.strip().str.lower(),
    })
    unknown = set(df["label_orig"].dropna()) - {"ham", "spam", "smishing"}
    if unknown or df["label_orig"].isna().any():
        raise ValueError(f"Unexpected Mendeley labels: {unknown or 'missing values'}")

    df = clean(df, log)

    df["label_spam"] = pd.NA
    df["label_fraud"] = df["label_orig"].map({"smishing": 1, "ham": 0})
    df["subtype"] = pd.NA
    df["dataset"] = "mendeley"
    df["source"] = "mendeley"
    df["channel"] = "sms"
    return finalise(df, log)


def load_email_fraud(zenodo_dir: Path = RAW_DIR / "zenodo") -> pd.DataFrame:
    """Nazario_5 (phishing) and Nigerian_5 (advance-fee fraud), merged into one set.

    Both come from the Zenodo record of Champa et al. (2024). Only the email body
    is used. The two files share legitimate emails; one copy is kept and its
    source is "nazario_5+nigerian_5".
    label_fraud: fraud = 1, legitimate = 0.
    """
    parts = []
    for source in ["nazario_5", "nigerian_5"]:
        raw = pd.read_csv(zenodo_dir / f"{source.capitalize()}.csv")
        if not set(raw["label"]) <= {0, 1}:
            raise ValueError(f"Unexpected {source} labels: {set(raw['label'])}")
        parts.append(pd.DataFrame({
            "id": [f"{source}_{i}" for i in raw.index],
            "text": raw["body"],
            "label_orig": raw["label"].map({1: "fraud", 0: "legitimate"}),
            "source": source,
        }))
    df = pd.concat(parts, ignore_index=True)
    log = {"raw_rows": len(df)}

    key = df["text"].fillna("").map(html.unescape).map(dedup_key)
    in_both = df.groupby(key)["source"].transform("nunique") > 1
    df.loc[in_both, "source"] = "nazario_5+nigerian_5"
    df = clean(df, log)

    df["label_spam"] = pd.NA
    df["label_fraud"] = df["label_orig"].map({"fraud": 1, "legitimate": 0})
    df["subtype"] = pd.NA
    df["dataset"] = "email_fraud"
    df["channel"] = "email"
    return finalise(df, log)


BOTHBOSU_REVISION = "321b961b5ae353e19ed479b960658dcd223d5e06"  # pinned Hub commit
BOTHBOSU_TYPES = {
    "ssn", "refund", "support", "reward",                   # scam
    "delivery", "wrong", "insurance", "telemarketing",      # non-scam
}


def load_bothbosu(cache_dir: Path = RAW_DIR / "hf_cache") -> pd.DataFrame:
    """BothBosu/scam-dialogue (Hugging Face): synthetic phone dialogues, 8 types.

    The official split is kept in the id ("bothbosu_train_<i>" / "bothbosu_test_<i>").
    A dialogue found in both splits keeps only its train copy.
    label_fraud: scam = 1, non-scam = 0; the insurance and telemarketing types are
    genuine sales calls and get <NA> (left out of the fraud task, proposal 3.3).
    """
    from datasets import load_dataset

    data = load_dataset(
        "BothBosu/scam-dialogue", revision=BOTHBOSU_REVISION, cache_dir=str(cache_dir)
    )
    parts = []
    for split in ["train", "test"]:
        raw = data[split].to_pandas()
        if not set(raw["label"]) <= {0, 1}:
            raise ValueError(f"Unexpected BothBosu labels: {set(raw['label'])}")
        if not set(raw["type"]) <= BOTHBOSU_TYPES:
            raise ValueError(f"Unexpected BothBosu types: {set(raw['type']) - BOTHBOSU_TYPES}")
        parts.append(pd.DataFrame({
            "id": [f"bothbosu_{split}_{i}" for i in raw.index],
            "text": raw["dialogue"],
            "label_orig": raw["label"].map({1: "scam", 0: "non-scam"}),
            "subtype": raw["type"],
        }))
    df = pd.concat(parts, ignore_index=True)
    # label_fraud depends on subtype, so one dialogue must not carry two subtypes.
    n_types = df.groupby(df["text"].map(dedup_key))["subtype"].transform("nunique")
    if (n_types > 1).any():
        raise ValueError("A BothBosu dialogue appears with more than one type")
    log = {"raw_rows": len(df)}
    df = clean(df, log)

    df["label_spam"] = pd.NA
    sales = df["subtype"].isin(["insurance", "telemarketing"])
    df["label_fraud"] = df["label_orig"].map({"scam": 1, "non-scam": 0}).astype("Int64").mask(sales)
    df["dataset"] = "bothbosu"
    df["source"] = "bothbosu"
    df["channel"] = "phone"
    return finalise(df, log)
