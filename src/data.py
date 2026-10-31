"""Load and clean the scam message datasets into one common format.

Every loader returns a DataFrame with these columns:
    id, text, label_orig, label_spam, label_fraud, subtype,
    dataset, source, channel, template

label_spam and label_fraud use pandas' nullable Int64, so "not applicable"
is <NA> and never 0. The number of rows removed by each cleaning step is
stored in df.attrs["cleaning_log"].
"""

import csv
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


# Email artefacts that reveal the source corpus rather than spam (proposal 3.4).
FOOTER_LINE_RE = re.compile(
    r"^(_{10,}|-{20,}"               # separator lines
    r"|-+\s*original message\s*-+"  # start of a forwarded message
    r"|\S+ mailing list"             # "Python-Dev mailing list" (one-word list name)
    r"|\S*(listinfo|mailman/)\S*"    # a line that is only a list management URL
    r"|unsubscribe:.*"
    r"|this sf\.net email is sponsored by.*)$",
    re.IGNORECASE,
)
LIST_NAME_RE = re.compile(r"^\S+ mailing list$", re.IGNORECASE)
ADDRESS_RE = re.compile(r"^\S+@\S+$")
HEADER_LINE_RE = re.compile(
    r"^(from|sent|to|cc|subject|date|reply-to|message-id):", re.IGNORECASE
)
LIST_URL_RE = re.compile(r"\S*(listinfo|mailman/)\S*", re.IGNORECASE)
# Footers and digest headers inside bodies that have lost their line breaks
INLINE_FOOTER_RE = re.compile(
    r"_{10,}[\s>]*\S+ mailing list[\s>]*(\S+@\S+)?"
    r"|(?s:send \S+ mailing list submissions to.{0,800}?contents of \S+ digest\.*\"?)"
    r"|send \S+ mailing list submissions to\s*\S+",
    re.IGNORECASE,
)


def strip_artefacts(text: str) -> str:
    """Remove mailing-list footers, header lines copied into the body and the signature."""
    lines = text.split("\n")  # not splitlines(): keep \r and other characters as they are
    # The signature starts at the last "-- " line (RFC 3676 delimiter, with the
    # trailing space), if it is within the last 10 lines and is not the first line.
    # A bare "--" is not used, because spam often uses it as a separator.
    sig = [i for i, line in enumerate(lines) if line.rstrip("\r") == "-- "]
    if sig and sig[-1] > 0 and len(lines) - sig[-1] <= 10:
        lines = lines[:sig[-1]]
    bare = [line.lstrip("> \t").strip() for line in lines]
    is_header = [bool(HEADER_LINE_RE.match(b)) for b in bare]
    kept = []
    for i, line in enumerate(lines):
        # A copied header comes as a block (From/Sent/To/Subject); a single
        # "From: ..." line can be part of the message itself, so it is kept.
        in_block = is_header[i] and (
            (i > 0 and is_header[i - 1]) or (i + 1 < len(lines) and is_header[i + 1])
        )
        list_address = ADDRESS_RE.match(bare[i]) and i > 0 and LIST_NAME_RE.match(bare[i - 1])
        if FOOTER_LINE_RE.match(bare[i]) or in_block or list_address:
            continue
        kept.append(line)
    # Some bodies have lost their line breaks; drop footers and list URLs found inline.
    text = INLINE_FOOTER_RE.sub(" ", "\n".join(kept))
    return LIST_URL_RE.sub(" ", text).strip()


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


def load_uci(
    path: Path = RAW_DIR / "uci" / "SMSSpamCollection", keep_duplicates: bool = False
) -> pd.DataFrame:
    """UCI SMS Spam Collection (Almeida et al., 2011): ham / spam. Used in E1 only.

    keep_duplicates=True keeps the raw text, file order and duplicates, as in the
    published 30/70 protocol that E1 replicates.
    """
    raw = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                      quoting=csv.QUOTE_NONE)
    if not set(raw["label"]) <= {"ham", "spam"}:
        raise ValueError(f"Unexpected UCI labels: {set(raw['label'])}")
    df = pd.DataFrame({
        "id": [f"uci_{i}" for i in raw.index],
        "text": raw["text"],
        "label_orig": raw["label"],
    })
    log = {"raw_rows": len(df)}
    if keep_duplicates:
        df = drop_empty(df, log)  # raw text, as published: no unescape, no de-duplication
    else:
        df = clean(df, log)

    df["label_spam"] = pd.NA
    df["label_fraud"] = pd.NA
    df["subtype"] = pd.NA
    df["dataset"] = "uci"
    df["source"] = "uci"
    df["channel"] = "sms"
    return finalise(df, log)


def load_spam_email(name: str, strip: bool = False,
                    zenodo_dir: Path = RAW_DIR / "zenodo") -> pd.DataFrame:
    """CEAS_08 or SpamAssassin from the Zenodo record of Champa et al. (2024). Used in E1 and E4.

    name: "ceas_08" or "spamassassin". Only the email body is used.
    strip=True returns the text after strip_artefacts. Both versions keep the same
    rows (cleaning runs on the stripped text), so E4 compares the same emails
    with and without stripping.
    label_spam: spam = 1, legitimate = 0. label_fraud: <NA> (spam includes marketing).
    """
    file_names = {"ceas_08": "CEAS_08.csv", "spamassassin": "SpamAssasin.csv"}
    if name not in file_names:
        raise ValueError(f"name must be one of {sorted(file_names)}, got {name!r}")
    raw = pd.read_csv(zenodo_dir / file_names[name])
    if not set(raw["label"]) <= {0, 1}:
        raise ValueError(f"Unexpected {name} labels: {set(raw['label'])}")
    df = pd.DataFrame({
        "id": [f"{name}_{i}" for i in raw.index],
        "text": raw["body"],
        "label_orig": raw["label"].map({1: "spam", 0: "legitimate"}),
    })
    log = {"raw_rows": len(df)}
    df["text"] = df["text"].map(html.unescape, na_action="ignore")
    original = df["text"]
    df["text"] = df["text"].map(strip_artefacts, na_action="ignore")
    df = drop_empty(df, log)
    df = drop_conflicting_labels(df, log)
    df = drop_duplicates(df, log)
    # Texts that differ only in line breaks can strip differently; drop those
    # duplicates of the original text too, so both versions stay de-duplicated.
    dup_original = original[df.index].map(dedup_key).duplicated()
    log["duplicates"] += int(dup_original.sum())
    df = df[~dup_original].copy()

    df["label_spam"] = df["label_orig"].map({"spam": 1, "legitimate": 0})
    df["label_fraud"] = pd.NA
    df["subtype"] = pd.NA
    df["dataset"] = name
    df["source"] = name
    df["channel"] = "email"
    kept_rows = df.index
    df = finalise(df, log)  # template comes from the stripped text in both versions
    if not strip:
        df["text"] = original.loc[kept_rows].values  # finalise keeps the row order
    return df
