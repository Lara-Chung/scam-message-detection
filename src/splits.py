"""Leakage-safe train / val / test splits and overlap removal.

Each dataset is split once per seed, grouped by template (one template never
appears in two splits; CEAS_08 uses a template prefix, see PREFIX_GROUPED) and stratified by label_orig as far as the groups allow.
Every experiment uses the same split and drops the rows whose label column is
<NA>; the class balance after that drop is not controlled and is checked separately.

Run `python -m src.splits` from the repo root to write data/processed/splits.csv
(columns: id, dataset, seed, split) and print the checks.
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data import (
    REPO_ROOT, dedup_key, load_bothbosu, load_email_fraud, load_mendeley,
    load_spam_email, load_uci,
)

SEEDS = [0, 1, 2]
# CEAS_08 has mass campaigns (e.g. 2,930 "THE DAILY TOP 10 from CNN" spam emails)
# whose templates differ only after the opening, so it is grouped by a prefix.
# 90 characters is the longest prefix that keeps that campaign in one group.
PREFIX_GROUPED = {"ceas_08": 90}
SPLITS_PATH = REPO_ROOT / "data" / "processed" / "splits.csv"


def split_group(df: pd.DataFrame) -> pd.Series:
    """Group key used only for splitting: the template, or its prefix for PREFIX_GROUPED datasets."""
    n_chars = PREFIX_GROUPED.get(df["dataset"].iloc[0])
    return df["template"].str[:n_chars] if n_chars else df["template"]


def take_one_fold(df: pd.DataFrame, seed: int) -> pd.Series:
    """Boolean mask of one fold out of five (about 20% of df)."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, fold = next(sgkf.split(df, df["label_orig"], groups=split_group(df)))
    mask = pd.Series(False, index=df.index)
    mask.iloc[fold] = True
    return mask


def make_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """Return "train" / "val" / "test" for each row: about 64 / 16 / 20.

    BothBosu keeps its official test set; the seed only changes the val set.
    """
    split = pd.Series("train", index=df.index)
    if (df["dataset"] == "bothbosu").all():
        is_test = df["id"].str.startswith("bothbosu_test_")
    else:
        is_test = take_one_fold(df, seed)
    split[is_test] = "test"
    rest = df[~is_test]
    split[take_one_fold(rest, seed).reindex(df.index, fill_value=False)] = "val"
    return split


def leakage_counts(df: pd.DataFrame, split: pd.Series) -> dict:
    """Number of split groups, templates and texts found in more than one split (all must be 0)."""
    key = df["text"].map(dedup_key)
    return {
        "shared_groups": int((split.groupby(split_group(df)).nunique() > 1).sum()),
        "shared_templates": int((split.groupby(df["template"]).nunique() > 1).sum()),
        "shared_texts": int((split.groupby(key).nunique() > 1).sum()),
    }


def remove_overlap(source: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop source rows (train/val candidates) that also appear in the target test set (E3, E4).

    The target is never changed. A row counts as overlapping if its exact text
    (dedup_key) or its template matches a target row; the template match also
    catches one email in two formats. Returns the filtered source and the
    number of rows matched by each rule.
    """
    exact = source["text"].map(dedup_key).isin(set(target["text"].map(dedup_key)))
    template = source["template"].isin(set(target["template"]))
    counts = {"exact_text": int(exact.sum()), "template": int(template.sum()),
              "removed": int((exact | template).sum())}
    return source[~(exact | template)], counts


def load_all() -> list[pd.DataFrame]:
    """The six experimental datasets (CEAS_08 and SpamAssassin share ids across strip versions)."""
    return [
        load_mendeley(), load_email_fraud(), load_bothbosu(), load_uci(),
        load_spam_email("ceas_08"), load_spam_email("spamassassin"),
    ]


def main() -> None:
    rows, checks = [], []
    for df in load_all():
        name = df["dataset"].iloc[0]
        for seed in SEEDS:
            split = make_split(df, seed)
            rows.append(pd.DataFrame({"id": df["id"], "dataset": name, "seed": seed, "split": split}))
            counts = split.value_counts()
            leaks = leakage_counts(df, split)
            if any(leaks.values()):
                raise RuntimeError(f"Leakage in {name}, seed {seed}: {leaks}")
            checks.append({
                "dataset": name, "seed": seed,
                **{s: int(counts.get(s, 0)) for s in ["train", "val", "test"]},
                **leaks,
            })
    splits = pd.concat(rows, ignore_index=True)
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_PATH, index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    print(f"Wrote {len(splits)} rows to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
