"""Export the processed datasets for sharing outside the repo (e.g. a cloud folder).

Run `python -m src.export` from the repo root. It writes one gzip CSV per
dataset to data/processed/export/ (git-ignored): the columns from src/data.py
plus split_seed0, split_seed1 and split_seed2 from data/processed/splits.csv.
CEAS_08 and SpamAssassin also get a text_stripped column (artefacts removed).
uci_replication.csv.gz keeps the raw UCI file order and duplicates for the
published 30/70 protocol and has no split columns.
"""

import pandas as pd

from src.data import REPO_ROOT, load_spam_email, load_uci
from src.splits import SEEDS, SPLITS_PATH, load_all

EXPORT_DIR = REPO_ROOT / "data" / "processed" / "export"


def add_split_columns(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Add one split column per seed; the dataset and splits.csv must have the same ids."""
    name = df["dataset"].iloc[0]
    if set(splits.loc[splits["dataset"] == name, "id"]) != set(df["id"]):
        raise ValueError(f"{name}: ids differ from splits.csv; re-run `python -m src.splits`")
    for seed in SEEDS:
        s = splits[splits["seed"] == seed].set_index("id")["split"]
        df[f"split_seed{seed}"] = df["id"].map(s)
        if df[f"split_seed{seed}"].isna().any():
            raise ValueError(f"{name}: rows without a split for seed {seed}; "
                             "re-run `python -m src.splits`")
    return df


def main() -> None:
    splits = pd.read_csv(SPLITS_PATH)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for df in load_all():
        name = df["dataset"].iloc[0]
        if name in ("ceas_08", "spamassassin"):
            stripped = load_spam_email(name, strip=True)
            if list(stripped["id"]) != list(df["id"]):
                raise ValueError(f"{name}: strip versions have different rows")
            df["text_stripped"] = stripped["text"].values
        df = add_split_columns(df, splits)
        path = EXPORT_DIR / f"{name}.csv.gz"
        df.to_csv(path, index=False)
        print(f"{path.name}: {len(df)} rows")
    uci_full = load_uci(keep_duplicates=True)
    uci_full.to_csv(EXPORT_DIR / "uci_replication.csv.gz", index=False)
    print(f"uci_replication.csv.gz: {len(uci_full)} rows")


if __name__ == "__main__":
    main()
