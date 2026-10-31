import pandas as pd
import pytest

from src.data import RAW_DIR, load_mendeley
from src.splits import SEEDS, SPLITS_PATH, leakage_counts, make_split, remove_overlap


def toy_dataset(n_templates: int = 50, copies: int = 3) -> pd.DataFrame:
    """Each template appears `copies` times; every fifth template is positive."""
    rows = []
    for t in range(n_templates):
        for c in range(copies):
            rows.append({"id": f"toy_{t}_{c}", "text": f"message {t} copy {c}",
                         "template": f"t{t}", "label_orig": "pos" if t % 5 == 0 else "neg",
                         "dataset": "toy"})
    return pd.DataFrame(rows)


def test_split_keeps_templates_together():
    df = toy_dataset()
    split = make_split(df, seed=0)
    assert set(split) == {"train", "val", "test"}
    assert not any(leakage_counts(df, split).values())


def test_split_ratio_is_about_64_16_20():
    df = toy_dataset(n_templates=200)
    share = make_split(df, seed=0).value_counts(normalize=True)
    assert share["test"] == pytest.approx(0.20, abs=0.03)
    assert share["val"] == pytest.approx(0.16, abs=0.03)


def test_seeds_give_different_test_sets():
    df = toy_dataset(n_templates=200)
    tests = [set(df["id"][make_split(df, seed) == "test"]) for seed in (0, 1)]
    assert tests[0] != tests[1]


def test_remove_overlap_never_touches_target():
    source = pd.DataFrame({"text": ["Win NOW", "hello", "call 123"],
                           "template": ["winnow", "hello", "call<num>"]})
    target = pd.DataFrame({"text": ["win now"], "template": ["call<num>"]})
    kept, counts = remove_overlap(source, target)
    assert list(kept["text"]) == ["hello"]
    assert counts == {"exact_text": 1, "template": 1, "removed": 2}


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="run `python -m src.splits` first")
def test_saved_splits_have_one_row_per_id_and_seed():
    splits = pd.read_csv(SPLITS_PATH)
    assert not splits.duplicated(["id", "seed"]).any()
    assert set(splits["split"]) == {"train", "val", "test"}


def test_bothbosu_keeps_official_test_and_varies_val():
    df = toy_dataset(n_templates=100)
    df["dataset"] = "bothbosu"
    official_test = df["template"].isin([f"t{i}" for i in range(20)])
    df["id"] = [f"bothbosu_{'test' if t else 'train'}_{i}" for i, t in enumerate(official_test)]
    splits = [make_split(df, seed) for seed in (0, 1)]
    for split in splits:
        assert list(split == "test") == list(official_test)
        assert leakage_counts(df, split)["shared_templates"] == 0
    assert not splits[0].equals(splits[1])


mendeley_file = RAW_DIR / "mendeley" / "Dataset_5971.csv"


@pytest.mark.skipif(not (SPLITS_PATH.exists() and mendeley_file.exists()),
                    reason="needs splits.csv and the Mendeley raw data")
def test_saved_splits_match_current_code():
    # If this fails, re-run `python -m src.splits`
    saved = pd.read_csv(SPLITS_PATH)
    df = load_mendeley()
    for seed in SEEDS:
        expected = make_split(df, seed)
        got = saved[(saved["dataset"] == "mendeley") & (saved["seed"] == seed)]
        assert list(got["id"]) == list(df["id"])
        assert list(got["split"]) == list(expected)


def test_large_group_is_not_in_test_for_every_seed():
    # StratifiedGroupKFold puts the largest groups in the same fold for every seed,
    # so taking the same fold each time would keep one big campaign in test.
    df = toy_dataset(n_templates=200)
    big = pd.DataFrame({"id": [f"big_{i}" for i in range(60)], "text": [f"campaign {i}" for i in range(60)],
                        "template": "big", "label_orig": "pos", "dataset": "toy"})
    df = pd.concat([df, big], ignore_index=True)
    in_test = [(make_split(df, seed)[df["template"] == "big"] == "test").all() for seed in SEEDS]
    assert not all(in_test)
