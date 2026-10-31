import pytest

from src.data import (
    COLUMNS, RAW_DIR, dedup_key, load_email_fraud, load_mendeley, make_template,
)


def test_dedup_key_ignores_case_and_spacing():
    assert dedup_key("  Call   NOW\n") == dedup_key("call now")


def test_template_replaces_urls_and_digits():
    text = "Call 08712345678 or visit http://bank-verify.com/a1 to claim 500"
    assert make_template(text) == "call<num>orvisit<url>toclaim<num>"


def test_template_replaces_links_without_scheme():
    assert make_template("Claim at smsg.io/fCVbD now") == "claimat<url>now"
    assert make_template("Fun from jamster.co.uk!") == "funfrom<url>"


def test_template_keeps_non_links():
    # e-mail domains and words joined by a period are not links
    assert make_template("email me at john@gmail.com") == "emailmeatjohngmailcom"
    assert make_template("ok.me too") == "okmetoo"
    assert make_template("e.g. meet at 5") == "egmeetat<num>"


def test_template_matches_format_variants():
    # The same e-mail with and without line breaks and an <address> tag
    a = 'From: "BNA Highlights" <bhighlig@bna.com>\nSent: June 28, 2001'
    b = 'From: "BNA Highlights"  Sent: June 28, 2001'
    assert make_template(a) == make_template(b)


mendeley_file = RAW_DIR / "mendeley" / "Dataset_5971.csv"


@pytest.mark.skipif(not mendeley_file.exists(), reason="Mendeley raw data not downloaded")
def test_mendeley_labels_and_counts():
    df = load_mendeley()
    assert list(df.columns) == COLUMNS
    assert df["id"].is_unique
    assert set(df["label_orig"]) == {"ham", "spam", "smishing"}
    # spam is not applicable to the fraud task, so it must stay <NA>, not 0
    assert df.loc[df["label_orig"] == "spam", "label_fraud"].isna().all()
    assert (df.loc[df["label_orig"] == "smishing", "label_fraud"] == 1).all()
    assert (df.loc[df["label_orig"] == "ham", "label_fraud"] == 0).all()
    assert df["label_spam"].isna().all()
    assert not df["text"].map(dedup_key).duplicated().any()


zenodo_dir = RAW_DIR / "zenodo"


@pytest.mark.skipif(
    not (zenodo_dir / "Nazario_5.csv").exists() or not (zenodo_dir / "Nigerian_5.csv").exists(),
    reason="Nazario_5 / Nigerian_5 raw data not downloaded",
)
def test_email_fraud_labels_and_counts():
    df = load_email_fraud()
    assert list(df.columns) == COLUMNS
    assert df["id"].is_unique
    assert set(df["source"]) == {"nazario_5", "nigerian_5", "nazario_5+nigerian_5"}
    assert df["label_fraud"].notna().all()
    assert (df["label_fraud"] == (df["label_orig"] == "fraud")).all()
    assert not df["text"].map(dedup_key).duplicated().any()
    assert not (df["text"].str.strip() == "").any()
