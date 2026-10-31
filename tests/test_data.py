import pytest

from src.data import (
    BOTHBOSU_REVISION, COLUMNS, RAW_DIR, dedup_key, load_bothbosu, load_email_fraud, load_mendeley,
    load_spam_email, load_uci, make_template, strip_artefacts,
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
    assert make_template("mail john@my-site.com") == "mailjohnmysitecom"
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


bothbosu_cached = any((RAW_DIR / "hf_cache").rglob("*.arrow"))


@pytest.mark.skipif(not bothbosu_cached, reason="BothBosu not downloaded")
def test_bothbosu_labels_and_split_ids():
    datasets = pytest.importorskip("datasets")
    df = load_bothbosu()
    assert list(df.columns) == COLUMNS
    assert df["id"].is_unique
    assert df["id"].str.match(r"^bothbosu_(train|test)_\d+$").all()
    sales = df["subtype"].isin(["insurance", "telemarketing"])
    assert df.loc[sales, "label_fraud"].isna().all()
    assert df.loc[~sales, "label_fraud"].notna().all()
    assert (df.loc[~sales, "label_fraud"] == (df.loc[~sales, "label_orig"] == "scam")).all()
    assert not df["text"].map(dedup_key).duplicated().any()

    # A dialogue in both official splits keeps its train copy only.
    raw = datasets.load_dataset(
        "BothBosu/scam-dialogue", revision=BOTHBOSU_REVISION, cache_dir=str(RAW_DIR / "hf_cache")
    )
    train_keys = {dedup_key(x) for x in raw["train"]["dialogue"]}
    in_train = df["text"].map(dedup_key).isin(train_keys)
    assert df.loc[in_train, "id"].str.startswith("bothbosu_train_").all()


def test_strip_artefacts_removes_footer_headers_and_signature():
    body = (
        "Hi all,\n"
        "-----Original Message-----\n"
        "From: Someone [mailto:a@b.com]\n"
        "Subject: Re: draw\n"
        "The meeting is at 5.\n"
        "-- \n"
        "Greg\n"
        "_______________________________________________\n"
        "Python-Dev mailing list\n"
        "abc@python.org\n"
        "http://mail.python.org/mailman/listinfo/python-dev\n"
        "Unsubscribe: http://mail.python.org/mailman/options/python-dev/x\n"
    )
    assert strip_artefacts(body) == "Hi all,\nThe meeting is at 5."


def test_strip_artefacts_removes_digest_header():
    body = ("Send Foo mailing list submissions to foo@x.org To subscribe or unsubscribe "
            "via the World Wide Web, visit http://x.org/mailman/listinfo/foo or, via email, "
            "send a message with subject or body 'help' to foo-request@x.org When replying, "
            "please edit your Subject line so it is more specific than "
            "\"Re: Contents of Foo digest...\" Today's Topics: 1. Release")
    assert strip_artefacts(body) == "Today's Topics: 1. Release"


def test_strip_artefacts_handles_bodies_without_line_breaks():
    body = ("Thanks. _______________________________________________ "
            "rpm-list mailing list rpm-list@freshrpms.net "
            "http://lists.freshrpms.net/mailman/listinfo/rpm-list")
    assert strip_artefacts(body) == "Thanks."


def test_strip_artefacts_removes_footer_without_signature():
    body = (
        "Thanks for the patch.\n"
        "_______________________________________________\n"
        "Python-Dev mailing list\n"
        "abc@python.org\n"
        "http://mail.python.org/mailman/listinfo/python-dev\n"
    )
    assert strip_artefacts(body) == "Thanks for the patch."


def test_strip_artefacts_keeps_spam_text():
    for body in [
        "Win a prize now!\nTo be removed from this mailing list, reply REMOVE.",
        "Buy now\nYou received this because you joined our mailing list",
        "From: the desk of Dr. Abu\nI need your help",
        "To claim your prize reply to\nclaims@winner-lotto.com\nnow",
        "--\nWin $1000 now\nclick here",
        "Hot deal\n--\nBuy now at cheap-meds.com\nclick here",
        "Buy now\r\nCheap meds",
    ]:
        assert strip_artefacts(body) == body


uci_file = RAW_DIR / "uci" / "SMSSpamCollection"


@pytest.mark.skipif(not uci_file.exists(), reason="UCI raw data not downloaded")
def test_uci_dedup_and_replication_version():
    df = load_uci()
    assert list(df.columns) == COLUMNS
    assert not df["text"].map(dedup_key).duplicated().any()
    assert df["label_spam"].isna().all() and df["label_fraud"].isna().all()
    full = load_uci(keep_duplicates=True)
    # The published 30/70 protocol needs every row, in file order, as raw text
    assert len(full) == full.attrs["cleaning_log"]["raw_rows"]
    assert full["text"].str.contains("&lt;#&gt;", regex=False).any()
    assert list(full["id"]) == [f"uci_{i}" for i in range(len(full))]


@pytest.mark.parametrize("name, file_name", [
    ("ceas_08", "CEAS_08.csv"), ("spamassassin", "SpamAssasin.csv"),
])
def test_spam_email_labels(name, file_name):
    if not (zenodo_dir / file_name).exists():
        pytest.skip(f"{file_name} not downloaded")
    versions = {strip: load_spam_email(name, strip=strip) for strip in (False, True)}
    # E4 compares the same emails with and without stripping
    assert list(versions[False]["id"]) == list(versions[True]["id"])
    assert list(versions[False]["template"]) == list(versions[True]["template"])
    assert (versions[False]["text"] != versions[True]["text"]).any()
    for df in versions.values():
        assert list(df.columns) == COLUMNS
        assert df["id"].is_unique
        assert df["label_spam"].notna().all()
        assert (df["label_spam"] == (df["label_orig"] == "spam")).all()
        assert df["label_fraud"].isna().all()
        assert not df["text"].map(dedup_key).duplicated().any()
