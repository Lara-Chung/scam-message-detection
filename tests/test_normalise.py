import pytest

from src.normalise import normalise


@pytest.mark.parametrize("obfuscated, expected", [
    ("Verify your \u0430cc\u043eunt", "Verify your account"),        # Cyrillic a, o
    ("\u03a1ay now via \u03a1ay\u03a1al", "Pay now via PayPal"),        # Greek Rho
    ("Cl\u0456ck \u04bbere", "Click here"),                            # Cyrillic i, h
    ("ver\u200bify your acc\u200dount", "verify your account"),        # zero-width space/joiner
    ("\ufeffClick here", "Click here"),                                # byte order mark
    ("ver\u00adify now", "verify now"),                                # soft hyphen
    ("ver\u200eify now", "verify now"),                                # left-to-right mark
    ("Please v e r i f y now", "Please verify now"),
    ("U R G E N T: call us", "URGENT: call us"),
    ("v e r i f y2", "verify2"),
    ("\uff46\uff52\uff45\uff45 prize", "free prize"),                  # full-width letters
    ("Hello\u00a0there", "Hello there"),                               # no-break space
])
def test_undoes_obfuscation(obfuscated, expected):
    assert normalise(obfuscated) == expected


@pytest.mark.parametrize("text", [
    "I a m happy to see you",
    "Meet A B C at noon",
    "See you at 5, ok?",
    "Your parcel is waiting.",
])
def test_keeps_normal_text(text):
    assert normalise(text) == text


@pytest.mark.parametrize("text", [
    "Call 0871 234 5678 within 24 hours",
    "Pay \u00a3200 to restore access",
    "Pay \u00a35\u00bd now",                   # 5 and a half pounds
    "A 10 m\u00b2 flat",
    "Win \u00a31,000\u2122",
    "Visit http://bank-verify.com/login?id=42",
    "Email support@bank.co.uk",
    "Your acc0unt is locked",                  # digit swaps are left to the char TF-IDF defence
])
def test_keeps_entities_digits_and_symbols(text):
    assert normalise(text) == text


@pytest.mark.parametrize("text", [
    "Verify your \u0430cc\u043eunt",
    "v e r i f y\u200b now",
    "\uff46\uff52\uff45\uff45 prize",
    "cafe\u200b\u0301",                        # zero-width character before a combining mark
])
def test_is_idempotent(text):
    once = normalise(text)
    assert normalise(once) == once
