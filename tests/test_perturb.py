"""pytest checks for the E5 perturbation rules (proposal 4.4 and 4.7).

Usage:
    pytest tests/test_perturb.py      # from the repository root
"""
from __future__ import annotations

import pytest

from src.perturb import KINDS, ZERO_WIDTH, eligible_spans, perturb, perturb_with_changes

from src.normalise import normalise

MSG = "URGENT: your bank account has been suspended. Verify now at http://secure-bank.xyz"


def test_eligible_words() -> None:
    """Only 4+ letter non-stop words outside URLs are eligible."""
    words = [MSG[a:b] for a, b in eligible_spans(MSG)]
    assert words == ["URGENT", "bank", "account", "suspended", "Verify"]


@pytest.mark.parametrize("kind", KINDS)
def test_rate_and_reproducible(kind: str) -> None:
    """30% of 5 eligible words -> 2 changed; same seed and key give the same text."""
    t1, n_elig, n_chg = perturb_with_changes(MSG, kind, key="k")
    assert (n_elig, n_chg) == (5, 2)
    assert t1 != MSG
    assert perturb(MSG, kind, key="k") == t1


@pytest.mark.parametrize("kind", KINDS)
def test_url_and_short_words_untouched(kind: str) -> None:
    """The URL and the short / stop words keep their exact form."""
    out = perturb(MSG, kind, key="k")
    assert out.endswith("http://secure-bank.xyz")
    assert " your " in out and " now at " in out


def test_at_least_one_word() -> None:
    """A message with one eligible word still gets one change."""
    assert perturb("call me", "P2_dot") == "c.a.l.l me"


def test_no_eligible_words() -> None:
    """Messages without eligible words are returned unchanged."""
    assert perturb("ok see you at 5", "P1") == "ok see you at 5"


def test_p1_changes_one_letter() -> None:
    """P1 swaps exactly one letter of the word for a digit or symbol."""
    out = perturb("verify", "P1")
    assert len(out) == 6 and sum(a != b for a, b in zip(out, "verify")) == 1


@pytest.mark.parametrize("kind", ["P2_space", "P3_lookalike", "P3_zero_width", "P3"])
def test_normalise_undoes(kind: str) -> None:
    """D1 restores spaced-out letters, look-alikes and zero-width characters (4.5)."""
    assert normalise(perturb("Please verify", kind)) == "Please verify"


@pytest.mark.parametrize("kind", ["P1", "P2_dot"])
def test_normalise_does_not_undo(kind: str) -> None:
    """D1 leaves digit swaps and dotted words as they are (4.5)."""
    assert normalise(perturb("Please verify", kind)) != "Please verify"


def test_zero_width_inside_word() -> None:
    """The zero-width space goes inside the word, not at its ends."""
    out = perturb("verify", "P3_zero_width")
    assert ZERO_WIDTH in out[1:-1] and out.replace(ZERO_WIDTH, "") == "verify"


def test_unknown_kind() -> None:
    """An unknown kind is an error, not a silent no-op."""
    with pytest.raises(ValueError):
        perturb(MSG, "P9")
