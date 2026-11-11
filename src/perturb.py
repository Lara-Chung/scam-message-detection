"""Rule-based text perturbations for the robustness test E5 (EE6405 proposal v2, 4.4).

Three types of surface change, each applied to 30% of the eligible words of a message:
  P1  character substitution:      account -> acc0unt, verify -> ver1fy
  P2  inserted spaces or dots:     verify  -> v e r i f y  or  v.e.r.i.f.y
  P3  Unicode look-alikes or a zero-width character:
                                   account -> аccount (Cyrillic a), ver​ify
P2 and P3 pick one of their two forms per word at random. Each form can also be used on its
own (P2_space, P2_dot, P3_lookalike, P3_zero_width), because the D1 defence
(src/normalise.py) undoes P2_space, P3_lookalike and P3_zero_width but not P2_dot or P1
(proposal 4.5), and the DistilBERT tokenizer may already drop zero-width characters.

Eligible words are runs of 4 or more ASCII letters that are not English stop words
(NLTK's list) and are not part of a URL or email address. The number of words changed is
round(0.3 x eligible) (Python rounding), and at least 1 when the message has any eligible word.
P1 and P3_lookalike change one letter per word, and only words with a letter they can
change are picked. The rules change only the surface form, so the true label is kept.

Results are reproducible: pass the same seed and message key (for example the sample_id)
to get the same change.

Usage:
    from perturb import perturb, perturb_with_changes, KINDS
    perturb("Please verify your account now", "P1", key="msg_1")
    python perturb.py                  # prints one example of every kind

Outputs: the changed text; perturb_with_changes() also returns the number of eligible
and changed words.
"""
from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RATE = 0.3
MIN_LEN = 4
KINDS = ["P1", "P2", "P2_space", "P2_dot", "P3", "P3_lookalike", "P3_zero_width"]

# NLTK's English stop-word list (179 words), copied here to avoid the dependency. Only words of
# 4+ letters matter. scikit-learn's list is not used: it contains scam words such as "call" and "bill".
STOP_WORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves
he him his himself she she's her hers herself it it's its itself they them their theirs themselves
what which who whom this that that'll these those am is are was were be been being have has had
having do does did doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both each few more most other
some such no nor not only own same so than too very s t can will just don don't should should've
now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn
hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn
shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())

WORD_RE = re.compile(r"[A-Za-z]+")
# URLs and email addresses are left untouched: changing them would change where a link goes
# (bounded repeats keep the regex linear on very long emails, e.g. base64 attachments)
PROTECTED_RE = re.compile(r"(?:https?://|www\.)\S+|[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63})+"
                          r"|\b[\w-]{1,63}\.(?:com|net|org|co|uk|xyz|info|biz)\b\S*", re.IGNORECASE)

# P1: letter -> digit or symbol that reads the same
P1_MAP = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "5", "l": "1", "t": "7", "b": "8", "g": "9",
          "A": "4", "E": "3", "I": "1", "O": "0", "S": "5", "T": "7", "B": "8"}
# P3: Latin letter -> Cyrillic look-alike (all of them are undone by src/normalise.py)
P3_MAP = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у",
          "x": "х", "i": "і", "j": "ј", "s": "ѕ", "h": "һ",
          "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н",
          "O": "О", "P": "Р", "C": "С", "T": "Т", "X": "Х"}
ZERO_WIDTH = "​"


# --------------------------------------------------------------------------
# Word-level changes
# --------------------------------------------------------------------------
def _swap_one(word: str, mapping: dict[str, str], rng: random.Random) -> str:
    """Replace one random letter of word that has an entry in mapping."""
    pos = [i for i, c in enumerate(word) if c in mapping]
    i = rng.choice(pos)
    return word[:i] + mapping[word[i]] + word[i + 1:]


def p1(word: str, rng: random.Random) -> str:
    """P1: one letter becomes a digit or symbol (acc0unt)."""
    return _swap_one(word, P1_MAP, rng)


def p2_space(word: str, rng: random.Random) -> str:
    """P2, spaced form: v e r i f y."""
    return " ".join(word)


def p2_dot(word: str, rng: random.Random) -> str:
    """P2, dotted form: v.e.r.i.f.y."""
    return ".".join(word)


def p3_lookalike(word: str, rng: random.Random) -> str:
    """P3, look-alike form: one Latin letter becomes its Cyrillic twin."""
    return _swap_one(word, P3_MAP, rng)


def p3_zero_width(word: str, rng: random.Random) -> str:
    """P3, zero-width form: a zero-width space inside the word."""
    i = rng.randint(1, len(word) - 1)
    return word[:i] + ZERO_WIDTH + word[i:]


WordFn = Callable[[str, random.Random], str]
SINGLE: dict[str, WordFn] = {"P1": p1, "P2_space": p2_space, "P2_dot": p2_dot,
                             "P3_lookalike": p3_lookalike, "P3_zero_width": p3_zero_width}
MIXED: dict[str, list[WordFn]] = {"P2": [p2_space, p2_dot], "P3": [p3_lookalike, p3_zero_width]}


def _can_change(word: str, fn: WordFn) -> bool:
    """Whether fn has something to change in word (P1 and P3_lookalike need a mapped letter)."""
    if fn is p1:
        return any(c in P1_MAP for c in word)
    if fn is p3_lookalike:
        return any(c in P3_MAP for c in word)
    return True


# --------------------------------------------------------------------------
# Message-level perturbation
# --------------------------------------------------------------------------
def eligible_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the eligible words: 4+ letters, not a stop word, not in a URL or email."""
    protected = [m.span() for m in PROTECTED_RE.finditer(text)]   # sorted, non-overlapping
    spans, j = [], 0
    for m in WORD_RE.finditer(text):
        while j < len(protected) and protected[j][1] <= m.start():
            j += 1
        if j < len(protected) and protected[j][0] <= m.start():
            continue                                               # inside a URL or email
        w = m.group(0)
        if len(w) >= MIN_LEN and w.lower() not in STOP_WORDS:
            spans.append(m.span())
    return spans


def perturb_with_changes(text: str, kind: str, rate: float = RATE, seed: int = 0,
                         key: Optional[str] = None) -> tuple[str, int, int]:
    """Apply perturbation kind to round(rate x eligible) words (at least 1 if any are eligible).

    key (for example the sample_id) and seed make the result reproducible.
    Returns (changed text, number of eligible words, number of changed words).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    rng = random.Random(f"{seed}|{kind}|{key if key is not None else text}")
    spans = eligible_spans(text)
    fns = MIXED.get(kind) or [SINGLE[kind]]
    # draw the form per word first, then keep only the words that form can change
    choices = [(s, rng.choice(fns)) for s in spans]
    choices = [(s, fn) for s, fn in choices if _can_change(text[s[0]:s[1]], fn)]
    n_change = min(len(choices), max(1, round(rate * len(spans)))) if spans else 0
    picked = sorted(rng.sample(choices, n_change), key=lambda c: c[0][0])
    out, last = [], 0
    for (a, b), fn in picked:
        out.append(text[last:a])
        out.append(fn(text[a:b], rng))
        last = b
    out.append(text[last:])
    return "".join(out), len(spans), n_change


def perturb(text: str, kind: str, rate: float = RATE, seed: int = 0, key: Optional[str] = None) -> str:
    """Return text with perturbation kind applied (see perturb_with_changes)."""
    return perturb_with_changes(text, kind, rate, seed, key)[0]


# --------------------------------------------------------------------------
# E5 test sets
# --------------------------------------------------------------------------
def e5_test_sets(test: "pd.DataFrame", seed: int = 0, text_col: str = "text"
                 ) -> tuple[dict[str, "pd.DataFrame"], "pd.DataFrame"]:
    """Build the E5 evaluation sets from one clean test set (proposal 5, table row E5).

    Returns ({name: DataFrame}, log). Names are "clean", every kind in KINDS, and each of
    these with "+D1" (normalise() applied after the change; "clean+D1" checks that D1 does
    not create false alarms on clean text). The log has one row per message and kind with
    sample_id, kind, the original and changed text and the eligible / changed word counts.
    """
    import pandas as pd
    try:
        from normalise import normalise
    except ImportError:          # inside the group repo
        from src.normalise import normalise

    sets = {"clean": test}
    log = []
    for kind in KINDS:
        out = [perturb_with_changes(t, kind, seed=seed, key=str(i))
               for t, i in zip(test[text_col].fillna(""), test["id"])]
        d = test.copy()
        d[text_col] = [o[0] for o in out]
        sets[kind] = d
        log.append(pd.DataFrame({"sample_id": test["id"].values, "kind": kind,
                                 "original": test[text_col].values, "text": d[text_col].values,
                                 "n_eligible": [o[1] for o in out], "n_changed": [o[2] for o in out]}))
    for name in list(sets):
        d = sets[name].copy()
        d[text_col] = d[text_col].fillna("").map(normalise)
        sets[f"{name}+D1"] = d
    return sets, pd.concat(log, ignore_index=True)


if __name__ == "__main__":
    example = "URGENT: your bank account has been suspended. Verify now at http://secure-bank.xyz"
    for k in KINDS:
        t, n_elig, n_chg = perturb_with_changes(example, k, key="demo")
        print(f"{k:14s} {n_chg}/{n_elig}  {t!r}")
