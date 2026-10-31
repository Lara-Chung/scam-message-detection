"""Text normalisation defence (D1) against character-level obfuscation.

normalise(text) undoes four kinds of obfuscation in English messages:
invisible format characters (zero-width spaces and joiners, soft hyphens,
direction marks), full-width and other compatibility letters, Unicode
look-alike letters (Cyrillic/Greek), and spaced-out letters ("v e r i f y").

It does not undo digit-for-letter swaps ("acc0unt"), because a digit map would
also change prices and phone numbers, nor dotted letters ("v.e.r.i.f.y").
Digits and symbols ("\u00bd", "\u00b2", "\u2122") are left unchanged for the same reason.

Limits: designed for English text. The look-alike map changes Cyrillic and
Greek letters, so it is not suitable for messages written in those scripts.
Adjacent spaced-out words are joined into one ("v e r i f y n o w" -> "verifynow").
"""

import re
import unicodedata

# Cyrillic and Greek letters that look like Latin letters
LOOKALIKES = str.maketrans({
    # Cyrillic
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u0455": "s",
    "\u0501": "d", "\u04cf": "l", "\u04bb": "h", "\u051d": "w", "\u051b": "q",
    "\u0475": "v",
    "\u043a": "k",                                                    # lowercase ka
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0423": "Y", "\u0425": "X", "\u0406": "I", "\u0408": "J", "\u0405": "S",
    # Latin-extended
    "\u0131": "i", "\u0261": "g", "\u0251": "a",
    # Greek
    "\u03b1": "a", "\u03bf": "o", "\u03c1": "p", "\u03bd": "v", "\u03b9": "i",
    "\u03ba": "k", "\u03c5": "u", "\u03f2": "c", "\u03f3": "j",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X",
})

# Four or more single letters separated by single spaces: "v e r i f y".
# Three ("I a m", "A B C") are left alone, as they are often real words or initials.
SPACED_LETTERS_RE = re.compile(r"(?<![A-Za-z])[A-Za-z](?: [A-Za-z]){3,}(?![A-Za-z])")


def normalise(text: str) -> str:
    """Return text with invisible characters, look-alike letters and spaced-out letters undone."""
    # Invisible format characters (Unicode category Cf), e.g. U+200B, U+00AD, U+200E
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    # NFKC on letters and spaces only: full-width and math letters become ASCII,
    # a no-break space becomes a space; digits and symbols are not touched
    text = "".join(
        unicodedata.normalize("NFKC", c) if unicodedata.category(c)[0] in "LZ" else c
        for c in text
    )
    text = text.translate(LOOKALIKES)
    return SPACED_LETTERS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
