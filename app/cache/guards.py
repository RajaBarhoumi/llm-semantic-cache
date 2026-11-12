"""Text normalization and the entity guard.

Embeddings capture meaning, which is exactly why they are dangerous for a
cache: "invoice for order 1001" and "invoice for order 1002" are almost
identical vectors, yet returning one answer for the other is a bug. The guard
extracts "anchors" (numbers, ids, emails, capitalized names) and only allows a
semantic hit when each question's anchors also appear in the other one.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_HAS_DIGIT = re.compile(r"\b[a-z0-9]*\d[a-z0-9]*(?:-[a-z0-9]+)*\b", re.IGNORECASE)
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z'-]*")
_IGNORED_CAPITALIZED = {"i", "i'm", "i've", "i'd", "i'll"}


def normalize(text: str) -> str:
    """Lowercase, unify quotes and spaces, drop trailing punctuation: the exact-match key."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" ?!.")


def tokens(text: str) -> set[str]:
    lowered = text.lower().replace("’", "'")
    return set(_WORD.findall(lowered)) | set(_EMAIL.findall(lowered))


def anchors(text: str) -> set[str]:
    """Tokens that change the meaning of an answer: numbers, ids, emails, names."""
    found = {m.lower() for m in _HAS_DIGIT.findall(text)}
    found |= {m.lower() for m in _EMAIL.findall(text)}
    # Capitalized words that do not start the sentence: likely names, places, brands.
    for match in _CAPITALIZED.finditer(text):
        before = text[: match.start()].rstrip()
        sentence_start = not before or before[-1] in ".?!"
        word = match.group().lower().replace("’", "'")
        if not sentence_start and word not in _IGNORED_CAPITALIZED:
            found.add(word)
    return found


def compatible(question_a: str, question_b: str) -> bool:
    """True if each question's anchors appear (case-insensitively) in the other one."""
    return anchors(question_a) <= tokens(question_b) and anchors(question_b) <= tokens(question_a)
