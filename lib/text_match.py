"""Text normalization helpers for WER/CER comparison."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# Strip common punctuation; keep Bengali and word chars.
_PUNCT = re.compile(r"[^\w\s\u0980-\u09FF]", re.UNICODE)


def normalize_for_wer(text: str) -> str:
    """Normalize reference/hypothesis before jiwer."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
