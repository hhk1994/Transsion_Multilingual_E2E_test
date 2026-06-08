"""Prepare ASR hypotheses for TN using original written reference spans."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from lib.text_match import is_char_level_wer_language

_DIGIT_RUN = re.compile(r"\d+")
# Latin letter suffix glued to a number in written text (e.g. 300ETF, 5G).
_ASCII_ALNUM_SUFFIX = re.compile(r"^[A-Za-z0-9]+")


class TNRunner(Protocol):
    def normalize(self, text: str) -> str: ...


def default_source_txt(e2e_root: Path, language: str) -> Path:
    """Guess pre-TN source file for a locale."""
    lang = (language or "en").lower().split("-")[0]
    for candidate in (
        e2e_root / "input" / f"{lang}_1000_sample_sent.txt",
        e2e_root / "trassion_test" / f"{lang}_1000_sample_sent.txt",
    ):
        if candidate.is_file():
            return candidate
    return e2e_root / "input" / f"{lang}_1000_sample_sent.txt"


def load_written_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _extend_ascii_alnum_suffix(text: str, end: int) -> int:
    while end < len(text):
        ch = text[end]
        if ch.isascii() and ch.isalnum():
            end += 1
            continue
        break
    return end


def written_number_spans(written: str) -> list[tuple[int, int, str]]:
    """Return (start, end, snippet) for each digit run plus attached ASCII alnum suffix."""
    spans: list[tuple[int, int, str]] = []
    for match in _DIGIT_RUN.finditer(written):
        start = match.start()
        end = _extend_ascii_alnum_suffix(written, match.end())
        spans.append((start, end, written[start:end]))
    return spans


def merge_written_number_spans(hyp: str, written: str) -> str:
    """Splice written number snippets into hyp so TN sees the same digit context as ref.

    When ASR keeps Arabic digits but garbles nearby Latin (``300F`` vs ``300ETF``), TN may
    read the number as a large cardinal. Replacing each hyp digit span with the matching
    written span (same order, same count) aligns TN input with offline reference TN.
    """
    if not hyp or not written:
        return hyp

    ref_spans = written_number_spans(written)
    if not ref_spans:
        return hyp

    hyp_matches = list(_DIGIT_RUN.finditer(hyp))
    if len(hyp_matches) != len(ref_spans):
        return hyp

    parts: list[str] = []
    cursor = 0
    for (_, _, snippet), match in zip(ref_spans, hyp_matches):
        h_start = match.start()
        h_end = _extend_ascii_alnum_suffix(hyp, match.end())
        hyp_span = hyp[h_start:h_end]
        hyp_digits = hyp[match.start() : match.end()]
        written_digits = snippet[: len(hyp_digits)]
        parts.append(hyp[cursor:h_start])
        if hyp_digits != written_digits:
            # ASR read different digits than written source — do not overwrite.
            parts.append(hyp_span)
        elif snippet.startswith(hyp_digits) and len(snippet) > len(hyp_span):
            # Same digits; extend missing ASCII suffix from written (300F -> 300ETF).
            parts.append(snippet)
        else:
            parts.append(hyp_span)
        cursor = h_end
    parts.append(hyp[cursor:])
    return "".join(parts)


def should_merge_written_spans(language: str | None) -> bool:
    return is_char_level_wer_language(language)


def prepare_hyp_for_tn(
    hyp_raw: str,
    written: str,
    *,
    language: str | None,
    tn: TNRunner | None,
) -> tuple[str, str]:
    """Return (text_fed_to_tn, hyp_after_tn)."""
    merged = (
        merge_written_number_spans(hyp_raw, written)
        if should_merge_written_spans(language) and written
        else hyp_raw
    )
    hyp_tn = tn.normalize(merged) if tn is not None else merged
    return merged, hyp_tn
