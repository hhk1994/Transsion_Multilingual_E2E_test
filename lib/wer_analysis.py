"""Corpus-level WER breakdown, error cases, and confusion-pair aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import jiwer

from lib.text_match import is_char_level_wer_language


def error_unit_for_language(language: str) -> str:
    return "char" if is_char_level_wer_language(language) else "word"


def corpus_metric_key(unit: str) -> str:
    """JSON field name for the primary corpus error rate."""
    return "cer" if unit == "char" else "wer"


def corpus_error_rate(summary: dict[str, Any]) -> float:
    """Read the primary corpus error rate from a ``corpus_wer`` summary dict."""
    unit = str(summary.get("error_unit", "word"))
    key = corpus_metric_key(unit)
    if key in summary:
        return float(summary[key])
    # Backward compatibility for reports that only stored ``wer``.
    if unit == "char" and "wer" in summary:
        return float(summary["wer"])
    return float(summary["wer"])


def _process_batch(refs: list[str], hyps: list[str], unit: str):
    if unit == "char":
        return jiwer.process_characters(refs, hyps)
    return jiwer.process_words(refs, hyps)


def _process_one(ref: str, hyp: str, unit: str):
    return _process_batch([ref], [hyp], unit)


def _error_rate(count: int, n_ref: int) -> float:
    return round(count / n_ref, 6) if n_ref > 0 else 0.0


def _pair_type(ref: str, hyp: str) -> str:
    if ref and hyp:
        return "substitution"
    if ref and not hyp:
        return "deletion"
    if not ref and hyp:
        return "insertion"
    return "other"


def _pair_key(ref_part: str, hyp_part: str, chunk_type: str) -> tuple[str, str] | None:
    if chunk_type == "substitute":
        return ref_part, hyp_part
    if chunk_type == "delete":
        return ref_part, ""
    if chunk_type == "insert":
        return "", hyp_part
    return None


def _extract_alignment_pairs(
    ref_tokens: list[str],
    hyp_tokens: list[str],
    alignment,
) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for chunk in alignment:
        ref_part = "".join(ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx])
        hyp_part = "".join(hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx])
        key = _pair_key(ref_part, hyp_part, chunk.type)
        if key is not None:
            pairs.append((key[0], key[1], _pair_type(key[0], key[1])))
    return pairs


def _utterance_error_rate(output, unit: str) -> float:
    return float(output.cer if unit == "char" else output.wer)


def utterance_error_rates(ref: str, hyp: str, *, unit: str) -> tuple[float, float]:
    """Return (wer, cer) using the language-appropriate primary rate in ``wer``.

    For char-level languages (e.g. zh), ``wer`` is CER so per-utterance and corpus
    summaries stay consistent. Word-level languages keep standard jiwer WER/CER.
    """
    if not ref:
        empty = 0.0 if not hyp else 1.0
        return empty, empty
    word_wer = float(jiwer.wer(ref, hyp))
    char_cer = float(jiwer.cer(ref, hyp))
    if unit == "char":
        return char_cer, char_cer
    return word_wer, char_cer


def _corpus_summary(output, unit: str, n_utterances: int) -> dict[str, Any]:
    hits = int(output.hits)
    substitutions = int(output.substitutions)
    deletions = int(output.deletions)
    insertions = int(output.insertions)
    n_ref = hits + substitutions + deletions
    rate = round(_utterance_error_rate(output, unit), 6)
    metric = corpus_metric_key(unit)

    summary: dict[str, Any] = {
        "error_unit": unit,
        "metric": metric,
        "token_label": "characters" if unit == "char" else "words",
        "n_utterances": n_utterances,
        "n_ref_tokens": n_ref,
        "hits": hits,
        "substitutions": {
            "count": substitutions,
            "rate": _error_rate(substitutions, n_ref),
        },
        "deletions": {
            "count": deletions,
            "rate": _error_rate(deletions, n_ref),
        },
        "insertions": {
            "count": insertions,
            "rate": _error_rate(insertions, n_ref),
        },
    }
    summary[metric] = rate
    return summary


def build_error_cases(
    rows: list[dict[str, Any]],
    refs: list[str],
    hyps: list[str],
    *,
    unit: str,
) -> list[dict[str, Any]]:
    """All utterances with alignment errors, sorted most severe first."""
    error_rows: list[dict[str, Any]] = []
    for row, ref, hyp in zip(rows, refs, hyps):
        out = _process_one(ref, hyp, unit)
        if out.substitutions + out.deletions + out.insertions == 0:
            continue
        enriched = dict(row)
        enriched["error_rate"] = round(_utterance_error_rate(out, unit), 6)
        enriched["error_unit"] = unit
        enriched["substitutions"] = int(out.substitutions)
        enriched["deletions"] = int(out.deletions)
        enriched["insertions"] = int(out.insertions)
        error_rows.append(enriched)

    error_rows.sort(
        key=lambda r: (
            -r["error_rate"],
            -r["substitutions"] - r["deletions"] - r["insertions"],
            r.get("line_no", 0),
        )
    )
    return error_rows


def build_confusion_pairs(
    rows: list[dict[str, Any]],
    refs: list[str],
    hyps: list[str],
    *,
    unit: str,
) -> list[dict[str, Any]]:
    """Confusion pairs with per-pair utterance id lists."""
    pair_total: Counter[tuple[str, str]] = Counter()
    pair_case_ids: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row, ref, hyp in zip(rows, refs, hyps):
        out = _process_one(ref, hyp, unit)
        ref_tokens = out.references[0]
        hyp_tokens = out.hypotheses[0]
        utt_id = str(row.get("utt_id", ""))
        for ref_part, hyp_part, _ptype in _extract_alignment_pairs(
            ref_tokens, hyp_tokens, out.alignments[0]
        ):
            key = (ref_part, hyp_part)
            pair_total[key] += 1
            if utt_id:
                pair_case_ids[key].add(utt_id)

    confusion_pairs: list[dict[str, Any]] = []
    for (ref, hyp), count in sorted(
        pair_total.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        confusion_pairs.append(
            {
                "reference": ref,
                "hypothesis": hyp,
                "count": count,
                "type": _pair_type(ref, hyp),
                "case_ids": sorted(pair_case_ids[(ref, hyp)]),
            }
        )
    return confusion_pairs


def build_wer_analysis(
    rows: list[dict[str, Any]],
    refs: list[str],
    hyps: list[str],
    *,
    unit: str,
) -> dict[str, Any]:
    """Corpus summary, all error cases, and confusion pairs for one evaluation run."""
    if len(rows) != len(refs) or len(refs) != len(hyps):
        raise ValueError("rows/refs/hyps length mismatch")
    if not rows:
        raise ValueError("empty evaluation set")

    corpus_output = _process_batch(refs, hyps, unit)
    corpus_wer = _corpus_summary(corpus_output, unit, len(rows))
    error_cases = build_error_cases(rows, refs, hyps, unit=unit)
    confusion_pairs = build_confusion_pairs(rows, refs, hyps, unit=unit)

    corpus_wer["n_error_cases"] = len(error_cases)
    corpus_wer["n_confusion_pairs"] = len(confusion_pairs)

    return {
        "corpus_wer": corpus_wer,
        "top_wer": error_cases,
        "confusion_pairs": confusion_pairs,
    }
