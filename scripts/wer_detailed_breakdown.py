#!/usr/bin/env python3
"""Build detailed WER breakdown from wer_per_utt.csv.

Outputs:
  - detailed_wer_report.json
  - confusion_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

import jiwer

E2E_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(E2E_ROOT))

from lib.text_match import normalize_for_wer  # noqa: E402


_INDEP_VOWELS = {
    "অ": "o",
    "আ": "a",
    "ই": "i",
    "ঈ": "ii",
    "উ": "u",
    "ঊ": "uu",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
}

_CONSONANTS = {
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "c",
    "ছ": "ch",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "ny",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "ph",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "y",
    "র": "r",
    "ল": "l",
    "শ": "sh",
    "ষ": "sh",
    "স": "s",
    "হ": "h",
    "ড়": "r",
    "ঢ়": "rh",
    "য়": "y",
}

_VOWEL_SIGNS = {
    "া": "a",
    "ি": "i",
    "ী": "ii",
    "ু": "u",
    "ূ": "uu",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
}

_MARKS = {"ঁ": "n", "ং": "ng", "ঃ": "h", "ৎ": "t"}
_VIRAMA = "্"
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def bn_to_latin(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _INDEP_VOWELS:
            out.append(_INDEP_VOWELS[ch])
            i += 1
            continue
        if ch in _CONSONANTS:
            base = _CONSONANTS[ch]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == _VIRAMA:
                out.append(base)
                i += 2
                continue
            if nxt in _VOWEL_SIGNS:
                out.append(base + _VOWEL_SIGNS[nxt])
                i += 2
                continue
            out.append(base + "o")
            i += 1
            continue
        if ch in _VOWEL_SIGNS:
            out.append(_VOWEL_SIGNS[ch])
            i += 1
            continue
        if ch in _MARKS:
            out.append(_MARKS[ch])
            i += 1
            continue
        out.append(ch.translate(_DIGITS))
        i += 1
    return "".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detailed WER stats from wer_per_utt.csv")
    p.add_argument("--wer-csv", type=Path, required=True, help="Path to wer_per_utt.csv")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as wer-csv)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    wer_csv = args.wer_csv.resolve()
    if not wer_csv.is_file():
        print(f"wer csv not found: {wer_csv}", file=sys.stderr)
        return 1

    out_dir = (args.output_dir.resolve() if args.output_dir else wer_csv.parent.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with wer_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("No rows found in wer csv.", file=sys.stderr)
        return 2

    sub_total = 0
    del_total = 0
    ins_total = 0
    hit_total = 0
    ref_words_total = 0
    confusion_counter: Counter[tuple[str, str, str]] = Counter()

    for r in rows:
        ref = normalize_for_wer(r.get("ref", ""))
        hyp = normalize_for_wer(r.get("hyp", ""))
        m = jiwer.process_words(ref, hyp)

        sub_total += m.substitutions
        del_total += m.deletions
        ins_total += m.insertions
        hit_total += m.hits
        ref_words_total += len(ref.split())

        ref_words = ref.split()
        hyp_words = hyp.split()
        for chunks in m.alignments:
            for ch in chunks:
                op = ch.type
                ref_seg = " ".join(ref_words[ch.ref_start_idx : ch.ref_end_idx])
                hyp_seg = " ".join(hyp_words[ch.hyp_start_idx : ch.hyp_end_idx])
                if op == "equal":
                    continue
                if op == "delete":
                    hyp_seg = "<eps>"
                elif op == "insert":
                    ref_seg = "<eps>"
                confusion_counter[(op, ref_seg, hyp_seg)] += 1

    err_total = sub_total + del_total + ins_total
    wer_total = (err_total / ref_words_total) if ref_words_total else 0.0

    confusion_rows = []
    for (op, ref_seg, hyp_seg), c in confusion_counter.most_common():
        confusion_rows.append(
            {
                "op_type": op,
                "ref": ref_seg,
                "hyp": hyp_seg,
                "count": c,
                "ref_latin": bn_to_latin(ref_seg),
                "hyp_latin": bn_to_latin(hyp_seg),
            }
        )

    confusion_csv = out_dir / "confusion_pairs.csv"
    with confusion_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["op_type", "ref", "hyp", "count", "ref_latin", "hyp_latin"],
        )
        writer.writeheader()
        writer.writerows(confusion_rows)

    report = {
        "n_utterances": len(rows),
        "total_ref_words": ref_words_total,
        "total_errors": err_total,
        "total_wer": round(wer_total, 6),
        "error_breakdown": {
            "substitution": {
                "count": sub_total,
                "pct_of_ref_words": round((sub_total / ref_words_total) if ref_words_total else 0.0, 6),
            },
            "deletion": {
                "count": del_total,
                "pct_of_ref_words": round((del_total / ref_words_total) if ref_words_total else 0.0, 6),
            },
            "insertion": {
                "count": ins_total,
                "pct_of_ref_words": round((ins_total / ref_words_total) if ref_words_total else 0.0, 6),
            },
        },
        "confusion_pair_count": len(confusion_rows),
        "outputs": {
            "confusion_pairs_csv": str(confusion_csv),
        },
    }

    report_json = out_dir / "detailed_wer_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[detail] total_wer={report['total_wer']}")
    print(
        "[detail] S/D/I="
        f"{sub_total}/{del_total}/{ins_total} "
        f"(ref_words={ref_words_total})"
    )
    print(f"[detail] report: {report_json}")
    print(f"[detail] confusion: {confusion_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

