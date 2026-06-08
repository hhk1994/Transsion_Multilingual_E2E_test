#!/usr/bin/env python3
"""Rebuild unified wer_report.json from wer_per_utt.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

E2E_ROOT = Path(__file__).resolve().parents[1]
if str(E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(E2E_ROOT))

from lib.hyp_tn_prep import (  # noqa: E402
    default_source_txt,
    load_written_lines,
    prepare_hyp_for_tn,
    should_merge_written_spans,
)
from lib.text_match import normalize_texts_for_wer  # noqa: E402
from lib.utt_exclusions import load_exclude_utt_ids, partition_rows  # noqa: E402
from lib.wer_analysis import (  # noqa: E402
    build_wer_analysis,
    corpus_error_rate,
    corpus_metric_key,
    error_unit_for_language,
    utterance_error_rates,
)


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild unified wer_report.json")
    p.add_argument(
        "--asr-dir",
        type=Path,
        required=True,
        help="ASR output dir containing wer_per_utt.csv and wer_report.json",
    )
    p.add_argument(
        "--language",
        default="",
        help="Language code for char/word unit (default: read from wer_report.json)",
    )
    p.add_argument(
        "--error-unit",
        choices=["auto", "word", "char"],
        default="auto",
        help="Token unit for corpus errors (default: auto by language)",
    )
    p.add_argument(
        "--exclude-utt",
        action="append",
        default=[],
        metavar="UTT_ID",
        help="Utterance id to exclude from WER (repeatable)",
    )
    p.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="Text file with one utt_id per line (# comments allowed)",
    )
    p.add_argument(
        "--source-txt",
        type=Path,
        default=None,
        help="Pre-TN written references for hyp span merge + optional re-TN",
    )
    p.add_argument(
        "--tn-bin",
        type=Path,
        default=None,
        help="TN binary to re-run hyp TN after span merge (default: from wer_report.json)",
    )
    p.add_argument(
        "--no-retn-hyp",
        action="store_true",
        help="Do not re-run TN on hyp_raw after written-span merge",
    )
    p.add_argument(
        "--merge-written-spans",
        action="store_true",
        help="Splice written digit spans into hyp before re-TN",
    )
    return p.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("line_no"):
            row["line_no"] = int(row["line_no"])
        for key in ("wer", "cer"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
        for key in ("ref_len", "hyp_len"):
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])
    return rows


def main() -> int:
    args = parse_args()
    asr_dir = args.asr_dir.resolve()
    csv_path = asr_dir / "wer_per_utt.csv"
    report_path = asr_dir / "wer_report.json"

    if not csv_path.is_file():
        print(f"Missing {csv_path}", file=sys.stderr)
        return 1

    all_rows = load_rows(csv_path)
    if not all_rows:
        print("Empty wer_per_utt.csv", file=sys.stderr)
        return 2

    exclude_ids = load_exclude_utt_ids(
        utt_ids=args.exclude_utt,
        exclude_file=args.exclude_file,
    )
    rows, excluded_utt_ids = partition_rows(all_rows, exclude_ids)
    if exclude_ids and not rows:
        print("All utterances excluded; nothing to evaluate.", file=sys.stderr)
        return 3
    unknown_ids = exclude_ids - {str(r.get("utt_id", "")) for r in all_rows}
    if unknown_ids:
        print(
            f"[enrich] warning: exclude list has unknown utt_id(s): {sorted(unknown_ids)}",
            file=sys.stderr,
        )

    report: dict = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    language = args.language or report.get("language") or "en"
    unit = (
        error_unit_for_language(language)
        if args.error_unit == "auto"
        else args.error_unit
    )

    merge_written_spans = (
        args.merge_written_spans and should_merge_written_spans(language)
    )
    written_lines: list[str] = []
    source_path: Path | None = None
    if merge_written_spans:
        source_path = (
            args.source_txt.resolve()
            if args.source_txt
            else Path(report["source_txt"])
            if report.get("source_txt")
            else default_source_txt(E2E_ROOT, language)
        )
        if source_path.is_file():
            written_lines = load_written_lines(source_path)
        else:
            print(f"[enrich] WARN: written source not found ({source_path}); span merge disabled")
            merge_written_spans = False

    tn_bin = args.tn_bin or (Path(report["tn_bin"]) if report.get("tn_bin") else None)
    retn_hyp = (
        merge_written_spans
        and not args.no_retn_hyp
        and tn_bin is not None
        and tn_bin.is_file()
        and "hyp_raw" in all_rows[0]
    )
    tn_normalizer = None
    if retn_hyp:
        from scripts.transcribe_and_wer import TNHypNormalizer

        tn_normalizer = TNHypNormalizer(tn_bin.resolve())
        print(f"[enrich] re-TN hyp after written-span merge: {tn_bin}")

    try:
        if retn_hyp:
            for row in all_rows:
                line_no = int(row["line_no"]) if row.get("line_no") else 0
                written = (
                    written_lines[line_no - 1]
                    if merge_written_spans and 1 <= line_no <= len(written_lines)
                    else ""
                )
                hyp_raw = row.get("hyp_raw") or ""
                _merged, hyp_tn = prepare_hyp_for_tn(
                    hyp_raw,
                    written,
                    language=language if merge_written_spans else None,
                    tn=tn_normalizer,
                )
                row["hyp_tn"] = hyp_tn
                row["hyp"] = hyp_tn
    finally:
        if tn_normalizer is not None:
            tn_normalizer.close()

    if merge_written_spans:
        report["hyp_written_span_merge"] = True
        report["source_txt"] = str(source_path)
    if retn_hyp:
        hyp_path = asr_dir / "asr_hyp.tsv"
        hyp_path.write_text(
            "utt_id\tline_no\thypothesis\n"
            + "\n".join(
                f"{r['utt_id']}\t{r.get('line_no', '')}\t{r.get('hyp_tn', '')}"
                for r in all_rows
            )
            + "\n",
            encoding="utf-8",
        )

    refs_all: list[str] = []
    hyps_all: list[str] = []
    for r in all_rows:
        ref_norm, hyp_norm = normalize_texts_for_wer(
            r["ref"],
            r.get("hyp_tn") or r.get("hyp") or "",
            language=language,
        )
        refs_all.append(ref_norm)
        hyps_all.append(hyp_norm)
    for row, ref_norm, hyp_norm in zip(all_rows, refs_all, hyps_all):
        wer, cer = utterance_error_rates(ref_norm, hyp_norm, unit=unit)
        row["wer"] = round(wer, 6)
        row["cer"] = round(cer, 6)
        row["ref_len"] = len(ref_norm)
        row["hyp_len"] = len(hyp_norm)

    eval_rows, _ = partition_rows(all_rows, exclude_ids)
    ref_by_utt = {r["utt_id"]: ref for r, ref in zip(all_rows, refs_all)}
    hyp_by_utt = {r["utt_id"]: hyp for r, hyp in zip(all_rows, hyps_all)}
    eval_refs = [ref_by_utt[r["utt_id"]] for r in eval_rows]
    eval_hyps = [hyp_by_utt[r["utt_id"]] for r in eval_rows]
    wers = [float(r["wer"]) for r in eval_rows]
    cers = [float(r["cer"]) for r in eval_rows]

    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    analysis = build_wer_analysis(eval_rows, eval_refs, eval_hyps, unit=unit)

    n = len(wers)
    wers_sorted = sorted(wers)
    report["corpus_wer"] = analysis["corpus_wer"]
    report["top_wer"] = analysis["top_wer"]
    report["confusion_pairs"] = analysis["confusion_pairs"]
    report["wer"] = {
        "mean": round(sum(wers) / n, 6),
        "p50": round(percentile(wers_sorted, 0.5), 6),
        "p95": round(percentile(wers_sorted, 0.95), 6),
        "max": round(max(wers), 6),
    }
    report["cer"] = {
        "mean": round(sum(cers) / n, 6),
        "p50": round(percentile(sorted(cers), 0.5), 6),
        "p95": round(percentile(sorted(cers), 0.95), 6),
    }
    report["evaluation_scope"] = {
        "n_utterances_total": len(all_rows),
        "n_utterances_evaluated": n,
        "n_utterances_excluded": len(excluded_utt_ids),
        "excluded_utt_ids": excluded_utt_ids,
        "exclude_file": str(args.exclude_file.resolve()) if args.exclude_file else None,
    }

    ordered = {
        "corpus_wer": report.pop("corpus_wer"),
        "evaluation_scope": report.pop("evaluation_scope"),
        "top_wer": report.pop("top_wer"),
        "confusion_pairs": report.pop("confusion_pairs"),
        **report,
    }
    report = ordered
    report["outputs"] = report.get("outputs", {})
    report["outputs"]["wer_csv"] = str(csv_path)
    report["outputs"]["hyp_tsv"] = str(asr_dir / "asr_hyp.tsv")

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (asr_dir / "confusion_pairs.json").write_text(
        json.dumps(report["confusion_pairs"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cw = report["corpus_wer"]
    metric = corpus_metric_key(unit)
    metric_label = metric.upper()
    print(f"[enrich] asr_dir={asr_dir}")
    if excluded_utt_ids:
        print(
            f"[enrich] excluded {len(excluded_utt_ids)} utterances "
            f"(evaluating {n}/{len(all_rows)})"
        )
    print(
        f"[enrich] corpus {metric_label}={corpus_error_rate(cw)} unit={cw['error_unit']} "
        f"tokens={cw['n_ref_tokens']} (S={cw['substitutions']['count']} "
        f"D={cw['deletions']['count']} I={cw['insertions']['count']})"
    )
    print(
        f"[enrich] error cases={cw['n_error_cases']} "
        f"confusion pairs={cw['n_confusion_pairs']}"
    )
    print(
        f"[enrich] utterance {metric_label} mean={report[metric]['mean']} "
        f"p50={report[metric]['p50']} p95={report[metric]['p95']}"
    )
    print(f"[enrich] updated: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
