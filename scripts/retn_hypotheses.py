#!/usr/bin/env python3
"""Re-run TN on stored hyp_raw in wer_per_utt.csv, then rebuild WER report."""

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
from lib.wer_analysis import corpus_error_rate  # noqa: E402
from scripts.transcribe_and_wer import TNHypNormalizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-TN hyp_raw and refresh wer_per_utt.csv")
    p.add_argument(
        "--asr-dir",
        type=Path,
        required=True,
        help="ASR output dir with wer_per_utt.csv",
    )
    p.add_argument(
        "--tn-bin",
        type=Path,
        default=E2E_ROOT / "bin" / "en_tts",
        help="TN executable (default: e2e_test/bin/en_tts)",
    )
    p.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="Optional exclude list; if set, runs enrich_wer_report after update",
    )
    p.add_argument(
        "--language",
        default="en",
        help="Language for enrich step (default: en)",
    )
    p.add_argument("--no-enrich", action="store_true", help="Only update CSV, skip report")
    p.add_argument(
        "--source-txt",
        type=Path,
        default=None,
        help="Pre-TN written references for digit-span merge before TN",
    )
    p.add_argument(
        "--merge-written-spans",
        action="store_true",
        help="Splice written digit spans into hyp before TN",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    asr_dir = args.asr_dir.resolve()
    csv_path = asr_dir / "wer_per_utt.csv"
    if not csv_path.is_file():
        print(f"Missing {csv_path}", file=sys.stderr)
        return 1

    old_wer: float | None = None
    report_path = asr_dir / "wer_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        old_wer = corpus_error_rate(report["corpus_wer"]) if report.get("corpus_wer") else None

    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if "hyp_raw" not in fieldnames:
        print("wer_per_utt.csv has no hyp_raw column", file=sys.stderr)
        return 2

    merge_written_spans = (
        args.merge_written_spans and should_merge_written_spans(args.language)
    )
    written_lines: list[str] = []
    if merge_written_spans:
        source_path = (
            args.source_txt.resolve()
            if args.source_txt
            else default_source_txt(E2E_ROOT, args.language)
        )
        if source_path.is_file():
            written_lines = load_written_lines(source_path)
            print(f"[retn] written-span merge: on source={source_path}")
        else:
            print(f"[retn] WARN: written source not found ({source_path}); merge disabled")
            merge_written_spans = False

    normalizer = TNHypNormalizer(args.tn_bin.resolve())
    try:
        for i, row in enumerate(rows, start=1):
            hyp_raw = row.get("hyp_raw", "")
            line_no = int(row["line_no"]) if row.get("line_no") else 0
            written = (
                written_lines[line_no - 1]
                if merge_written_spans and 1 <= line_no <= len(written_lines)
                else ""
            )
            _merged, hyp_tn = prepare_hyp_for_tn(
                hyp_raw,
                written,
                language=args.language if merge_written_spans else None,
                tn=normalizer,
            )
            row["hyp_tn"] = hyp_tn
            row["hyp"] = hyp_tn
            if i % 100 == 0 or i == len(rows):
                print(f"[retn] TN {i}/{len(rows)}", flush=True)
    finally:
        normalizer.close()

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hyp_path = asr_dir / "asr_hyp.tsv"
    hyp_path.write_text(
        "utt_id\tline_no\thypothesis\n"
        + "\n".join(
            f"{r['utt_id']}\t{r.get('line_no', '')}\t{r.get('hyp_tn', '')}"
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[retn] updated: {csv_path}")
    print(f"[retn] updated: {hyp_path}")

    if args.no_enrich:
        return 0

    enrich_args = [
        sys.executable,
        str(E2E_ROOT / "scripts" / "enrich_wer_report.py"),
        "--asr-dir",
        str(asr_dir),
        "--language",
        args.language,
        "--no-retn-hyp",
    ]
    if merge_written_spans:
        source_for_enrich = (
            args.source_txt.resolve()
            if args.source_txt
            else default_source_txt(E2E_ROOT, args.language)
        )
        enrich_args.extend(["--source-txt", str(source_for_enrich)])
    if args.exclude_file:
        enrich_args.extend(["--exclude-file", str(args.exclude_file.resolve())])

    import subprocess

    subprocess.run(enrich_args, check=True)

    if old_wer is not None and report_path.is_file():
        new_report = json.loads(report_path.read_text(encoding="utf-8"))
        new_wer = (
            corpus_error_rate(new_report["corpus_wer"])
            if new_report.get("corpus_wer")
            else None
        )
        label = new_report.get("corpus_wer", {}).get("metric", "wer").upper()
        print(f"[retn] corpus {label}: {old_wer} -> {new_wer}")
        if new_wer is not None and old_wer is not None:
            delta = new_wer - old_wer
            print(f"[retn] delta: {delta:+.6f} ({delta * 100:+.3f} pp)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
