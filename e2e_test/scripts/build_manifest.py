#!/usr/bin/env python3
"""Build LITs TTS manifest (wav_path|text) from normalized lines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TTS manifest from normalized.txt")
    p.add_argument(
        "--normalized",
        type=Path,
        required=True,
        help="Input normalized text (one sentence per line)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest path (wav|text per line)",
    )
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional TSV: utt_id, line_no, char_len, skipped_reason",
    )
    p.add_argument(
        "--locale",
        default="bn",
        help="Utterance id prefix (default: bn)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only use first N lines (0 = all)",
    )
    p.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="Skip blank lines (default: on)",
    )
    p.add_argument(
        "--no-skip-empty",
        action="store_false",
        dest="skip_empty",
        help="Keep blank lines in manifest",
    )
    p.add_argument(
        "--utt-width",
        type=int,
        default=4,
        help="Zero-pad width for line number in utt id (default: 4)",
    )
    return p.parse_args()


def build_manifest(
    normalized_path: Path,
    manifest_path: Path,
    index_path: Path | None,
    *,
    locale: str,
    limit: int,
    skip_empty: bool,
    utt_width: int,
) -> dict:
    lines = normalized_path.read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[:limit]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    index_rows: list[str] = []
    manifest_rows: list[str] = []

    stats = {
        "locale": locale,
        "normalized": str(normalized_path),
        "manifest": str(manifest_path),
        "input_lines": len(lines),
        "written": 0,
        "skipped_empty": 0,
        "skipped_other": 0,
    }

    for line_no, raw in enumerate(lines, start=1):
        text = raw.rstrip("\r\n")
        utt_id = f"{locale}_{line_no:0{utt_width}d}"
        wav_name = f"{utt_id}.wav"

        if skip_empty and not text.strip():
            stats["skipped_empty"] += 1
            if index_path is not None:
                index_rows.append(f"{utt_id}\t{line_no}\t0\tskipped_empty")
            continue

        # LITs splits only on the first '|'; keep text as-is otherwise.
        manifest_rows.append(f"{wav_name}|{text}")
        stats["written"] += 1
        if index_path is not None:
            index_rows.append(f"{utt_id}\t{line_no}\t{len(text)}\tok")

    manifest_path.write_text("\n".join(manifest_rows) + ("\n" if manifest_rows else ""), encoding="utf-8")

    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        header = "utt_id\tline_no\tchar_len\tstatus\n"
        index_path.write_text(header + "\n".join(index_rows) + ("\n" if index_rows else ""), encoding="utf-8")
        stats["index"] = str(index_path)

    return stats


def main() -> int:
    args = parse_args()
    if not args.normalized.is_file():
        print(f"Input not found: {args.normalized}", file=sys.stderr)
        return 1

    stats = build_manifest(
        args.normalized,
        args.output,
        args.index,
        locale=args.locale,
        limit=args.limit,
        skip_empty=args.skip_empty,
        utt_width=args.utt_width,
    )

    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[manifest] {args.normalized} -> {args.output}")
    print(
        f"[manifest] input_lines={stats['input_lines']} "
        f"written={stats['written']} skipped_empty={stats['skipped_empty']}"
    )
    if args.index:
        print(f"[manifest] index: {args.index}")
    print(f"[manifest] stats: {stats_path}")

    if stats["written"] == 0:
        print("[manifest] ERROR: no utterances written", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
