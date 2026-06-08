#!/usr/bin/env python3
"""Sample FLEURS ar_eg clips and export for e2e Whisper ASR evaluation."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

E2E_ROOT = Path(__file__).resolve().parents[1]
FLEURS_ID = "google/fleurs"
FLEURS_CONFIG = "ar_eg"
SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare 1000 FLEURS ar_eg utterances for e2e ASR")
    p.add_argument("--count", type=int, default=1000, help="Number of utterances to export")
    p.add_argument("--min-duration", type=float, default=10.0, help="Min clip length (seconds)")
    p.add_argument("--max-duration", type=float, default=30.0, help="Max clip length (seconds)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument(
        "--wav-dir",
        type=Path,
        default=E2E_ROOT / "output" / "wav" / "ar_1000_fleurs",
        help="Output directory for ar_XXXX.wav files",
    )
    p.add_argument(
        "--ref-txt",
        type=Path,
        default=E2E_ROOT / "input" / "ar_1000_sample_sent.txt",
        help="Reference text (one line per utterance, aligned with wav ids)",
    )
    p.add_argument(
        "--index-tsv",
        type=Path,
        default=E2E_ROOT / "output" / "fleurs_ar_1000.index.tsv",
        help="Metadata TSV: utt_id, fleurs_id, split, duration_sec",
    )
    p.add_argument(
        "--stats-json",
        type=Path,
        default=E2E_ROOT / "output" / "fleurs_ar_1000.stats.json",
        help="Sampling statistics JSON",
    )
    p.add_argument(
        "--locale-prefix",
        default="ar",
        help="Utterance id prefix (ar_0001.wav)",
    )
    return p.parse_args()


def load_fleurs_pool():
    from datasets import Audio, concatenate_datasets, load_dataset

    parts = []
    for split in ("train", "validation", "test"):
        ds = load_dataset(FLEURS_ID, FLEURS_CONFIG, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.add_column("_split", [split] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts)


def read_audio(audio: dict) -> tuple[np.ndarray, int]:
    raw = audio.get("bytes")
    if raw:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sr)
    path = audio.get("path")
    if path and Path(path).is_file():
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sr)
    arr = audio.get("array")
    if arr is not None:
        sr = int(audio.get("sampling_rate") or SAMPLE_RATE)
        return np.asarray(arr, dtype=np.float32), sr
    raise ValueError("audio entry has no bytes, readable path, or array")


def clip_duration_sec(row: dict) -> float:
    n = row.get("num_samples")
    if n is not None and int(n) > 0:
        return int(n) / SAMPLE_RATE
    return clip_duration_sec_from_audio(row.get("audio") or {})


def clip_duration_sec_from_audio(audio: dict) -> float:
    raw = audio.get("bytes")
    if raw:
        info = sf.info(io.BytesIO(raw))
        return float(info.duration)
    return 0.0


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    print(f"[fleurs-ar] Loading {FLEURS_ID} ({FLEURS_CONFIG}) ...", flush=True)
    pool = load_fleurs_pool()
    print(f"[fleurs-ar] Total rows in pool: {len(pool)}", flush=True)

    candidates: list[tuple[int, float]] = []
    for i in range(len(pool)):
        row = pool[i]
        dur = clip_duration_sec(row)
        if args.min_duration <= dur <= args.max_duration:
            text = (row.get("transcription") or row.get("raw_transcription") or "").strip()
            if text:
                candidates.append((i, dur))

    print(
        f"[fleurs-ar] Candidates in [{args.min_duration}, {args.max_duration}]s: "
        f"{len(candidates)}",
        flush=True,
    )
    if len(candidates) < args.count:
        print(
            f"[fleurs-ar] ERROR: need {args.count} clips but only {len(candidates)} match filters.",
            file=sys.stderr,
        )
        return 1

    picked = random.sample(candidates, args.count)
    picked.sort(key=lambda x: x[0])

    args.wav_dir.mkdir(parents=True, exist_ok=True)
    args.ref_txt.parent.mkdir(parents=True, exist_ok=True)
    args.index_tsv.parent.mkdir(parents=True, exist_ok=True)

    ref_lines: list[str] = []
    index_lines = ["utt_id\tfleurs_id\tsplit\tduration_sec\tline_no"]
    durations: list[float] = []

    for n, (pool_idx, dur) in enumerate(picked, start=1):
        row = pool[pool_idx]
        utt_id = f"{args.locale_prefix}_{n:04d}"
        wav_path = args.wav_dir / f"{utt_id}.wav"

        arr, sr = read_audio(row["audio"])
        sf.write(wav_path, arr, sr, subtype="PCM_16")

        text = (row["transcription"] or row["raw_transcription"] or "").strip()
        ref_lines.append(text)
        fleurs_id = str(row.get("id") or "")
        split = str(row.get("_split") or "")
        index_lines.append(f"{utt_id}\t{fleurs_id}\t{split}\t{dur:.3f}\t{n}")
        durations.append(dur)

    args.ref_txt.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
    args.index_tsv.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    stats = {
        "source": f"{FLEURS_ID}/{FLEURS_CONFIG}",
        "count": args.count,
        "min_duration_sec": args.min_duration,
        "max_duration_sec": args.max_duration,
        "seed": args.seed,
        "pool_size": len(pool),
        "candidates": len(candidates),
        "duration_sec": {
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
        },
        "wav_dir": str(args.wav_dir),
        "ref_txt": str(args.ref_txt),
    }
    args.stats_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"[fleurs-ar] Wrote {args.count} wav files -> {args.wav_dir}", flush=True)
    print(f"[fleurs-ar] Reference text -> {args.ref_txt}", flush=True)
    print(f"[fleurs-ar] Index -> {args.index_tsv}", flush=True)
    print(
        f"[fleurs-ar] Duration range: {stats['duration_sec']['min']:.2f}s - "
        f"{stats['duration_sec']['max']:.2f}s (mean {stats['duration_sec']['mean']:.2f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
