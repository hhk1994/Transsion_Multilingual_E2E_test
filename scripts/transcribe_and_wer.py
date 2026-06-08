#!/usr/bin/env python3
"""ASR on wav files + WER/CER vs normalized.txt.

Supported backends:
- faster-whisper
- hf (Whisper pipeline)
- indic-conformer (ai4bharat/indic-conformer-600m-multilingual)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol

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
from lib.wer_analysis import (  # noqa: E402
    build_wer_analysis,
    corpus_error_rate,
    corpus_metric_key,
    error_unit_for_language,
    utterance_error_rates,
)

# e.g. bn_0001, zh_0042, en_1000, ar-en_0123
_UTT_LINE_RE = re.compile(r"^([a-z][a-z0-9-]*)_(\d+)$", re.IGNORECASE)
HF_MODEL_PREFIXES = ("mozilla-ai/", "openai/")
INDIC_CONFORMER_PREFIXES = ("ai4bharat/indic-conformer",)


class Transcriber(Protocol):
    def transcribe(self, wav_path: Path) -> str: ...


class TNHypNormalizer:
    """Normalize ASR hypotheses with TN bn_tts binary."""

    def __init__(self, tn_bin: Path) -> None:
        self.tn_bin = tn_bin.resolve()
        if not self.tn_bin.is_file():
            raise FileNotFoundError(f"TN binary not found: {self.tn_bin}")
        if not self.tn_bin.stat().st_mode & 0o111:
            raise PermissionError(f"TN binary is not executable: {self.tn_bin}")
        self._proc = subprocess.Popen(
            [str(self.tn_bin)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("TN process stdio unavailable")
        self._proc.stdin.write(text.rstrip("\n") + "\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline()
        return out.rstrip("\n")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class FasterWhisperTranscriber:
    def __init__(self, model: str, language: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        self.language = language
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: Path) -> str:
        segments, _info = self._model.transcribe(
            str(wav_path),
            language=self.language,
            task="transcribe",
            vad_filter=True,
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(parts)


class HFWhisperTranscriber:
    def __init__(self, model_id: str, language: str, device: str) -> None:
        import torch
        from transformers import pipeline

        device_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
        if device == "cuda" and device_id < 0:
            print("[asr] WARN: cuda requested but unavailable, using cpu", file=sys.stderr)
        self.language = language
        dtype = torch.float16 if device_id >= 0 else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device_id,
            dtype=dtype,
        )

    @staticmethod
    def _load_audio_16k(wav_path: Path) -> dict:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio.functional as AF

        audio, sr = sf.read(str(wav_path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        if sr != 16000:
            wav = AF.resample(wav, sr, 16000)
        return {"array": wav.squeeze(0).numpy(), "sampling_rate": 16000}

    def transcribe(self, wav_path: Path) -> str:
        inputs = self._load_audio_16k(wav_path)
        result = self._pipe(
            inputs,
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("text", "")).strip()
        return str(result).strip()


class IndicConformerTranscriber:
    def __init__(self, model_id: str, language: str, device: str, decoding: str) -> None:
        import torch
        from transformers import AutoModel

        self.language = language
        self.decoding = decoding
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        if device == "cuda" and self.device == "cpu":
            print("[asr] WARN: cuda requested but unavailable, using cpu", file=sys.stderr)

        self._torch = torch
        try:
            self._model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            msg = str(exc)
            if "gated repo" in msg.lower() or "401" in msg:
                raise RuntimeError(
                    "IndicConformer model is gated on Hugging Face. "
                    "Please log in, accept access terms on the model page, and set HF_TOKEN. "
                    f"Model: https://huggingface.co/{model_id}"
                ) from exc
            raise
        self._model.to(self.device)
        self._model.eval()

    def transcribe(self, wav_path: Path) -> str:
        inputs = HFWhisperTranscriber._load_audio_16k(wav_path)
        wav = self._torch.from_numpy(inputs["array"]).unsqueeze(0).to(self.device)
        # IndicConformer custom forward: model(wav, lang_code, decoder_type)
        out = self._model(wav, self.language, self.decoding)
        return str(out).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASR transcribe + WER vs normalized.txt")
    p.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help="Directory with <locale>_NNNN.wav files (e.g. bn_0001, zh_0042, en_1000)",
    )
    p.add_argument(
        "--utt-prefix",
        default="",
        help="Only evaluate wav files with this locale prefix (e.g. zh, en, bn)",
    )
    p.add_argument(
        "--normalized",
        type=Path,
        default=E2E_ROOT / "output" / "normalized.txt",
        help="Reference text (one line per utterance)",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="ASR outputs (hyp, csv, report)")
    p.add_argument(
        "--backend",
        choices=["auto", "faster-whisper", "hf", "indic-conformer"],
        default="auto",
        help="ASR backend (auto picks by model id prefix)",
    )
    p.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper size or HuggingFace model id (e.g. mozilla-ai/whisper-large-v3-bn)",
    )
    p.add_argument("--language", default="bn", help="Whisper language code")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    p.add_argument(
        "--decoding",
        default="ctc",
        choices=["ctc", "rnnt"],
        help="Decoding for indic-conformer backend",
    )
    p.add_argument(
        "--compute-type",
        default="",
        help="faster-whisper compute type (default: float16 on cuda, int8 on cpu)",
    )
    p.add_argument(
        "--tn-normalize-hyp",
        action="store_true",
        default=True,
        help="Normalize ASR hypothesis with TN bn_tts before WER (default: on)",
    )
    p.add_argument(
        "--no-tn-normalize-hyp",
        action="store_false",
        dest="tn_normalize_hyp",
        help="Disable TN normalization on ASR hypothesis",
    )
    p.add_argument(
        "--tn-bin",
        type=Path,
        default=E2E_ROOT / "bin" / "bn_tts",
        help="Path to TN bn_tts binary for hypothesis normalization",
    )
    p.add_argument(
        "--source-txt",
        type=Path,
        default=None,
        help="Pre-TN written references (for digit-span merge before hyp TN; "
        "default: input/<lang>_1000_sample_sent.txt when present)",
    )
    p.add_argument(
        "--merge-written-spans",
        action="store_true",
        help="Splice written digit spans into hyp before TN (char-level langs; "
        "fixes TN cardinal vs digit-by-digit when ASR keeps Arabic digits)",
    )
    p.add_argument("--limit", type=int, default=0, help="Max wav files (0 = all)")
    return p.parse_args()


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    if args.model.startswith(INDIC_CONFORMER_PREFIXES):
        return "indic-conformer"
    if "/" in args.model or args.model.startswith(HF_MODEL_PREFIXES):
        return "hf"
    return "faster-whisper"


def build_transcriber(args: argparse.Namespace, backend: str, compute_type: str) -> Transcriber:
    if backend == "indic-conformer":
        return IndicConformerTranscriber(args.model, args.language, args.device, args.decoding)
    if backend == "hf":
        return HFWhisperTranscriber(args.model, args.language, args.device)
    return FasterWhisperTranscriber(args.model, args.language, args.device, compute_type)


def parse_utt_stem(stem: str) -> tuple[str, int] | None:
    m = _UTT_LINE_RE.match(stem)
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))


def line_no_from_stem(stem: str) -> int:
    parsed = parse_utt_stem(stem)
    if parsed is None:
        raise ValueError(f"Unexpected wav stem (expected <locale>_NNNN, e.g. zh_0001): {stem}")
    return parsed[1]


def load_normalized_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def list_wavs(wav_dir: Path, limit: int, utt_prefix: str = "") -> list[Path]:
    prefix = utt_prefix.lower()
    wavs: list[Path] = []
    skipped: list[str] = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        parsed = parse_utt_stem(wav_path.stem)
        if parsed is None:
            skipped.append(wav_path.name)
            continue
        locale, _line_no = parsed
        if prefix and locale != prefix:
            continue
        wavs.append(wav_path)

    if skipped:
        print(
            f"[asr] WARN: skipped {len(skipped)} wav(s) with unrecognized names "
            f"(expected <locale>_NNNN): {', '.join(skipped[:5])}"
            + (" ..." if len(skipped) > 5 else ""),
            file=sys.stderr,
        )

    wavs.sort(key=lambda p: parse_utt_stem(p.stem)[1])  # type: ignore[index]
    if limit > 0:
        wavs = wavs[:limit]
    return wavs


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def main() -> int:
    args = parse_args()
    wav_dir = args.wav_dir.resolve()
    normalized_path = args.normalized.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wav_dir.is_dir():
        print(f"wav-dir not found: {wav_dir}", file=sys.stderr)
        return 1
    if not normalized_path.is_file():
        print(f"normalized file not found: {normalized_path}", file=sys.stderr)
        return 1

    refs_all = load_normalized_lines(normalized_path)
    merge_written_spans = (
        args.merge_written_spans and should_merge_written_spans(args.language)
    )
    source_path: Path | None = None
    written_lines: list[str] = []
    if merge_written_spans:
        source_path = args.source_txt.resolve() if args.source_txt else default_source_txt(E2E_ROOT, args.language)
        if source_path.is_file():
            written_lines = load_written_lines(source_path)
        else:
            print(f"[asr] WARN: written source not found ({source_path}); hyp span merge disabled")
            merge_written_spans = False
            source_path = None

    wavs = list_wavs(wav_dir, args.limit, args.utt_prefix)
    if not wavs:
        hint = f" (utt_prefix={args.utt_prefix})" if args.utt_prefix else ""
        print(f"No matching wav files in {wav_dir}{hint}", file=sys.stderr)
        return 2

    backend = resolve_backend(args)
    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    print(
        f"[asr] backend={backend} model={args.model} lang={args.language} "
        f"device={args.device} compute={compute_type if backend == 'faster-whisper' else 'n/a'}"
    )
    print(f"[asr] wav_dir={wav_dir} n={len(wavs)}")
    if args.utt_prefix:
        print(f"[asr] utt_prefix={args.utt_prefix}")
    if args.tn_normalize_hyp:
        print(f"[asr] tn_normalize_hyp=on tn_bin={args.tn_bin}")
    else:
        print("[asr] tn_normalize_hyp=off (Whisper hyp used as-is for WER)")
    if merge_written_spans:
        print(f"[asr] hyp written-span merge: on source={source_path}")

    t0 = time.time()
    transcriber = build_transcriber(args, backend, compute_type)
    hyp_tn_normalizer = TNHypNormalizer(args.tn_bin) if args.tn_normalize_hyp else None

    rows: list[dict] = []
    hyp_lines: list[str] = []
    ref_norms: list[str] = []
    hyp_norms: list[str] = []
    error_unit = error_unit_for_language(args.language)

    try:
        for wav_path in wavs:
            utt_id = wav_path.stem
            line_no = line_no_from_stem(utt_id)
            if line_no < 1 or line_no > len(refs_all):
                print(f"[asr] SKIP {utt_id}: line_no {line_no} out of range (normalized has {len(refs_all)} lines)")
                continue

            ref_raw = refs_all[line_no - 1]

            print(f"[asr] transcribing {utt_id} ...")
            hyp_raw = transcriber.transcribe(wav_path)
            written = (
                written_lines[line_no - 1]
                if merge_written_spans and line_no <= len(written_lines)
                else ""
            )
            _hyp_merged, hyp_tn = prepare_hyp_for_tn(
                hyp_raw,
                written,
                language=args.language if merge_written_spans else None,
                tn=hyp_tn_normalizer,
            )
            ref_norm, hyp_norm = normalize_texts_for_wer(
                ref_raw,
                hyp_tn,
                language=args.language,
            )

            wer, cer = utterance_error_rates(ref_norm, hyp_norm, unit=error_unit)

            ref_norms.append(ref_norm)
            hyp_norms.append(hyp_norm)

            rows.append(
                {
                    "utt_id": utt_id,
                    "line_no": line_no,
                    "wav": str(wav_path),
                    "wer": round(wer, 6),
                    "cer": round(cer, 6),
                    "ref_len": len(ref_norm),
                    "hyp_len": len(hyp_norm),
                    "ref": ref_raw,
                    "hyp_raw": hyp_raw,
                    "hyp_tn": hyp_tn,
                    "hyp": hyp_tn,
                }
            )
            hyp_lines.append(f"{utt_id}\t{line_no}\t{hyp_tn}")
    finally:
        if hyp_tn_normalizer is not None:
            hyp_tn_normalizer.close()

    if not rows:
        print("[asr] ERROR: no utterances evaluated", file=sys.stderr)
        return 2

    csv_path = out_dir / "wer_per_utt.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id",
                "line_no",
                "wav",
                "wer",
                "cer",
                "ref_len",
                "hyp_len",
                "ref",
                "hyp_raw",
                "hyp_tn",
                "hyp",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    hyp_path = out_dir / "asr_hyp.tsv"
    hyp_path.write_text("utt_id\tline_no\thypothesis\n" + "\n".join(hyp_lines) + "\n", encoding="utf-8")

    wers = [r["wer"] for r in rows]
    cers = [r["cer"] for r in rows]
    wers_sorted = sorted(wers)
    n = len(wers)

    analysis = build_wer_analysis(rows, ref_norms, hyp_norms, unit=error_unit)

    report = {
        "corpus_wer": analysis["corpus_wer"],
        "top_wer": analysis["top_wer"],
        "confusion_pairs": analysis["confusion_pairs"],
        "wav_dir": str(wav_dir),
        "normalized": str(normalized_path),
        "asr_backend": backend,
        "asr_model": args.model,
        "language": args.language,
        "device": args.device,
        "compute_type": compute_type if backend == "faster-whisper" else None,
        "tn_normalize_hyp": args.tn_normalize_hyp,
        "tn_bin": str(args.tn_bin.resolve()) if args.tn_normalize_hyp else None,
        "hyp_written_span_merge": merge_written_spans,
        "source_txt": str(source_path) if source_path else None,
        "utt_prefix": args.utt_prefix or None,
        "n_evaluated": n,
        "elapsed_sec": round(time.time() - t0, 2),
        "wer": {
            "mean": round(sum(wers) / n, 6),
            "p50": round(percentile(wers_sorted, 0.5), 6),
            "p95": round(percentile(wers_sorted, 0.95), 6),
            "max": round(max(wers), 6),
        },
        "cer": {
            "mean": round(sum(cers) / n, 6),
            "p50": round(percentile(sorted(cers), 0.5), 6),
            "p95": round(percentile(sorted(cers), 0.95), 6),
        },
        "outputs": {
            "wer_csv": str(csv_path),
            "hyp_tsv": str(hyp_path),
        },
    }
    report_path = out_dir / "wer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "confusion_pairs.json").write_text(
        json.dumps(report["confusion_pairs"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metric = corpus_metric_key(error_unit)
    metric_label = metric.upper()
    print(
        f"[asr] {metric_label} mean={report[metric]['mean']} "
        f"p50={report[metric]['p50']} p95={report[metric]['p95']} (n={n})"
    )
    if metric == "wer":
        print(f"[asr] CER mean={report['cer']['mean']}")
    cw = report["corpus_wer"]
    print(
        f"[asr] corpus {metric_label}={corpus_error_rate(cw)} "
        f"({cw['error_unit']}-level, {cw['n_ref_tokens']} ref tokens) "
        f"S={cw['substitutions']['count']}({cw['substitutions']['rate']}) "
        f"D={cw['deletions']['count']}({cw['deletions']['rate']}) "
        f"I={cw['insertions']['count']}({cw['insertions']['rate']})"
    )
    print(f"[asr] confusion pairs: {cw['n_confusion_pairs']}, error cases: {cw['n_error_cases']}")
    print(f"[asr] report: {report_path}")
    print(f"[asr] per-utt: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
