from google import genai
from tqdm import tqdm
import time

import argparse
import json
import os


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GEMINI_API_KEY. Export your key, e.g. export GEMINI_API_KEY='...'"
        )
    return genai.Client(api_key=api_key)


def transcribe_audio(client: genai.Client, file_path: str) -> str:
    print(f"正在上传 {file_path} ...")

    audio_file = client.files.upload(file=file_path)
    print(f"上传完成: {audio_file.uri}")

    while audio_file.state.name == "PROCESSING":
        print("正在处理音频文件...")
        time.sleep(2)
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name != "ACTIVE":
        raise Exception(f"文件处理失败，状态: {audio_file.state.name}")

    print("音频处理完毕，准备发送给模型...")

    prompt_prefix = "Please transcribe this Bengali audio:"
    prompt_suffix = (
        "Output the transcription ONLY ONCE. Do NOT output anything else. "
        "Except english words, for Bengali words, do not latinize the text, "
        "only transcribe in Bangla alphabet."
    )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt_prefix, audio_file, prompt_suffix],
    )

    print(response.prompt_feedback)
    return response.text


def load_existing_predictions(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def should_skip_prediction(text: str) -> bool:
    """Non-empty successful prediction -> skip on resume."""
    if text is None:
        return False
    s = str(text).strip()
    if not s:
        return False
    return not s.startswith("[ERROR]")


def load_manifest_entries(manifest_path: str):
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            entries.append(
                {
                    "utterance_id": o["utterance_id"],
                    "audio_path": o["audio_path"],
                }
            )
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bengali ASR with Gemini (folder of WAVs or manifest.jsonl)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--audios_root",
        type=str,
        help="Directory containing audio files (legacy: keys are filenames).",
    )
    group.add_argument(
        "--manifest",
        type=str,
        help="manifest.jsonl from prepare_indictts_sample.py (keys: utterance_id).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <folder_basename>_asr_output.json or manifest_predictions.json).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load existing output; re-transcribe everything.",
    )
    args = parser.parse_args()

    client = get_client()
    results = {}

    try:
        if args.manifest:
            entries = load_manifest_entries(args.manifest)
            out_path = args.output or os.path.join(
                os.path.dirname(os.path.abspath(args.manifest)),
                "gemini_predictions.json",
            )
            if not args.no_resume:
                results = load_existing_predictions(out_path)
                n_skip = sum(
                    1
                    for ent in entries
                    if should_skip_prediction(results.get(ent["utterance_id"]))
                )
                if n_skip:
                    print(f"Resume: skipping {n_skip} utterances already in {out_path}")
            for ent in tqdm(entries, desc="Transcribing"):
                uid = ent["utterance_id"]
                audio_abs = ent["audio_path"]
                if not args.no_resume and should_skip_prediction(results.get(uid)):
                    continue
                print(f"Processing {uid}")
                try:
                    results[uid] = transcribe_audio(client, audio_abs)
                    print("\n--- 转录结果 ---\n")
                    print(results[uid])
                except Exception as e:
                    print(f"发生错误: {e}")
                    results[uid] = f"[ERROR] {e}"
                finally:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            audios = sorted(os.listdir(args.audios_root))
            audios_root_basename = os.path.basename(args.audios_root.rstrip(os.sep))
            out_path = args.output or f"{audios_root_basename}_asr_output.json"

            if not args.no_resume:
                results = load_existing_predictions(out_path)
                n_skip = sum(
                    1 for a in audios if should_skip_prediction(results.get(a))
                )
                if n_skip:
                    print(f"Resume: skipping {n_skip} files already in {out_path}")

            for audio in tqdm(audios, desc="Transcribing"):
                audio_abs = os.path.join(args.audios_root, audio)
                if not os.path.isfile(audio_abs):
                    continue
                if not args.no_resume and should_skip_prediction(results.get(audio)):
                    continue
                print(f"Processing audio {audio}")
                try:
                    results[audio] = transcribe_audio(client, audio_abs)
                    print("\n--- 转录结果 ---\n")
                    print(results[audio])
                except Exception as e:
                    print(f"发生错误: {e}")
                    results[audio] = f"[ERROR] {e}"
                finally:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
    finally:
        client.close()

    print(f"Saved predictions to {out_path}")
