#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""將 M4A 音檔轉為 UTF-8 文字檔。"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import whisper
from tqdm import tqdm


def check_dependencies() -> None:
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(f"找不到 {command}，請安裝 FFmpeg 並加入 PATH。")


def get_duration(input_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(input_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def parse_ffmpeg_time(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def convert_to_wav(input_path: Path, wav_path: Path) -> None:
    duration = get_duration(input_path)
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-progress", "pipe:1", "-nostats", "-y", str(wav_path),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert process.stdout is not None
    errors = []
    with tqdm(total=duration, desc="轉換 WAV", unit="秒", dynamic_ncols=True) as bar:
        for line in process.stdout:
            line = line.strip()
            if line.startswith("out_time="):
                try:
                    elapsed = parse_ffmpeg_time(line.partition("=")[2])
                    bar.update(max(0.0, min(duration, elapsed) - bar.n))
                except ValueError:
                    pass
            elif line and "=" not in line:
                errors.append(line)

        if process.wait() != 0:
            raise RuntimeError("FFmpeg 轉換失敗：" + "\n".join(errors[-10:]))
        bar.update(max(0.0, duration - bar.n))

def add_punctuation(result: dict) -> str:
    """保留 Whisper 的標點，並在明顯停頓及文末補上句號。"""
    segments = result.get("segments", [])
    chinese = result.get("language") == "zh"
    period = "。" if chinese else "."
    punctuation = "。！？.!?，,；;：:、…"

    if not segments:
        text = result.get("text", "").strip()
        return text if not text or text[-1] in punctuation else text + period

    parts = []
    for index, segment in enumerate(segments):
        part = segment["text"]
        parts.append(part)
        if index + 1 < len(segments):
            gap = segments[index + 1]["start"] - segment["end"]
            if gap >= 1.2 and part.rstrip() and part.rstrip()[-1] not in punctuation:
                parts[-1] = part.rstrip() + period

    text = "".join(parts).strip()
    return text if not text or text[-1] in punctuation else text + period

def transcribe(wav_path: Path, model_name: str, language: str | None) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"載入 Whisper 模型：{model_name}（{device}）...", flush=True)
    model = whisper.load_model(model_name, device=device)
    print("開始轉錄...", flush=True)
    options = {"verbose": False, "task": "transcribe"}
    if language:
        options["language"] = language
    if language == "zh":
        options["initial_prompt"] = "這是一段帶有自然標點符號的繁體中文逐字稿。"
        options["carry_initial_prompt"] = True
    result = model.transcribe(str(wav_path), **options)
    return add_punctuation(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="使用本地 Whisper 將 M4A 轉為文字。")
    parser.add_argument("input", type=Path, help="輸入 M4A 檔案")
    parser.add_argument("output", type=Path, help="輸出 TXT 檔案")
    parser.add_argument("--model", default="small", help="Whisper 模型名稱（預設：small）")
    parser.add_argument("--language", help="語言代碼，例如 zh、en；省略則自動偵測")
    args = parser.parse_args()

    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"找不到輸入檔案：{args.input}")
        if args.input.resolve() == args.output.resolve():
            raise ValueError("輸入與輸出路徑不能相同。")
        check_dependencies()

        with tempfile.TemporaryDirectory(prefix="m4a_to_txt_") as tmp_dir:
            wav_path = Path(tmp_dir) / "audio.wav"
            convert_to_wav(args.input, wav_path)
            text = transcribe(wav_path, args.model, args.language)

        args.output.write_text(text, encoding="utf-8")
        print(f"轉錄完成，已寫入：{args.output}")
        return 0
    except (FileNotFoundError, ValueError, OSError, RuntimeError,
            subprocess.CalledProcessError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())