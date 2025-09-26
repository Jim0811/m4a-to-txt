#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m4a_to_txt.py
簡單的命令列工具：把輸入的 m4a 檔案轉成文字 (.txt)。
使用流程：
    python m4a_to_txt.py input.m4a output.txt --model small
"""

import sys
import os
import argparse
from pydub import AudioSegment
import whisper
from tqdm import tqdm
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)

def ensure_ffmpeg():
    # 嘗試執行 ffmpeg --version 確認是否可用
    from shutil import which
    if which("ffmpeg") is None:
        raise EnvironmentError("找不到 ffmpeg，請先安裝並把 ffmpeg 加到 PATH。")

def m4a_to_wav(m4a_path, wav_path):
    """使用 pydub 將 m4a 轉成 wav（16k/16-bit）以提高相容性"""
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    # 轉成單聲道、16kHz、16-bit PCM
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_with_whisper(wav_path, model_name="small", language=None, verbose=True):
    """使用 whisper 進行轉錄，回傳完整文字"""
    if verbose:
        print(f"載入 Whisper 模型：{model_name}（視模型大小載入時間會較長）...")
    model = whisper.load_model(model_name)

    # 使用 model.transcribe（會自動做分段）
    if verbose:
        print("開始轉錄...")
    options = {}
    if language:
        options["language"] = language
        options["task"] = "transcribe"
    # Whisper 的 transcribe 已含語音段落與時間標記，這裡我們只取 text
    result = model.transcribe(wav_path, **options)
    text = result.get("text", "").strip()
    return text, result

def main():
    parser = argparse.ArgumentParser(description="將 m4a 轉為文字（.txt），使用本地 Whisper。")
    parser.add_argument("input", help="輸入檔案 (m4a)")
    parser.add_argument("output", help="輸出檔案 (.txt)")
    parser.add_argument("--model", default="small", help="Whisper 模型大小，e.g. tiny, base, small, medium, large (預設: small)")
    parser.add_argument("--language", default=None, help="如果知道音檔語言，可指定語言代碼（例如 zh, en），可加速與提高正確率")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    model_name = args.model
    language = args.language

    if not os.path.isfile(input_path):
        print(f"錯誤：找不到輸入檔案：{input_path}")
        sys.exit(1)

    try:
        ensure_ffmpeg()
    except EnvironmentError as e:
        print("錯誤：", e)
        sys.exit(1)

    tmp_wav = os.path.splitext(output_path)[0] + "_tmp.wav"

    try:
        print("轉換 m4a -> wav ...")
        m4a_to_wav(input_path, tmp_wav)
        print("轉換完成，暫存 wav：", tmp_wav)

        text, meta = transcribe_with_whisper(tmp_wav, model_name=model_name, language=language)

        # 將結果寫入 output txt
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"轉錄完成，已寫入：{output_path}")
    except Exception as e:
        print("處理過程發生錯誤：", e)
        raise
    finally:
        # 可選擇刪除暫存 wav（若需要保留可註解掉）
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

if __name__ == "__main__":
    main()
