#!/usr/bin/env python3
"""
transcribe.py  —  Step 1 of 3
==============================
Extracts audio from a video file and transcribes it using Groq Whisper.

Output:
    transcript.txt   — every line is:  [HH:MM:SS] spoken text

Requirements:
    pip install groq moviepy pydub python-dotenv
"""

import os
import sys
import json
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
#  CONFIG — edit these values directly
# ============================================================

VIDEO_PATH   = "المحاضرة الاولى المثابرة .عدم التهور.mp4"
OUTPUT_FILE  = "transcript.txt"

# ffmpeg paths:
#   - On Railway (Linux): leave as "ffmpeg" and "ffprobe" — available system-wide
#   - On Windows locally: set FFMPEG_PATH and FFPROBE_PATH in your .env file
#     e.g. FFMPEG_PATH=C:\Users\...\ffmpeg.exe
FFMPEG_PATH  = os.getenv("FFMPEG_PATH",  "ffmpeg")
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")

MAX_RETRIES  = 5    # retries per chunk on connection error
RETRY_DELAY  = 5    # seconds before first retry (doubles each attempt)
MAX_CHUNK_MB = 15   # max MB per audio chunk (Groq Whisper limit is 25 MB)

# ============================================================


# ── Patch pydub BEFORE importing AudioSegment ──────────────
import pydub.utils
import pydub.audio_segment

def _ffprobe_mediainfo_json(filename, read_ahead_limit=-1):
    cmd = [FFPROBE_PATH, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(filename)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        raise RuntimeError(
            f"ffprobe returned no output for '{filename}'.\nstderr: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)

pydub.utils.mediainfo_json         = _ffprobe_mediainfo_json
pydub.audio_segment.mediainfo_json = _ffprobe_mediainfo_json

from pydub import AudioSegment

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe   = FFPROBE_PATH
# ───────────────────────────────────────────────────────────


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_audio(video_path: str, output_path: str) -> None:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("Error: moviepy not installed. Run: pip install moviepy")
        sys.exit(1)

    print(f"[1/2] Extracting audio from: {video_path}")
    clip  = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_path, logger=None)
    audio.close()
    clip.close()
    print(f"      Saved to: {output_path}")


def find_silence_boundary(audio: AudioSegment, target_ms: int, search_window_ms: int = 10_000) -> int:
    """
    Find the nearest silence boundary to `target_ms` within ±search_window_ms.
    Scans in 100ms steps and returns the position of the quietest moment.
    If no clear silence is found, returns target_ms (hard cut as fallback).
    """
    search_start = max(0,          target_ms - search_window_ms)
    search_end   = min(len(audio), target_ms + search_window_ms)
    window       = audio[search_start:search_end]

    step_ms  = 100
    best_ms  = target_ms
    best_db  = float("inf")

    for offset in range(0, len(window) - step_ms, step_ms):
        chunk_db = window[offset : offset + step_ms].dBFS
        if chunk_db < best_db:
            best_db = chunk_db
            best_ms = search_start + offset

    return best_ms


def chunk_audio_preserve_timing(audio_path: str) -> list[tuple[AudioSegment, float]]:
    """
    Split audio into chunks of at most MAX_CHUNK_MB, cutting only at silence
    boundaries. Silence is KEPT — never removed — so time offsets stay accurate.

    Returns a list of (chunk_audio, start_seconds) tuples where start_seconds
    is the exact position of that chunk in the original audio.
    """
    MAX_BYTES       = MAX_CHUNK_MB * 1024 * 1024
    audio           = AudioSegment.from_mp3(audio_path)
    total_ms        = len(audio)
    file_size       = os.path.getsize(audio_path)

    if file_size <= MAX_BYTES:
        print(f"      File is {file_size / 1024 / 1024:.1f} MB — sending as one chunk.")
        return [(audio, 0.0)]

    print(f"      File is {file_size / 1024 / 1024:.1f} MB — chunking at silence boundaries...")

    # Calculate how many ms correspond to MAX_CHUNK_MB, with 10% safety margin
    ms_per_byte     = total_ms / file_size
    target_chunk_ms = int(MAX_BYTES * ms_per_byte * 0.90)

    chunks = []
    pos_ms = 0

    # Walk through audio in target_chunk_ms steps.
    # For each step find the nearest silence boundary to cut at.
    # Stop only when the next cut would be at or beyond the end.
    while pos_ms < total_ms:
        target_end = pos_ms + target_chunk_ms

        if target_end >= total_ms:
            # Remaining audio fits in one chunk — take everything to the end
            chunks.append((audio[pos_ms:total_ms], pos_ms / 1000))
            pos_ms = total_ms
            break

        # Find quietest point near target_end — cut there, keep all audio
        cut_ms = find_silence_boundary(audio, target_end)

        # Safety: never cut before or at current position
        if cut_ms <= pos_ms:
            cut_ms = target_end

        # Safety: never cut beyond total audio length
        cut_ms = min(cut_ms, total_ms)

        chunks.append((audio[pos_ms:cut_ms], pos_ms / 1000))
        pos_ms = cut_ms

    # Final safety net: verify last chunk reaches the very end of the audio.
    # If there is any remaining audio (even 1ms), append it.
    if chunks:
        last_chunk, last_start_s = chunks[-1]
        last_end_ms = int(last_start_s * 1000) + len(last_chunk)
        if last_end_ms < total_ms:
            chunks.append((audio[last_end_ms:total_ms], last_end_ms / 1000))

    print(f"      Ready: {len(chunks)} chunks.")
    return chunks


def transcribe_chunk_with_retry(chunk_path: str, client) -> object:
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(chunk_path, "rb") as f:
                return client.audio.transcriptions.create(
                    file=(Path(chunk_path).name, f),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    language="ar",
                )
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"\n      [ERROR] Failed after {MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                raise
            print(f"\n      [{type(e).__name__}] Attempt {attempt}/{MAX_RETRIES} failed. "
                  f"Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def transcribe(audio_path: str, client) -> str:
    """
    Transcribe the audio and return the full transcript as a string.
    Each line: [HH:MM:SS] spoken text
    """
    print("[2/2] Transcribing with Groq Whisper...")

    audio_chunks = chunk_audio_preserve_timing(audio_path)

    lines   = []
    tmp_dir = tempfile.mkdtemp()

    try:
        for i, (chunk, start_seconds) in enumerate(audio_chunks):
            duration_s = len(chunk) / 1000
            print(f"      Chunk {i + 1}/{len(audio_chunks)}: "
                  f"starts at {format_timestamp(start_seconds)}, "
                  f"{duration_s:.0f}s...",
                  end="", flush=True)

            chunk_path = os.path.join(tmp_dir, f"chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")

            transcription = transcribe_chunk_with_retry(chunk_path, client)

            for seg in transcription.segments:
                # seg.start is relative to the chunk — add start_seconds for real video time
                seg_start = seg["start"] if isinstance(seg, dict) else seg.start
                seg_text  = seg["text"]  if isinstance(seg, dict) else seg.text
                real_time = seg_start + start_seconds
                lines.append(f"[{format_timestamp(real_time)}] {seg_text.strip()}")

            print(" done.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"      Transcription complete: {len(lines)} lines.")
    return "\n".join(lines)



def download_audio_from_youtube(url: str, output_path: str) -> None:
    """Download audio from a YouTube URL and save as MP3 using yt-dlp."""
    print(f"[1/2] Downloading audio from YouTube: {url}")

    # Get ffmpeg directory from FFMPEG_PATH env var (yt-dlp needs the folder, not the binary)
    ffmpeg_dir = str(Path(FFMPEG_PATH).parent) if FFMPEG_PATH != "ffmpeg" else None

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", output_path,
    ]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr.strip()}")
    print(f"      Saved to: {output_path}")


def transcribe_from_video(video_path: str, client) -> str:
    """
    Extract audio from a local video file and transcribe it.
    Returns the full transcript string.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        extract_audio(video_path, audio_path)
        return transcribe(audio_path, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def transcribe_from_youtube(url: str, client) -> str:
    """
    Download audio from a YouTube URL and transcribe it.
    Returns the full transcript string.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        download_audio_from_youtube(url, audio_path)
        return transcribe(audio_path, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env")
        sys.exit(1)
    if not os.path.isfile(VIDEO_PATH):
        print(f"Error: Video not found: {VIDEO_PATH}")
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("Error: groq not installed. Run: pip install groq")
        sys.exit(1)

    client  = Groq(api_key=groq_api_key)
    tmp_dir = tempfile.mkdtemp()

    print("\n=== Step 1: Transcription ===\n")
    try:
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        extract_audio(VIDEO_PATH, audio_path)
        transcript = transcribe(audio_path, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    output_path = Path(__file__).parent / OUTPUT_FILE
    output_path.write_text(transcript, encoding="utf-8")
    print(f"\n  Transcript saved → {output_path}")
    print(f"\n--- Preview (first 5 lines) ---")
    print("\n".join(transcript.split("\n")[:5]))
    print(f"\n=== Done. Run segment.py next. ===\n")


if __name__ == "__main__":
    main()