#!/usr/bin/env python3
"""
Audio downloader — downloads YouTube audio as 16kHz mono WAV for Whisper.
"""

import subprocess
import sys
from pathlib import Path


def download_audio(video_id, output_dir="audio"):
    """Download audio from YouTube video as 16kHz mono WAV.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save audio files

    Returns:
        Path to downloaded WAV file, or None on failure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_path = output_path / f"{video_id}.wav"

    if wav_path.exists():
        print(f"    Audio already downloaded: {wav_path}")
        return wav_path

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        # Download audio-only and convert to 16kHz mono WAV (Whisper's native format)
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
            "--output", str(wav_path),
            "--no-playlist",
            "--js-runtimes", "nodejs",
            url,
        ]

        print(f"    Downloading audio for {video_id}...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            print(f"    yt-dlp error: {result.stderr[:500]}")
            return None

        if wav_path.exists():
            size_mb = wav_path.stat().st_size / 1024 / 1024
            print(f"    Downloaded: {wav_path} ({size_mb:.1f} MB)")
            return wav_path

        # yt-dlp sometimes adds extensions
        for ext_path in output_path.glob(f"{video_id}.*"):
            if ext_path.suffix in (".wav", ".webm", ".m4a", ".opus"):
                if ext_path != wav_path:
                    ext_path.rename(wav_path)
                    return wav_path

        print(f"    Audio file not found after download")
        return None

    except subprocess.TimeoutExpired:
        print(f"    Download timed out for {video_id}")
        return None
    except Exception as e:
        print(f"    Error downloading audio: {e}")
        return None


def get_video_metadata(video_id):
    """Fetch video metadata (title, duration) via yt-dlp.

    Returns dict with title, duration, upload_date or None on failure.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-playlist",
            "--js-runtimes", "nodejs",
            url,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return None

        import json

        data = json.loads(result.stdout)
        return {
            "title": data.get("title", ""),
            "duration": data.get("duration", 0),
            "upload_date": data.get("upload_date", ""),
        }

    except Exception as e:
        print(f"    Error fetching metadata: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_audio.py <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    path = download_audio(video_id)
    if path:
        print(f"Success: {path}")
    else:
        print("Failed to download audio")
        sys.exit(1)
