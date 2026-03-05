from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def probe_media(path: str | Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    audio_path = Path(path)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
    )
    stdout = result.stdout.decode("utf-8", errors="replace") if isinstance(result.stdout, bytes) else result.stdout
    payload = json.loads(stdout or "{}")
    format_info = payload.get("format", {})
    streams = payload.get("streams", [])
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    duration = format_info.get("duration") or audio_stream.get("duration")
    sample_rate = audio_stream.get("sample_rate")
    channels = audio_stream.get("channels")
    raw_tags = format_info.get("tags", {}) or {}
    tags = {str(key).casefold(): value for key, value in raw_tags.items()}
    return {
        "duration_sec": float(duration) if duration is not None else None,
        "sample_rate": int(sample_rate) if sample_rate else None,
        "channels": int(channels) if channels else None,
        "codec_name": audio_stream.get("codec_name"),
        "bit_rate": int(format_info["bit_rate"]) if format_info.get("bit_rate") else None,
        "tags": tags,
    }


def probe_audio(path: str | Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    media = probe_media(path, ffprobe_bin=ffprobe_bin)
    return {
        "duration_sec": media["duration_sec"],
        "sample_rate": media["sample_rate"],
        "channels": media["channels"],
        "codec_name": media["codec_name"],
    }
