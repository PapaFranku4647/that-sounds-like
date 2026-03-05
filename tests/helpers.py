from __future__ import annotations

import math
import shutil
import subprocess
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path


def write_sine_wave(path: str | Path, frequency_hz: float, duration_sec: float, sample_rate: int = 24000) -> None:
    output_path = Path(path)
    frames = int(duration_sec * sample_rate)
    amplitude = 0.4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for frame in range(frames):
            value = int(
                max(-1.0, min(1.0, amplitude * math.sin(2.0 * math.pi * frequency_hz * frame / sample_rate)))
                * 32767
            )
            handle.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


def transcode_to_mp3_with_tags(source_path: str | Path, output_path: str | Path, **tags: str) -> None:
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(source_path)]
    for key, value in tags.items():
        command.extend(["-metadata", f"{key}={value}"])
    command.extend(["-codec:a", "libmp3lame", str(output_path)])
    subprocess.run(command, check=True, capture_output=True)


@contextmanager
def workspace_temp_dir() -> Path:
    temp_root = Path(__file__).resolve().parents[1] / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
