from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def decode_audio(
    path: str | Path,
    sample_rate: int,
    channels: int = 1,
    ffmpeg_bin: str = "ffmpeg",
) -> np.ndarray:
    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(Path(path)),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if channels == 1:
        return waveform
    return waveform.reshape(-1, channels)


def duration_seconds(waveform: np.ndarray, sample_rate: int) -> float:
    return float(waveform.shape[0]) / float(sample_rate)
