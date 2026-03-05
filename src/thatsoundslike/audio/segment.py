from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SegmentSpec:
    segment_index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


def _unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted(set(values))


def build_segment_specs(
    num_samples: int,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> list[SegmentSpec]:
    if num_samples <= 0:
        raise ValueError("waveform must contain at least one sample")
    window_frames = max(1, int(round(window_sec * sample_rate)))
    hop_frames = max(1, int(round(hop_sec * sample_rate)))
    starts = [0]
    if num_samples > window_frames:
        starts = list(range(0, num_samples - window_frames + 1, hop_frames))
        last_start = max(0, num_samples - window_frames)
        starts.append(last_start)
        starts = _unique_sorted(starts)
    specs: list[SegmentSpec] = []
    for index, start_frame in enumerate(starts):
        end_frame = min(num_samples, start_frame + window_frames)
        specs.append(
            SegmentSpec(
                segment_index=index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start_frame / sample_rate,
                end_sec=end_frame / sample_rate,
            )
        )
    return specs


def segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> tuple[np.ndarray, list[SegmentSpec]]:
    specs = build_segment_specs(
        num_samples=waveform.shape[0],
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    window_frames = max(1, int(round(window_sec * sample_rate)))
    segments = np.zeros((len(specs), window_frames), dtype=np.float32)
    for spec in specs:
        raw = waveform[spec.start_frame : spec.end_frame]
        segments[spec.segment_index, : raw.shape[0]] = raw.astype(np.float32, copy=False)
    return segments, specs
