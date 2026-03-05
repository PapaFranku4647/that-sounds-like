from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thatsoundslike.audio.decode import decode_audio
from thatsoundslike.audio.segment import segment_waveform
from thatsoundslike.embeddings.storage import FeatureStore
from thatsoundslike.models.base import EmbeddingModel


def resolve_audio_path(row: pd.Series) -> Path:
    canonical_path = str(row.get("canonical_path", "")).strip()
    if canonical_path:
        return Path(canonical_path)
    return Path(str(row["source_path"]))


def build_segment_vectors(
    manifest_df: pd.DataFrame,
    model: EmbeddingModel,
    store: FeatureStore,
    ffmpeg_bin: str = "ffmpeg",
    segment_grids: list[dict[str, float]] | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    grids = segment_grids or [
        {
            "window_sec": model.descriptor.window_sec,
            "hop_sec": model.descriptor.hop_sec,
        }
    ]
    song_count = 0
    segment_count = 0
    skipped_count = 0
    for _, row in manifest_df.iterrows():
        song_id = str(row["song_id"])
        if not overwrite and store.has_segment_vectors(song_id):
            skipped_count += 1
            continue
        waveform = decode_audio(
            resolve_audio_path(row),
            sample_rate=model.descriptor.sample_rate,
            ffmpeg_bin=ffmpeg_bin,
        )
        all_vectors: list[np.ndarray] = []
        metadata_rows: list[dict[str, float | int | str]] = []
        running_index = 0
        for grid_index, grid in enumerate(grids):
            segments, specs = segment_waveform(
                waveform=waveform,
                sample_rate=model.descriptor.sample_rate,
                window_sec=float(grid["window_sec"]),
                hop_sec=float(grid["hop_sec"]),
            )
            vectors = model.embed_segments(segments)
            all_vectors.append(vectors)
            for spec in specs:
                metadata_rows.append(
                    {
                        "song_id": song_id,
                        "segment_index": running_index + spec.segment_index,
                        "segment_index_within_grid": spec.segment_index,
                        "grid_index": grid_index,
                        "window_sec": float(grid["window_sec"]),
                        "hop_sec": float(grid["hop_sec"]),
                        "start_frame": spec.start_frame,
                        "end_frame": spec.end_frame,
                        "start_sec": spec.start_sec,
                        "end_sec": spec.end_sec,
                    }
                )
            running_index += len(specs)
        matrix = np.concatenate(all_vectors, axis=0) if all_vectors else np.zeros((0, 0), dtype=np.float32)
        metadata = pd.DataFrame(metadata_rows)
        store.save_segment_vectors(song_id, matrix, metadata)
        song_count += 1
        segment_count += len(metadata_rows)
    return {"songs": song_count, "segments": segment_count, "skipped_songs": skipped_count}
