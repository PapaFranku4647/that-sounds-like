from __future__ import annotations

import numpy as np
import pandas as pd

from thatsoundslike.audio.pooling import pool_song_vectors
from thatsoundslike.embeddings.storage import FeatureStore


def build_song_vectors(
    manifest_df: pd.DataFrame,
    store: FeatureStore,
    pooler: str = "mean",
    section_length_sec: float = 30.0,
    overwrite: bool = False,
) -> dict[str, int]:
    if store.has_song_vectors() and not overwrite:
        rows, vectors = store.load_song_vectors()
        return {
            "songs": int(rows.shape[0]),
            "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0,
            "reused": True,
        }
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for _, row in manifest_df.iterrows():
        segment_vectors, metadata = store.load_segment_vectors(str(row["song_id"]))
        rows.append(row.to_dict())
        vectors.append(
            pool_song_vectors(
                segment_vectors,
                metadata=metadata,
                pooler=pooler,
                section_length_sec=section_length_sec,
            ).astype(np.float32)
        )
    matrix = np.stack(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
    store.save_song_vectors(pd.DataFrame(rows), matrix)
    return {
        "songs": len(rows),
        "embedding_dim": int(matrix.shape[1]) if matrix.size else 0,
        "reused": False,
    }
