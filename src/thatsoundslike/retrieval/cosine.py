from __future__ import annotations

import numpy as np
import pandas as pd

from thatsoundslike.audio.pooling import l2_normalize


def normalized_matrix(vectors: np.ndarray) -> np.ndarray:
    return l2_normalize(vectors.astype(np.float32), axis=1)


def song_index(rows: pd.DataFrame, song_id: str) -> int:
    matches = rows.index[rows["song_id"] == song_id]
    if len(matches) == 0:
        raise KeyError(f"Unknown song_id: {song_id}")
    return int(matches[0])


def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    normalized_query = l2_normalize(query_vector.astype(np.float32), axis=0).reshape(-1)
    normalized = normalized_matrix(matrix)
    return normalized @ normalized_query


def top_k_rows(
    rows: pd.DataFrame,
    matrix: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = 10,
    exclude_indices: set[int] | None = None,
) -> pd.DataFrame:
    scores = cosine_scores(query_vector, matrix)
    if exclude_indices:
        for index in exclude_indices:
            scores[index] = -np.inf
    selected = np.argsort(scores)[::-1][:top_k]
    results = rows.iloc[selected].copy().reset_index(drop=True)
    results["score"] = scores[selected]
    return results
