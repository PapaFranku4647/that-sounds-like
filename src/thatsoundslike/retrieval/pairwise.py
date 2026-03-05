from __future__ import annotations

import numpy as np
import pandas as pd

from thatsoundslike.retrieval.cosine import normalized_matrix, song_index, top_k_rows


def nearest_neighbors(
    rows: pd.DataFrame,
    vectors: np.ndarray,
    song_id: str,
    top_k: int = 10,
) -> pd.DataFrame:
    index = song_index(rows, song_id)
    return top_k_rows(rows, vectors, vectors[index], top_k=top_k, exclude_indices={index})


def best_cross_artist_pair(
    rows: pd.DataFrame,
    vectors: np.ndarray,
    artist_a: str,
    artist_b: str,
) -> dict[str, object]:
    mask_a = rows["artist"].fillna("").str.casefold() == artist_a.casefold()
    mask_b = rows["artist"].fillna("").str.casefold() == artist_b.casefold()
    rows_a = rows.loc[mask_a].reset_index(drop=True)
    rows_b = rows.loc[mask_b].reset_index(drop=True)
    vectors_a = normalized_matrix(vectors[mask_a.to_numpy()])
    vectors_b = normalized_matrix(vectors[mask_b.to_numpy()])
    if vectors_a.shape[0] == 0:
        raise ValueError(f"No songs found for artist '{artist_a}'")
    if vectors_b.shape[0] == 0:
        raise ValueError(f"No songs found for artist '{artist_b}'")
    scores = vectors_a @ vectors_b.T
    left_index, right_index = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return {
        "artist_a_song_id": rows_a.iloc[left_index]["song_id"],
        "artist_a_title": rows_a.iloc[left_index]["title"],
        "artist_b_song_id": rows_b.iloc[right_index]["song_id"],
        "artist_b_title": rows_b.iloc[right_index]["title"],
        "score": float(scores[left_index, right_index]),
    }
