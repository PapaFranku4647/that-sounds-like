from __future__ import annotations

import numpy as np
import pandas as pd

from thatsoundslike.audio.pooling import l2_normalize
from thatsoundslike.retrieval.cosine import song_index, top_k_rows


def _artist_mask(rows: pd.DataFrame, artist: str) -> pd.Series:
    return rows["artist"].fillna("").str.casefold() == artist.casefold()


def artist_centroid(
    rows: pd.DataFrame,
    vectors: np.ndarray,
    artist: str,
    exclude_song_id: str | None = None,
) -> np.ndarray:
    mask = _artist_mask(rows, artist)
    if exclude_song_id is not None:
        mask &= rows["song_id"] != exclude_song_id
    subset = vectors[mask.to_numpy()]
    if subset.shape[0] == 0:
        raise ValueError(f"No songs available to build centroid for artist '{artist}'")
    centroid = subset.mean(axis=0)
    return l2_normalize(centroid.astype(np.float32), axis=0).reshape(-1)


def centroid_query(
    rows: pd.DataFrame,
    vectors: np.ndarray,
    source_artist: str,
    target_artist: str,
    top_k: int = 10,
) -> pd.DataFrame:
    query = artist_centroid(rows, vectors, source_artist)
    target_mask = _artist_mask(rows, target_artist)
    target_rows = rows.loc[target_mask].reset_index(drop=True)
    target_vectors = vectors[target_mask.to_numpy()]
    if target_vectors.shape[0] == 0:
        raise ValueError(f"No songs found for target artist '{target_artist}'")
    return top_k_rows(target_rows, target_vectors, query, top_k=top_k)


def most_artist_like_within_artist(
    rows: pd.DataFrame,
    vectors: np.ndarray,
    artist: str,
    top_k: int = 1,
) -> pd.DataFrame:
    mask = _artist_mask(rows, artist)
    artist_rows = rows.loc[mask].reset_index(drop=True)
    artist_vectors = vectors[mask.to_numpy()]
    if artist_vectors.shape[0] < 2:
        raise ValueError(f"Need at least 2 songs for artist '{artist}'")
    results: list[dict[str, object]] = []
    for _, row in artist_rows.iterrows():
        centroid = artist_centroid(artist_rows, artist_vectors, artist, exclude_song_id=str(row["song_id"]))
        index = song_index(artist_rows, str(row["song_id"]))
        score = float(np.dot(l2_normalize(artist_vectors[index], axis=0), centroid))
        payload = row.to_dict()
        payload["score"] = score
        results.append(payload)
    output = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return output.head(top_k)
