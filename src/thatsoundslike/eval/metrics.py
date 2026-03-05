from __future__ import annotations

import numpy as np
import pandas as pd

from thatsoundslike.retrieval.cosine import normalized_matrix


def _recall_at_k(mask: np.ndarray, ranking: np.ndarray, k: int) -> float:
    top_indices = ranking[:k]
    return float(mask[top_indices].any())


def same_artist_recall(rows: pd.DataFrame, vectors: np.ndarray, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    normalized = normalized_matrix(vectors)
    scores = normalized @ normalized.T
    np.fill_diagonal(scores, -np.inf)
    eligible = 0
    totals = {k: 0.0 for k in ks}
    artists = rows["artist"].fillna("").astype(str)
    for index in range(len(rows)):
        mask = np.asarray((artists == artists.iloc[index]).to_numpy(), dtype=bool).copy()
        mask[index] = False
        if not mask.any():
            continue
        eligible += 1
        ranking = np.argsort(scores[index])[::-1]
        for k in ks:
            totals[k] += _recall_at_k(mask, ranking, k)
    if eligible == 0:
        return {f"same_artist_recall@{k}": 0.0 for k in ks}
    return {f"same_artist_recall@{k}": totals[k] / eligible for k in ks}


def album_recall(rows: pd.DataFrame, vectors: np.ndarray, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    if "album" not in rows.columns:
        return {f"album_recall@{k}": 0.0 for k in ks}
    normalized = normalized_matrix(vectors)
    scores = normalized @ normalized.T
    np.fill_diagonal(scores, -np.inf)
    eligible = 0
    totals = {k: 0.0 for k in ks}
    albums = rows["album"].fillna("").astype(str)
    for index in range(len(rows)):
        album_name = albums.iloc[index]
        if not album_name:
            continue
        mask = np.asarray((albums == album_name).to_numpy(), dtype=bool).copy()
        mask[index] = False
        if not mask.any():
            continue
        eligible += 1
        ranking = np.argsort(scores[index])[::-1]
        for k in ks:
            totals[k] += _recall_at_k(mask, ranking, k)
    if eligible == 0:
        return {f"album_recall@{k}": 0.0 for k in ks}
    return {f"album_recall@{k}": totals[k] / eligible for k in ks}


def triplet_accuracy(rows: pd.DataFrame, vectors: np.ndarray, triplets: pd.DataFrame) -> float:
    if triplets.empty:
        return 0.0
    normalized = normalized_matrix(vectors)
    lookup = {song_id: index for index, song_id in enumerate(rows["song_id"])}
    correct = 0
    total = 0
    for triplet in triplets.itertuples(index=False):
        if (
            triplet.anchor_song_id not in lookup
            or triplet.positive_song_id not in lookup
            or triplet.negative_song_id not in lookup
        ):
            continue
        total += 1
        anchor = lookup[triplet.anchor_song_id]
        positive = lookup[triplet.positive_song_id]
        negative = lookup[triplet.negative_song_id]
        positive_score = float(np.dot(normalized[anchor], normalized[positive]))
        negative_score = float(np.dot(normalized[anchor], normalized[negative]))
        if positive_score > negative_score:
            correct += 1
    return float(correct) / float(total) if total else 0.0
