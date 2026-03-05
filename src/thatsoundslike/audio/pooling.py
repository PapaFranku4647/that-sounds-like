from __future__ import annotations

import numpy as np
import pandas as pd


def l2_normalize(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, eps)


def mean_pool_segment_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("segment vectors must be a 2D matrix")
    normalized = l2_normalize(vectors, axis=1)
    pooled = normalized.mean(axis=0)
    return l2_normalize(pooled.astype(np.float32), axis=0).reshape(-1)


def _pooled_groups(vectors: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    pooled_groups = [mean_pool_segment_vectors(vectors[group]) for group in groups if group.size]
    if not pooled_groups:
        return mean_pool_segment_vectors(vectors)
    return mean_pool_segment_vectors(np.stack(pooled_groups, axis=0))


def pool_song_vectors(
    vectors: np.ndarray,
    metadata: pd.DataFrame | None = None,
    pooler: str = "mean",
    section_length_sec: float = 30.0,
) -> np.ndarray:
    if pooler == "mean" or metadata is None or metadata.empty:
        return mean_pool_segment_vectors(vectors)
    working = metadata.reset_index(drop=True).copy()
    if "grid_index" not in working.columns:
        working["grid_index"] = 0
    if pooler == "scale_mean":
        groups = [group.index.to_numpy(dtype=int) for _, group in working.groupby("grid_index", sort=True)]
        return _pooled_groups(vectors, groups)
    if pooler == "section_mean":
        if "start_sec" not in working.columns or "end_sec" not in working.columns:
            return mean_pool_segment_vectors(vectors)
        midpoints = (working["start_sec"].astype(float) + working["end_sec"].astype(float)) / 2.0
        working["section_index"] = np.floor(midpoints / max(float(section_length_sec), 1.0)).astype(int)
        groups = [
            group.index.to_numpy(dtype=int)
            for _, group in working.groupby(["grid_index", "section_index"], sort=True)
        ]
        return _pooled_groups(vectors, groups)
    raise ValueError(f"Unsupported pooler: {pooler}")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = l2_normalize(left.astype(np.float32), axis=0)
    right_norm = l2_normalize(right.astype(np.float32), axis=0)
    return float(np.dot(left_norm, right_norm))
