from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from thatsoundslike.retrieval.cosine import normalized_matrix


def sample_hard_triplets(rows: pd.DataFrame, vectors: np.ndarray, count: int = 100) -> pd.DataFrame:
    if rows.empty or vectors.size == 0:
        return pd.DataFrame(columns=["anchor_song_id", "positive_song_id", "negative_song_id", "margin"])
    normalized = normalized_matrix(vectors)
    scores = normalized @ normalized.T
    np.fill_diagonal(scores, -np.inf)
    artists = rows["artist"].fillna("").astype(str)
    triplets: list[dict[str, Any]] = []
    for anchor_index, anchor_row in rows.reset_index(drop=True).iterrows():
        same_mask = (artists == artists.iloc[anchor_index]).to_numpy(dtype=bool)
        same_mask[anchor_index] = False
        diff_mask = ~same_mask
        diff_mask[anchor_index] = False
        if not same_mask.any() or not diff_mask.any():
            continue
        positive_candidates = np.where(same_mask)[0]
        negative_candidates = np.where(diff_mask)[0]
        positive_index = int(positive_candidates[np.argmax(scores[anchor_index, positive_candidates])])
        negative_index = int(negative_candidates[np.argmax(scores[anchor_index, negative_candidates])])
        positive_score = float(scores[anchor_index, positive_index])
        negative_score = float(scores[anchor_index, negative_index])
        triplets.append(
            {
                "anchor_song_id": str(anchor_row["song_id"]),
                "positive_song_id": str(rows.iloc[positive_index]["song_id"]),
                "negative_song_id": str(rows.iloc[negative_index]["song_id"]),
                "anchor_artist": str(anchor_row.get("artist", "")),
                "positive_artist": str(rows.iloc[positive_index].get("artist", "")),
                "negative_artist": str(rows.iloc[negative_index].get("artist", "")),
                "anchor_title": str(anchor_row.get("title", "")),
                "positive_title": str(rows.iloc[positive_index].get("title", "")),
                "negative_title": str(rows.iloc[negative_index].get("title", "")),
                "positive_score": positive_score,
                "negative_score": negative_score,
                "margin": positive_score - negative_score,
            }
        )
    output = pd.DataFrame(triplets).sort_values("margin", ascending=True).reset_index(drop=True)
    return output.head(count)


def compare_reports(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def _targets(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        targets: dict[str, Mapping[str, Any]] = {}
        for result in report.get("results", []):
            target_key = str(result.get("target", result.get("model", "")))
            targets[target_key] = result
        return targets

    left_targets = _targets(left)
    right_targets = _targets(right)
    shared = sorted(set(left_targets) & set(right_targets))
    comparisons: list[dict[str, Any]] = []
    for target in shared:
        left_result = left_targets[target]
        right_result = right_targets[target]
        metrics = sorted(set(left_result.get("metrics", {})) | set(right_result.get("metrics", {})))
        metric_deltas = []
        for metric in metrics:
            left_value = left_result.get("metrics", {}).get(metric)
            right_value = right_result.get("metrics", {}).get(metric)
            delta = None
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                delta = round(float(right_value) - float(left_value), 10)
            metric_deltas.append(
                {
                    "metric": metric,
                    "left": left_value,
                    "right": right_value,
                    "delta": delta,
                }
            )
        comparisons.append({"target": target, "metrics": metric_deltas})
    return {
        "left_name": left.get("name", "left"),
        "right_name": right.get("name", "right"),
        "shared_targets": comparisons,
        "left_only_targets": sorted(set(left_targets) - set(right_targets)),
        "right_only_targets": sorted(set(right_targets) - set(left_targets)),
    }
