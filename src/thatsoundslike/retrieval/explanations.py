from __future__ import annotations

import numpy as np

from thatsoundslike.embeddings.storage import FeatureStore
from thatsoundslike.retrieval.cosine import normalized_matrix


def best_segment_pair(song_id_a: str, song_id_b: str, store: FeatureStore) -> dict[str, float | int]:
    vectors_a, meta_a = store.load_segment_vectors(song_id_a)
    vectors_b, meta_b = store.load_segment_vectors(song_id_b)
    scores = normalized_matrix(vectors_a) @ normalized_matrix(vectors_b).T
    left_index, right_index = np.unravel_index(int(np.argmax(scores)), scores.shape)
    left_meta = meta_a.iloc[left_index]
    right_meta = meta_b.iloc[right_index]
    return {
        "song_id_a": song_id_a,
        "song_id_b": song_id_b,
        "segment_index_a": int(left_meta["segment_index"]),
        "segment_index_b": int(right_meta["segment_index"]),
        "start_sec_a": float(left_meta["start_sec"]),
        "end_sec_a": float(left_meta["end_sec"]),
        "start_sec_b": float(right_meta["start_sec"]),
        "end_sec_b": float(right_meta["end_sec"]),
        "score": float(scores[left_index, right_index]),
    }
