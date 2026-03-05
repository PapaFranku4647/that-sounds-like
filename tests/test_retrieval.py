from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from thatsoundslike.retrieval.centroid import centroid_query, most_artist_like_within_artist
from thatsoundslike.retrieval.pairwise import best_cross_artist_pair, nearest_neighbors


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            [
                {"song_id": "a1", "artist": "Artist A", "title": "A One", "album": "X"},
                {"song_id": "a2", "artist": "Artist A", "title": "A Two", "album": "X"},
                {"song_id": "b1", "artist": "Artist B", "title": "B One", "album": "Y"},
                {"song_id": "b2", "artist": "Artist B", "title": "B Two", "album": "Y"},
            ]
        )
        self.vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        )

    def test_nearest_neighbor_excludes_self(self) -> None:
        results = nearest_neighbors(self.rows, self.vectors, "a1", top_k=1)
        self.assertEqual(results.iloc[0]["song_id"], "a2")

    def test_centroid_query_returns_target_artist_song(self) -> None:
        results = centroid_query(self.rows, self.vectors, "Artist A", "Artist B", top_k=1)
        self.assertEqual(results.iloc[0]["song_id"], "b2")

    def test_most_artist_like_within_artist(self) -> None:
        results = most_artist_like_within_artist(self.rows, self.vectors, "Artist A", top_k=1)
        self.assertEqual(results.iloc[0]["song_id"], "a1")

    def test_best_cross_artist_pair(self) -> None:
        result = best_cross_artist_pair(self.rows, self.vectors, "Artist A", "Artist B")
        self.assertEqual(result["artist_b_song_id"], "b2")


if __name__ == "__main__":
    unittest.main()
