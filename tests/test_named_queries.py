from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from thatsoundslike.eval.named_queries import run_named_queries, summarize_named_queries


class NamedQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            [
                {"song_id": "a1", "artist": "Artist A", "title": "A One", "album": "Album A"},
                {"song_id": "a2", "artist": "Artist A", "title": "A Two", "album": "Album A"},
                {"song_id": "b1", "artist": "Artist B", "title": "B One", "album": "Album B"},
                {"song_id": "b2", "artist": "Artist B", "title": "B Two", "album": "Album B"},
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

    def test_named_queries_include_evaluation_summary(self) -> None:
        query_set = pd.DataFrame(
            [
                {
                    "name": "Centroid A to B",
                    "query_type": "centroid",
                    "source_artist": "Artist A",
                    "target_artist": "Artist B",
                    "top_k": 3,
                    "expected_song_id": "b2",
                },
                {
                    "name": "Pair A to B",
                    "query_type": "pair",
                    "artist_a": "Artist A",
                    "artist_b": "Artist B",
                    "expected_song_id_a": "a2",
                    "expected_song_id_b": "b2",
                },
            ]
        )
        results = run_named_queries(self.rows, self.vectors, query_set)
        summary = summarize_named_queries(results)
        self.assertEqual(summary["total_queries"], 2)
        self.assertEqual(summary["evaluated_queries"], 2)
        self.assertEqual(summary["passed_queries"], 2)
        self.assertAlmostEqual(summary["match_rate"], 1.0)
        self.assertTrue(all(result["evaluation"]["matched"] for result in results))


if __name__ == "__main__":
    unittest.main()
