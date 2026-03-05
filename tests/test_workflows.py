from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from thatsoundslike.eval.workflows import compare_reports, sample_hard_triplets


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            [
                {"song_id": "a1", "artist": "Artist A", "title": "One", "album": "Album A"},
                {"song_id": "a2", "artist": "Artist A", "title": "Two", "album": "Album A"},
                {"song_id": "b1", "artist": "Artist B", "title": "Three", "album": "Album B"},
                {"song_id": "b2", "artist": "Artist B", "title": "Four", "album": "Album B"},
            ]
        )
        self.vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def test_sample_hard_triplets_returns_expected_shape(self) -> None:
        triplets = sample_hard_triplets(self.rows, self.vectors, count=2)
        self.assertEqual(len(triplets), 2)
        self.assertIn("anchor_song_id", triplets.columns)
        self.assertIn("margin", triplets.columns)

    def test_compare_reports_computes_metric_deltas(self) -> None:
        left = {
            "name": "left",
            "results": [{"target": "mert95/baseline_v1", "metrics": {"same_artist_recall@1": 0.4}}],
        }
        right = {
            "name": "right",
            "results": [{"target": "mert95/baseline_v1", "metrics": {"same_artist_recall@1": 0.6}}],
        }
        comparison = compare_reports(left, right)
        self.assertEqual(comparison["left_name"], "left")
        self.assertEqual(comparison["right_name"], "right")
        self.assertEqual(comparison["shared_targets"][0]["metrics"][0]["delta"], 0.2)


if __name__ == "__main__":
    unittest.main()
