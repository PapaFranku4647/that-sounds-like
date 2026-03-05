from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from thatsoundslike.audio.pooling import mean_pool_segment_vectors, pool_song_vectors


class PoolingTests(unittest.TestCase):
    def test_mean_pool_normalizes_output(self) -> None:
        pooled = mean_pool_segment_vectors(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(pooled, np.array([2 ** -0.5, 2 ** -0.5], dtype=np.float32), atol=1e-6)

    def test_section_mean_pooling_balances_sections(self) -> None:
        vectors = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        metadata = pd.DataFrame(
            [
                {"grid_index": 0, "start_sec": 0.0, "end_sec": 5.0},
                {"grid_index": 0, "start_sec": 5.0, "end_sec": 10.0},
                {"grid_index": 0, "start_sec": 30.0, "end_sec": 35.0},
                {"grid_index": 0, "start_sec": 35.0, "end_sec": 40.0},
            ]
        )
        pooled = pool_song_vectors(vectors, metadata=metadata, pooler="section_mean", section_length_sec=30.0)
        np.testing.assert_allclose(pooled, np.array([2 ** -0.5, 2 ** -0.5], dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
