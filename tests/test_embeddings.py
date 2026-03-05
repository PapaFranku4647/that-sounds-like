from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.helpers import workspace_temp_dir, write_sine_wave
from thatsoundslike.embeddings.build_segment_vectors import build_segment_vectors
from thatsoundslike.embeddings.build_song_vectors import build_song_vectors
from thatsoundslike.embeddings.storage import FeatureStore
from thatsoundslike.models.registry import create_model


class EmbeddingPipelineTests(unittest.TestCase):
    def test_multiscale_embedding_builds_and_reuses_artifacts(self) -> None:
        with workspace_temp_dir() as root:
            source = root / "audio" / "song.wav"
            write_sine_wave(source, frequency_hz=220.0, duration_sec=1.0, sample_rate=8000)
            manifest = pd.DataFrame(
                [
                    {
                        "song_id": "song-1",
                        "artist": "Artist",
                        "album": "Album",
                        "title": "Song",
                        "source_path": str(source),
                    }
                ]
            )
            model = create_model(
                {
                    "name": "stats",
                    "kind": "stats",
                    "sample_rate": 8000,
                    "window_sec": 0.5,
                    "hop_sec": 0.25,
                    "embedding_dim": 40,
                    "batch_size": 4,
                    "layer_indices": [-1],
                }
            )
            store = FeatureStore(root / "features", model_name="stats", profile_name="multiscale_test", prefer_parquet=False)
            summary = build_segment_vectors(
                manifest,
                model,
                store,
                segment_grids=[
                    {"window_sec": 0.25, "hop_sec": 0.25},
                    {"window_sec": 0.5, "hop_sec": 0.5},
                ],
            )
            self.assertEqual(summary["songs"], 1)
            self.assertGreater(summary["segments"], 1)
            vectors, metadata = store.load_segment_vectors("song-1")
            self.assertEqual(vectors.shape[0], metadata.shape[0])
            self.assertEqual(sorted(metadata["grid_index"].unique().tolist()), [0, 1])

            repeated = build_segment_vectors(manifest, model, store)
            self.assertEqual(repeated["skipped_songs"], 1)

            song_summary = build_song_vectors(manifest, store, pooler="section_mean", overwrite=True)
            self.assertEqual(song_summary["songs"], 1)
            reused = build_song_vectors(manifest, store, pooler="section_mean", overwrite=False)
            self.assertTrue(reused["reused"])

    def test_feature_store_baseline_profile_reads_legacy_layout(self) -> None:
        with workspace_temp_dir() as root:
            legacy_dir = root / "features" / "stats"
            (legacy_dir / "segment_vectors").mkdir(parents=True, exist_ok=True)
            (legacy_dir / "segment_metadata").mkdir(parents=True, exist_ok=True)
            np.save(legacy_dir / "song_vectors.npy", np.array([[1.0, 0.0]], dtype=np.float32))
            pd.DataFrame([{"song_id": "song-1", "artist": "Artist", "title": "Song"}]).to_csv(
                legacy_dir / "song_rows.csv", index=False
            )
            np.save(legacy_dir / "segment_vectors" / "song-1.npy", np.array([[1.0, 0.0]], dtype=np.float32))
            pd.DataFrame([{"segment_index": 0, "start_sec": 0.0, "end_sec": 1.0}]).to_csv(
                legacy_dir / "segment_metadata" / "song-1.csv",
                index=False,
            )

            store = FeatureStore(root / "features", model_name="stats", profile_name="baseline_v1", prefer_parquet=False)
            self.assertTrue(store.has_song_vectors())
            rows, vectors = store.load_song_vectors()
            self.assertEqual(rows.iloc[0]["song_id"], "song-1")
            self.assertEqual(vectors.shape, (1, 2))
            self.assertTrue(store.has_segment_vectors("song-1"))


if __name__ == "__main__":
    unittest.main()
