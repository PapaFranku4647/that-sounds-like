from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import workspace_temp_dir, write_sine_wave
from thatsoundslike.eval.benchmark import run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_stats_benchmark_runs_end_to_end(self) -> None:
        with workspace_temp_dir() as root:
            audio_dir = root / "audio"
            write_sine_wave(audio_dir / "a1.wav", 220.0, 0.5)
            write_sine_wave(audio_dir / "a2.wav", 230.0, 0.5)
            write_sine_wave(audio_dir / "b1.wav", 880.0, 0.5)
            write_sine_wave(audio_dir / "b2.wav", 900.0, 0.5)

            manifest = pd.DataFrame(
                [
                    {"song_id": "a1", "artist": "Artist A", "album": "Album A", "title": "One", "source_path": str(audio_dir / "a1.wav")},
                    {"song_id": "a2", "artist": "Artist A", "album": "Album A", "title": "Two", "source_path": str(audio_dir / "a2.wav")},
                    {"song_id": "b1", "artist": "Artist B", "album": "Album B", "title": "Three", "source_path": str(audio_dir / "b1.wav")},
                    {"song_id": "b2", "artist": "Artist B", "album": "Album B", "title": "Four", "source_path": str(audio_dir / "b2.wav")},
                ]
            )
            manifest_path = root / "library.csv"
            manifest.to_csv(manifest_path, index=False)

            named_queries = pd.DataFrame(
                [
                    {"name": "Nearest A1", "query_type": "nearest", "song_id": "a1", "top_k": 1},
                    {"name": "A centroid to B", "query_type": "centroid", "source_artist": "Artist A", "target_artist": "Artist B", "top_k": 1},
                ]
            )
            named_queries_path = root / "named_queries.csv"
            named_queries.to_csv(named_queries_path, index=False)

            triplets = pd.DataFrame(
                [{"anchor_song_id": "a1", "positive_song_id": "a2", "negative_song_id": "b1"}]
            )
            triplets_path = root / "triplets.csv"
            triplets.to_csv(triplets_path, index=False)

            config = {
                "name": "test-benchmark",
                "manifest": str(manifest_path),
                "targets": [{"model": "stats", "profile": "baseline_v1"}],
                "named_queries": str(named_queries_path),
                "triplets": str(triplets_path),
                "paths": {
                    "features_dir": str(root / "features"),
                    "reports_dir": str(root / "reports"),
                    "indexes_dir": str(root / "indexes"),
                    "canonical_dir": str(root / "canonical"),
                },
                "storage": {"prefer_parquet": False},
            }

            report = run_benchmark(config)
            self.assertEqual(report["name"], "test-benchmark")
            self.assertEqual(report["results"][0]["model"], "stats")
            self.assertEqual(report["results"][0]["profile"], "baseline_v1")
            self.assertEqual(report["results"][0]["song_count"], 4)
            self.assertIn("same_artist_recall@1", report["results"][0]["metrics"])
            self.assertEqual(len(report["results"][0]["named_queries"]), 2)
            self.assertEqual(report["results"][0]["named_query_summary"]["total_queries"], 2)


if __name__ == "__main__":
    unittest.main()
