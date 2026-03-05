from __future__ import annotations

import unittest

import pandas as pd

from tests.helpers import workspace_temp_dir, write_sine_wave
from thatsoundslike.ingest.canonicalize import canonicalize_manifest


class CanonicalizeTests(unittest.TestCase):
    def test_canonicalize_manifest_handles_csv_loaded_nan_canonical_path(self) -> None:
        with workspace_temp_dir() as root:
            source = root / "song.wav"
            write_sine_wave(source, frequency_hz=220.0, duration_sec=0.25, sample_rate=8000)
            manifest = pd.DataFrame(
                [
                    {
                        "song_id": "song-1",
                        "artist": "Artist",
                        "album": "Album",
                        "title": "Song",
                        "source_path": str(source),
                        "canonical_path": None,
                    }
                ]
            )
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            loaded = pd.read_csv(manifest_path)
            output = canonicalize_manifest(
                manifest_df=loaded,
                output_dir=root / "canonical",
                sample_rate=8000,
                channels=1,
                codec="flac",
            )
            self.assertTrue(output.loc[0, "canonical_path"].endswith("song-1.flac"))


if __name__ == "__main__":
    unittest.main()
