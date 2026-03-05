from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import workspace_temp_dir, write_sine_wave
from thatsoundslike.ingest.manifest import normalize_manifest, validate_manifest


class ManifestTests(unittest.TestCase):
    def test_normalize_manifest_generates_song_id_and_resolves_paths(self) -> None:
        with workspace_temp_dir() as root:
            source = root / "folder with  two spaces" / "song.wav"
            write_sine_wave(source, frequency_hz=220.0, duration_sec=0.25, sample_rate=8000)
            manifest_path = root / "manifest.csv"
            frame = pd.DataFrame(
                [
                    {
                        "artist": "Artist",
                        "title": "Song",
                        "source_path": str(source.relative_to(root)),
                        "album": "Album",
                    }
                ]
            )
            normalized = normalize_manifest(frame, manifest_path)
            self.assertTrue(normalized.loc[0, "song_id"])
            self.assertEqual(normalized.loc[0, "source_path"], str(source.resolve()))
            self.assertEqual(validate_manifest(normalized), [])

    def test_validate_manifest_reports_duplicates(self) -> None:
        with workspace_temp_dir() as root:
            source = root / "song.wav"
            write_sine_wave(source, frequency_hz=330.0, duration_sec=0.25, sample_rate=8000)
            frame = pd.DataFrame(
                [
                    {"song_id": "dup", "artist": "Artist", "title": "One", "source_path": str(source)},
                    {"song_id": "dup", "artist": "Artist", "title": "Two", "source_path": str(source)},
                ]
            )
            normalized = normalize_manifest(frame, root / "manifest.csv")
            errors = validate_manifest(normalized)
            self.assertTrue(any("Duplicate song_id" in error for error in errors))

    def test_normalize_manifest_recovers_local_music_raw_from_windows_path(self) -> None:
        with workspace_temp_dir() as root:
            raw_source = root / "music_raw" / "Artist Name" / "Album Name" / "Song Name.mp3"
            write_sine_wave(raw_source, frequency_hz=220.0, duration_sec=0.25, sample_rate=8000)
            manifest_dir = root / "artifacts" / "indexes"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "library_manifest.csv"
            frame = pd.DataFrame(
                [
                    {
                        "artist": "Artist Name",
                        "album": "Album Name",
                        "title": "Song Name",
                        "source_path": r"C:\Users\pipin\Desktop\Coding\Personal\2026\ThatSoundsLike_v2\music_raw\Artist Name\Album Name\Song Name.mp3",
                        "source_rel_path": "Artist Name/Album Name/Song Name.mp3",
                    }
                ]
            )
            normalized = normalize_manifest(frame, manifest_path)
            self.assertEqual(normalized.loc[0, "source_path"], str(raw_source.resolve()))
            self.assertEqual(validate_manifest(normalized), [])

    def test_normalize_manifest_normalizes_source_rel_path_separators(self) -> None:
        with workspace_temp_dir() as root:
            raw_source = root / "music_raw" / "Artist Name" / "Album Name" / "Song Name.mp3"
            write_sine_wave(raw_source, frequency_hz=220.0, duration_sec=0.25, sample_rate=8000)
            manifest_path = root / "artifacts" / "indexes" / "library_manifest.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(
                [
                    {
                        "artist": "Artist Name",
                        "album": "Album Name",
                        "title": "Song Name",
                        "source_path": str(raw_source),
                        "source_rel_path": r"Artist Name\Album Name\Song Name.mp3",
                    }
                ]
            )
            normalized = normalize_manifest(frame, manifest_path)
            self.assertEqual(normalized.loc[0, "source_rel_path"], "Artist Name/Album Name/Song Name.mp3")


if __name__ == "__main__":
    unittest.main()
