from __future__ import annotations

import unittest

from tests.helpers import transcode_to_mp3_with_tags, workspace_temp_dir, write_sine_wave
from thatsoundslike.ingest.raw_library import audit_raw_library, scan_raw_library


class RawLibraryTests(unittest.TestCase):
    def test_scan_raw_library_falls_back_to_path_and_sanitizes_title(self) -> None:
        with workspace_temp_dir() as root:
            raw_root = root / "music_raw"
            song_path = raw_root / "Pink Floyd" / "The Wall" / "01_In the Flesh\uff1f.wav"
            write_sine_wave(song_path, frequency_hz=440.0, duration_sec=0.25)
            scanned = scan_raw_library(raw_root)
            self.assertEqual(len(scanned), 1)
            row = scanned.iloc[0]
            self.assertEqual(row["artist"], "Pink Floyd")
            self.assertEqual(row["album"], "The Wall")
            self.assertEqual(row["title"], "In the Flesh?")
            self.assertEqual(row["track_number"], "01")
            self.assertIn("non_ascii_path", row["notes"])

    def test_scan_raw_library_prefers_mp3_tags(self) -> None:
        with workspace_temp_dir() as root:
            raw_root = root / "music_raw"
            wav_path = raw_root / "Artist Folder" / "Album Folder" / "01_Filename Only.wav"
            write_sine_wave(wav_path, frequency_hz=220.0, duration_sec=0.25)
            mp3_path = wav_path.with_suffix(".mp3")
            transcode_to_mp3_with_tags(
                wav_path,
                mp3_path,
                artist="Tagged Artist",
                album="Tagged Album",
                title="Tagged Title",
                date="1973",
            )
            wav_path.unlink()
            scanned = scan_raw_library(raw_root)
            row = scanned.iloc[0]
            self.assertEqual(row["artist"], "Tagged Artist")
            self.assertEqual(row["album"], "Tagged Album")
            self.assertEqual(row["title"], "Tagged Title")
            self.assertEqual(row["year"], "1973")

    def test_audit_raw_library_reports_non_ascii(self) -> None:
        with workspace_temp_dir() as root:
            raw_root = root / "music_raw"
            song_path = raw_root / "Pink Floyd" / "The Wall" / "Is There Anybody Out There\uff1f.wav"
            write_sine_wave(song_path, frequency_hz=330.0, duration_sec=0.25)
            audit = audit_raw_library(raw_root)
            self.assertEqual(audit["file_count"], 1)
            self.assertEqual(len(audit["non_ascii_paths"]), 1)


if __name__ == "__main__":
    unittest.main()
