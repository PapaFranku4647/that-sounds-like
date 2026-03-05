from __future__ import annotations

import math
import unittest

import pandas as pd

from thatsoundslike.presentation import json_ready, nearest_payload, song_brief


class PresentationTests(unittest.TestCase):
    def test_song_brief_normalizes_rel_path_and_drops_nan(self) -> None:
        payload = song_brief(
            {
                "song_id": "song-1",
                "artist": "Artist",
                "album": "Album",
                "title": "Title",
                "duration_sec": 123.45678,
                "source_rel_path": r"Artist\Album\Title.mp3",
                "score": math.nan,
            }
        )
        self.assertEqual(payload["source_rel_path"], "Artist/Album/Title.mp3")
        self.assertEqual(payload["duration_sec"], 123.457)
        self.assertNotIn("score", payload)

    def test_nearest_payload_is_json_safe(self) -> None:
        rows = pd.DataFrame(
            [
                {"song_id": "a1", "artist": "Artist A", "album": "Album A", "title": "One", "source_rel_path": "Artist A/Album A/One.mp3"},
                {"song_id": "a2", "artist": "Artist A", "album": "Album A", "title": "Two", "source_rel_path": "Artist A/Album A/Two.mp3"},
            ]
        )
        matches = pd.DataFrame(
            [
                {"song_id": "a2", "artist": "Artist A", "album": "Album A", "title": "Two", "source_rel_path": r"Artist A\Album A\Two.mp3", "score": 0.987654321},
            ]
        )
        payload = nearest_payload(rows, "a1", matches)
        safe = json_ready(payload)
        self.assertEqual(safe["matches"][0]["score"], 0.987654)
        self.assertEqual(safe["matches"][0]["source_rel_path"], "Artist A/Album A/Two.mp3")


if __name__ == "__main__":
    unittest.main()
