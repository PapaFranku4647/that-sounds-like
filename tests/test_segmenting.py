from __future__ import annotations

import unittest

import numpy as np

from thatsoundslike.audio.segment import build_segment_specs, segment_waveform


class SegmentingTests(unittest.TestCase):
    def test_short_waveform_is_padded_to_single_segment(self) -> None:
        waveform = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        segments, specs = segment_waveform(waveform, sample_rate=4, window_sec=2.0, hop_sec=1.0)
        self.assertEqual(segments.shape, (1, 8))
        self.assertEqual(specs[0].start_frame, 0)
        self.assertEqual(specs[0].end_frame, 3)
        np.testing.assert_allclose(segments[0, :3], waveform)

    def test_segment_specs_cover_the_tail(self) -> None:
        specs = build_segment_specs(num_samples=20, sample_rate=4, window_sec=2.0, hop_sec=1.0)
        self.assertEqual([spec.start_frame for spec in specs], [0, 4, 8, 12])


if __name__ == "__main__":
    unittest.main()
