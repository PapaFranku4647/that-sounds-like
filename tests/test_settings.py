from __future__ import annotations

import unittest

from thatsoundslike.settings import PROJECT_ROOT, load_runtime_config


class SettingsTests(unittest.TestCase):
    def test_load_runtime_config_merges_profile_settings(self) -> None:
        config = load_runtime_config("mert95", "multiscale_v1", project_root=PROJECT_ROOT)
        self.assertEqual(config["name"], "mert95")
        self.assertEqual(config["profile_name"], "multiscale_v1")
        self.assertEqual(config["pooler"], "section_mean")
        self.assertEqual(len(config["segment_grids"]), 3)
        self.assertEqual(config["layer_indices"], [-1, -3, -5])


if __name__ == "__main__":
    unittest.main()
