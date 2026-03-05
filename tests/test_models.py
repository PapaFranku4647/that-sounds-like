from __future__ import annotations

import unittest

from thatsoundslike.models.mert import MertEmbeddingModel
from thatsoundslike.models.music2vec import Music2VecEmbeddingModel
from thatsoundslike.models.registry import create_model


class ModelRegistryTests(unittest.TestCase):
    def test_create_model_passes_trust_remote_code_to_mert(self) -> None:
        model = create_model(
            {
                "name": "mert95",
                "kind": "mert",
                "model_name": "m-a-p/MERT-v1-95M",
                "sample_rate": 24000,
                "window_sec": 8.0,
                "hop_sec": 4.0,
                "batch_size": 4,
                "layer_indices": [-1, -3],
                "trust_remote_code": True,
            }
        )
        self.assertIsInstance(model, MertEmbeddingModel)
        self.assertTrue(model.trust_remote_code)
        self.assertEqual(model.descriptor.layer_indices, (-1, -3))

    def test_create_model_passes_trust_remote_code_to_music2vec(self) -> None:
        model = create_model(
            {
                "name": "music2vec",
                "kind": "music2vec",
                "model_name": "m-a-p/music2vec-v1",
                "sample_rate": 24000,
                "window_sec": 8.0,
                "hop_sec": 4.0,
                "batch_size": 4,
                "layer_indices": [-1, -5],
                "trust_remote_code": True,
            }
        )
        self.assertIsInstance(model, Music2VecEmbeddingModel)
        self.assertTrue(model.trust_remote_code)
        self.assertEqual(model.descriptor.layer_indices, (-1, -5))


if __name__ == "__main__":
    unittest.main()
