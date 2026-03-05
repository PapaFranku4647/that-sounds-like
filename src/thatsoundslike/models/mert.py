from __future__ import annotations

from typing import Any

import numpy as np

from .base import EmbeddingModel, ModelDescriptor


class MertEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        descriptor: ModelDescriptor,
        model_name: str,
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(descriptor)
        self.model_name = model_name
        self.requested_device = device
        self.trust_remote_code = trust_remote_code
        self._feature_extractor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install the 'ml' extras to use MERT models.") from exc
        self._torch = torch
        device_name = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_name)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        ).to(self._device).eval()

    def _pooled_hidden_state(self, model_outputs: Any):
        assert self._torch is not None
        hidden_states = getattr(model_outputs, "hidden_states", None)
        if not hidden_states:
            return model_outputs.last_hidden_state.mean(dim=1)
        selected = [hidden_states[index].mean(dim=1) for index in self.descriptor.layer_indices]
        if len(selected) == 1:
            return selected[0]
        stacked = self._torch.stack(selected, dim=0)
        return stacked.mean(dim=0)

    def embed_segments(self, segments: np.ndarray) -> np.ndarray:
        self._lazy_load()
        assert self._feature_extractor is not None
        assert self._model is not None
        assert self._torch is not None
        outputs: list[np.ndarray] = []
        batch_size = self.descriptor.batch_size
        for start in range(0, segments.shape[0], batch_size):
            batch = [segment.astype(np.float32, copy=False) for segment in segments[start : start + batch_size]]
            encoded = self._feature_extractor(
                batch,
                sampling_rate=self.descriptor.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.no_grad():
                model_outputs = self._model(**encoded, output_hidden_states=True)
            pooled = self._pooled_hidden_state(model_outputs).detach().cpu().numpy().astype(np.float32)
            outputs.append(pooled)
        matrix = np.concatenate(outputs, axis=0)
        self.descriptor.embedding_dim = int(matrix.shape[1])
        return matrix
