from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from thatsoundslike.audio.pooling import l2_normalize


@dataclass
class ModelDescriptor:
    name: str
    kind: str
    sample_rate: int
    window_sec: float
    hop_sec: float
    embedding_dim: int | None = None
    batch_size: int = 4
    layer_indices: tuple[int, ...] = (-1,)


class EmbeddingModel(ABC):
    def __init__(self, descriptor: ModelDescriptor) -> None:
        self.descriptor = descriptor

    @abstractmethod
    def embed_segments(self, segments: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class StatisticalEmbeddingModel(EmbeddingModel):
    def __init__(self, descriptor: ModelDescriptor, embedding_dim: int = 40) -> None:
        descriptor.embedding_dim = embedding_dim
        super().__init__(descriptor)
        self.band_count = max(1, embedding_dim - 20)

    def embed_segments(self, segments: np.ndarray) -> np.ndarray:
        vectors = np.stack([self._segment_features(segment) for segment in segments], axis=0)
        return l2_normalize(vectors.astype(np.float32), axis=1)

    def _segment_features(self, segment: np.ndarray) -> np.ndarray:
        if not np.any(segment):
            return np.zeros(self.descriptor.embedding_dim or 40, dtype=np.float32)
        sample_rate = self.descriptor.sample_rate
        window = np.hanning(segment.shape[0]).astype(np.float32)
        weighted = segment * window
        magnitude = np.abs(np.fft.rfft(weighted)).astype(np.float32) + 1e-8
        power = magnitude**2
        freqs = np.fft.rfftfreq(segment.shape[0], d=1.0 / sample_rate).astype(np.float32)
        normalized_power = power / np.sum(power)
        centroid = float(np.sum(freqs * normalized_power) / max(sample_rate / 2.0, 1.0))
        bandwidth = float(
            np.sqrt(np.sum(((freqs - centroid * (sample_rate / 2.0)) ** 2) * normalized_power))
            / max(sample_rate / 2.0, 1.0)
        )
        cumulative = np.cumsum(normalized_power)
        rolloff_index = int(np.searchsorted(cumulative, 0.85))
        rolloff = float(freqs[min(rolloff_index, freqs.shape[0] - 1)] / max(sample_rate / 2.0, 1.0))
        flatness = float(np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude))
        mean_abs = float(np.mean(np.abs(segment)))
        std = float(np.std(segment))
        rms = float(np.sqrt(np.mean(segment**2)))
        peak = float(np.max(np.abs(segment)))
        crest = peak / max(rms, 1e-8)
        zero_cross = float(np.mean(np.abs(np.diff(np.signbit(segment)))))
        dynamic_range = float(np.percentile(np.abs(segment), 95) - np.percentile(np.abs(segment), 5))
        abs_quantiles = np.quantile(np.abs(segment), [0.25, 0.5, 0.75, 0.9]).astype(np.float32)
        low_ratio = float(np.sum(normalized_power[freqs < 250]))
        mid_ratio = float(np.sum(normalized_power[(freqs >= 250) & (freqs < 4000)]))
        high_ratio = float(np.sum(normalized_power[freqs >= 4000]))
        entropy = float(
            -np.sum(normalized_power * np.log(normalized_power + 1e-12))
            / np.log(normalized_power.shape[0])
        )
        max_ratio = float(np.max(normalized_power))
        broad_features = np.array(
            [
                mean_abs,
                std,
                rms,
                peak,
                crest,
                zero_cross,
                centroid,
                bandwidth,
                rolloff,
                flatness,
                low_ratio,
                mid_ratio,
                high_ratio,
                dynamic_range,
                float(abs_quantiles[0]),
                float(abs_quantiles[1]),
                float(abs_quantiles[2]),
                float(abs_quantiles[3]),
                entropy,
                max_ratio,
            ],
            dtype=np.float32,
        )
        band_edges = np.geomspace(20.0, sample_rate / 2.0 + 1.0, num=self.band_count + 1)
        band_features = np.zeros(self.band_count, dtype=np.float32)
        for band_index in range(self.band_count):
            left = band_edges[band_index]
            right = band_edges[band_index + 1]
            if band_index == self.band_count - 1:
                mask = (freqs >= left) & (freqs <= right)
            else:
                mask = (freqs >= left) & (freqs < right)
            band_features[band_index] = float(np.sum(normalized_power[mask]))
        features = np.concatenate([broad_features, band_features]).astype(np.float32)
        target_dim = self.descriptor.embedding_dim or features.shape[0]
        if features.shape[0] < target_dim:
            padding = np.zeros(target_dim - features.shape[0], dtype=np.float32)
            features = np.concatenate([features, padding])
        return features[:target_dim]
