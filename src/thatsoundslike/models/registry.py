from __future__ import annotations

from pathlib import Path
from typing import Any

from thatsoundslike.models.base import EmbeddingModel, ModelDescriptor, StatisticalEmbeddingModel
from thatsoundslike.models.mert import MertEmbeddingModel
from thatsoundslike.models.music2vec import Music2VecEmbeddingModel
from thatsoundslike.settings import PROJECT_ROOT, load_runtime_config


def build_descriptor(config: dict[str, Any]) -> ModelDescriptor:
    layer_indices = config.get("layer_indices")
    if layer_indices is None:
        layer_indices = [int(config.get("layer", -1))]
    return ModelDescriptor(
        name=config["name"],
        kind=config["kind"],
        sample_rate=int(config["sample_rate"]),
        window_sec=float(config["window_sec"]),
        hop_sec=float(config["hop_sec"]),
        embedding_dim=int(config["embedding_dim"]) if config.get("embedding_dim") else None,
        batch_size=int(config.get("batch_size", 4)),
        layer_indices=tuple(int(value) for value in layer_indices),
    )


def create_model(config: dict[str, Any]) -> EmbeddingModel:
    descriptor = build_descriptor(config)
    kind = config["kind"]
    if kind == "stats":
        return StatisticalEmbeddingModel(descriptor, embedding_dim=int(config.get("embedding_dim", 40)))
    if kind == "mert":
        return MertEmbeddingModel(
            descriptor=descriptor,
            model_name=config["model_name"],
            device=config.get("device"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    if kind == "music2vec":
        return Music2VecEmbeddingModel(
            descriptor=descriptor,
            model_name=config["model_name"],
            device=config.get("device"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"Unsupported model kind: {kind}")


def load_named_model(
    name_or_path: str,
    profile_name_or_path: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> EmbeddingModel:
    return create_model(load_runtime_config(name_or_path, profile_name_or_path, project_root=project_root))
