from __future__ import annotations

from pathlib import Path

import pandas as pd

from thatsoundslike.embeddings.build_segment_vectors import build_segment_vectors
from thatsoundslike.embeddings.build_song_vectors import build_song_vectors
from thatsoundslike.embeddings.storage import FeatureStore
from thatsoundslike.eval.datasets import load_optional_csv
from thatsoundslike.eval.metrics import album_recall, same_artist_recall, triplet_accuracy
from thatsoundslike.eval.named_queries import run_named_queries, summarize_named_queries
from thatsoundslike.ingest.manifest import load_manifest, normalize_manifest, validate_manifest
from thatsoundslike.models.registry import load_named_model
from thatsoundslike.settings import (
    PROJECT_ROOT,
    ensure_project_directories,
    load_runtime_config,
    resolve_path,
)


def _optional_dataset(config: dict[str, object], key: str, project_root: Path) -> pd.DataFrame:
    raw = str(config.get(key, "")).strip()
    if not raw:
        return pd.DataFrame()
    return load_optional_csv(resolve_path(raw, root=project_root))


def _expand_targets(experiment_config: dict[str, object]) -> list[dict[str, str]]:
    if experiment_config.get("targets"):
        targets: list[dict[str, str]] = []
        for target in experiment_config.get("targets", []):
            model_name = str(target["model"])
            profile_name = str(target.get("profile", experiment_config.get("default_profile", "baseline_v1")))
            targets.append({"model": model_name, "profile": profile_name})
        return targets
    models = [str(name) for name in experiment_config.get("models", [])]
    profiles = [str(name) for name in experiment_config.get("profiles", [])]
    if not profiles:
        profiles = [str(experiment_config.get("default_profile", "baseline_v1"))]
    return [{"model": model_name, "profile": profile_name} for model_name in models for profile_name in profiles]


def run_benchmark(
    experiment_config: dict[str, object],
    project_root: Path = PROJECT_ROOT,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, object]:
    resolved_paths = ensure_project_directories(experiment_config, project_root=project_root)
    manifest_path = resolve_path(str(experiment_config["manifest"]), root=project_root)
    manifest_df = normalize_manifest(load_manifest(manifest_path), manifest_path)
    errors = validate_manifest(manifest_df)
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))
    named_queries = _optional_dataset(experiment_config, "named_queries", project_root)
    triplets = _optional_dataset(experiment_config, "triplets", project_root)
    results: list[dict[str, object]] = []
    targets = _expand_targets(experiment_config)
    prefer_parquet = bool(experiment_config.get("storage", {}).get("prefer_parquet", True))
    for target in targets:
        model_name = target["model"]
        profile_name = target["profile"]
        runtime_config = load_runtime_config(model_name, profile_name, project_root=project_root)
        model = load_named_model(model_name, profile_name, project_root=project_root)
        store = FeatureStore(
            resolved_paths["features_dir"],
            model_name=model.descriptor.name,
            profile_name=profile_name,
            prefer_parquet=prefer_parquet,
        )
        if not store.has_song_vectors():
            build_segment_vectors(
                manifest_df,
                model,
                store,
                ffmpeg_bin=ffmpeg_bin,
                segment_grids=list(runtime_config.get("segment_grids", [])),
                overwrite=bool(runtime_config.get("overwrite_embeddings", False)),
            )
            build_song_vectors(
                manifest_df,
                store,
                pooler=str(runtime_config.get("pooler", "mean")),
                section_length_sec=float(runtime_config.get("section_length_sec", 30.0)),
                overwrite=bool(runtime_config.get("overwrite_embeddings", False)),
            )
        rows, vectors = store.load_song_vectors()
        metrics = {}
        metrics.update(same_artist_recall(rows, vectors))
        metrics.update(album_recall(rows, vectors))
        metrics["triplet_accuracy"] = triplet_accuracy(rows, vectors, triplets)
        named_query_results = run_named_queries(rows, vectors, named_queries, store=store)
        named_query_summary = summarize_named_queries(named_query_results)
        if "match_rate" in named_query_summary:
            metrics["named_query_match_rate"] = named_query_summary["match_rate"]
        results.append(
            {
                "model": model.descriptor.name,
                "profile": profile_name,
                "target": f"{model.descriptor.name}/{profile_name}",
                "song_count": int(rows.shape[0]),
                "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0,
                "metrics": metrics,
                "named_query_summary": named_query_summary,
                "named_queries": named_query_results,
            }
        )
    return {
        "name": experiment_config.get("name", "benchmark"),
        "manifest": str(manifest_path),
        "results": results,
    }
