from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from thatsoundslike.embeddings.build_segment_vectors import build_segment_vectors
from thatsoundslike.embeddings.build_song_vectors import build_song_vectors
from thatsoundslike.embeddings.storage import FeatureStore
from thatsoundslike.eval.benchmark import run_benchmark
from thatsoundslike.eval.datasets import load_optional_csv
from thatsoundslike.eval.named_queries import run_named_queries
from thatsoundslike.eval.workflows import compare_reports, sample_hard_triplets
from thatsoundslike.ingest.canonicalize import canonicalize_manifest
from thatsoundslike.ingest.manifest import load_manifest, normalize_manifest, save_manifest, validate_manifest
from thatsoundslike.ingest.raw_library import audit_raw_library, discover_raw_library_dir, scan_raw_library
from thatsoundslike.logging_utils import configure_logging
from thatsoundslike.models.registry import load_named_model
from thatsoundslike.presentation import (
    artist_like_payload,
    centroid_payload,
    json_ready,
    nearest_payload,
    pair_payload,
)
from thatsoundslike.reports.html import render_benchmark_html, render_query_html
from thatsoundslike.reports.markdown import render_benchmark_report, render_query_report
from thatsoundslike.retrieval.centroid import centroid_query, most_artist_like_within_artist
from thatsoundslike.retrieval.explanations import best_segment_pair
from thatsoundslike.retrieval.pairwise import best_cross_artist_pair, nearest_neighbors
from thatsoundslike.settings import (
    PROJECT_ROOT,
    ensure_project_directories,
    load_base_config,
    load_experiment_config,
    load_runtime_config,
    resolve_path,
)


def _print_json(payload: Any) -> None:
    print(json.dumps(json_ready(payload), indent=2))


def _load_and_validate_manifest(manifest_path: str | Path, probe_missing_duration: bool = False) -> DataFrame:
    raw = load_manifest(manifest_path)
    normalized = normalize_manifest(raw, manifest_path, probe_missing_duration=probe_missing_duration)
    errors = validate_manifest(normalized)
    if errors:
        joined = "\n".join(errors)
        raise ValueError(f"Manifest validation failed:\n{joined}")
    return normalized


def _default_manifest_output(paths: dict[str, Path]) -> Path:
    return paths["indexes_dir"] / "library_manifest.csv"


def handle_ingest_validate(args: argparse.Namespace) -> int:
    config = load_base_config(PROJECT_ROOT)
    paths = ensure_project_directories(config, project_root=PROJECT_ROOT)
    raw = load_manifest(args.manifest)
    normalized = normalize_manifest(raw, args.manifest, probe_missing_duration=args.probe_duration)
    errors = validate_manifest(normalized)
    output_path = resolve_path(args.output, root=PROJECT_ROOT) if args.output else _default_manifest_output(paths)
    save_manifest(normalized, output_path)
    print(f"Wrote normalized manifest to {output_path}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Validated {len(normalized)} songs")
    return 0


def handle_ingest_scan_raw(args: argparse.Namespace) -> int:
    config = load_base_config(PROJECT_ROOT)
    paths = ensure_project_directories(config, project_root=PROJECT_ROOT)
    input_dir = (
        resolve_path(args.input_dir, root=PROJECT_ROOT)
        if args.input_dir
        else discover_raw_library_dir(PROJECT_ROOT)
    )
    scanned = scan_raw_library(input_dir=input_dir, probe_duration=not args.skip_duration)
    manifest_path = resolve_path(args.output, root=PROJECT_ROOT) if args.output else _default_manifest_output(paths)
    normalized = normalize_manifest(scanned, manifest_path, probe_missing_duration=False)
    errors = validate_manifest(normalized)
    save_manifest(normalized, manifest_path)
    audit = audit_raw_library(input_dir)
    audit_path = (
        resolve_path(args.audit_output, root=PROJECT_ROOT)
        if args.audit_output
        else paths["reports_dir"] / "raw_library_audit.json"
    )
    Path(audit_path).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Scanned {len(normalized)} audio files from {input_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote raw library audit to {audit_path}")
    if errors:
        print("\n".join(errors))
        return 1
    return 0


def handle_ingest_audit_raw(args: argparse.Namespace) -> int:
    config = load_base_config(PROJECT_ROOT)
    paths = ensure_project_directories(config, project_root=PROJECT_ROOT)
    input_dir = (
        resolve_path(args.input_dir, root=PROJECT_ROOT)
        if args.input_dir
        else discover_raw_library_dir(PROJECT_ROOT)
    )
    audit = audit_raw_library(input_dir)
    output_path = (
        resolve_path(args.output, root=PROJECT_ROOT)
        if args.output
        else paths["reports_dir"] / "raw_library_audit.json"
    )
    Path(output_path).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Wrote raw library audit to {output_path}")
    return 0


def handle_ingest_canonicalize(args: argparse.Namespace) -> int:
    config = load_base_config(PROJECT_ROOT)
    paths = ensure_project_directories(config, project_root=PROJECT_ROOT)
    manifest_df = _load_and_validate_manifest(args.manifest)
    audio_config = config["audio"]
    canonicalized = canonicalize_manifest(
        manifest_df=manifest_df,
        output_dir=paths["canonical_dir"],
        sample_rate=int(audio_config["canonical_sample_rate"]),
        channels=int(audio_config["canonical_channels"]),
        codec=str(audio_config["canonical_codec"]),
        overwrite=args.overwrite,
    )
    output_path = resolve_path(args.output, root=PROJECT_ROOT) if args.output else _default_manifest_output(paths)
    save_manifest(canonicalized, output_path)
    print(f"Canonicalized {len(canonicalized)} songs")
    print(f"Updated manifest written to {output_path}")
    return 0


def _load_runtime(model_name: str, profile_name: str | None = None) -> dict[str, Any]:
    return load_runtime_config(model_name, profile_name, project_root=PROJECT_ROOT)


def handle_embed_run(args: argparse.Namespace) -> int:
    config = _load_runtime(args.model, args.profile)
    paths = ensure_project_directories(config, project_root=PROJECT_ROOT)
    manifest_df = _load_and_validate_manifest(args.manifest)
    profile_name = str(config.get("profile_name", args.profile or "baseline_v1"))
    model = load_named_model(args.model, profile_name, project_root=PROJECT_ROOT)
    store = FeatureStore(
        paths["features_dir"],
        model_name=model.descriptor.name,
        profile_name=profile_name,
        prefer_parquet=bool(config.get("storage", {}).get("prefer_parquet", True)),
    )
    overwrite = bool(args.overwrite or config.get("overwrite_embeddings", False))
    segment_summary = build_segment_vectors(
        manifest_df,
        model,
        store,
        segment_grids=list(config.get("segment_grids", [])),
        overwrite=overwrite,
    )
    song_summary = build_song_vectors(
        manifest_df,
        store,
        pooler=str(config.get("pooler", "mean")),
        section_length_sec=float(config.get("section_length_sec", 30.0)),
        overwrite=overwrite,
    )
    _print_json(
        {
            "model": model.descriptor.name,
            "profile": profile_name,
            "segment_summary": segment_summary,
            "song_summary": song_summary,
        }
    )
    return 0


def _load_song_artifacts(model_name: str, profile_name: str | None = None) -> tuple[FeatureStore, DataFrame, Any]:
    config = _load_runtime(model_name, profile_name)
    paths = ensure_project_directories(config, project_root=PROJECT_ROOT)
    resolved_profile = str(config.get("profile_name", profile_name or "baseline_v1"))
    store = FeatureStore(
        paths["features_dir"],
        model_name=model_name,
        profile_name=resolved_profile,
        prefer_parquet=bool(config.get("storage", {}).get("prefer_parquet", True)),
    )
    rows, vectors = store.load_song_vectors()
    return store, rows, vectors


def handle_query_nearest(args: argparse.Namespace) -> int:
    store, rows, vectors = _load_song_artifacts(args.model, args.profile)
    matches = nearest_neighbors(rows, vectors, args.song_id, top_k=args.top_k)
    if not matches.empty:
        explanation = best_segment_pair(args.song_id, str(matches.iloc[0]["song_id"]), store)
        _print_json(nearest_payload(rows, args.song_id, matches, explanation))
    else:
        _print_json(nearest_payload(rows, args.song_id, matches))
    return 0


def handle_query_centroid(args: argparse.Namespace) -> int:
    _, rows, vectors = _load_song_artifacts(args.model, args.profile)
    matches = centroid_query(
        rows,
        vectors,
        source_artist=args.source_artist,
        target_artist=args.target_artist,
        top_k=args.top_k,
    )
    _print_json(centroid_payload(rows, args.source_artist, args.target_artist, matches))
    return 0


def handle_query_pair(args: argparse.Namespace) -> int:
    store, rows, vectors = _load_song_artifacts(args.model, args.profile)
    result = best_cross_artist_pair(rows, vectors, args.artist_a, args.artist_b)
    explanation = best_segment_pair(str(result["artist_a_song_id"]), str(result["artist_b_song_id"]), store)
    _print_json(pair_payload(rows, args.artist_a, args.artist_b, result, explanation))
    return 0


def handle_query_artist_like(args: argparse.Namespace) -> int:
    _, rows, vectors = _load_song_artifacts(args.model, args.profile)
    result = most_artist_like_within_artist(rows, vectors, args.artist, top_k=args.top_k)
    _print_json(artist_like_payload(rows, args.artist, result))
    return 0


def handle_benchmark_run(args: argparse.Namespace) -> int:
    experiment = load_experiment_config(args.experiment, project_root=PROJECT_ROOT)
    paths = ensure_project_directories(experiment, project_root=PROJECT_ROOT)
    report = run_benchmark(experiment, project_root=PROJECT_ROOT)
    output_name = str(experiment.get("output_name", experiment.get("name", "benchmark")))
    json_path = paths["reports_dir"] / f"{output_name}.json"
    markdown_path = paths["reports_dir"] / f"{output_name}.md"
    html_path = paths["reports_dir"] / f"{output_name}.html"
    json_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    markdown_path.write_text(render_benchmark_report(report), encoding="utf-8")
    html_path.write_text(render_benchmark_html(report), encoding="utf-8")
    print(f"Wrote benchmark report to {json_path}")
    return 0


def handle_eval_score(args: argparse.Namespace) -> int:
    base_config = load_base_config(PROJECT_ROOT)
    paths = ensure_project_directories(base_config, project_root=PROJECT_ROOT)
    profile_name = args.profile or "baseline_v1"
    experiment = {
        **base_config,
        "name": args.output_name or f"eval_{args.model}_{profile_name}",
        "manifest": args.manifest,
        "targets": [{"model": args.model, "profile": profile_name}],
        "named_queries": args.named_queries or "",
        "triplets": args.triplets or "",
        "output_name": args.output_name or f"eval_{args.model}_{profile_name}",
    }
    report = run_benchmark(experiment, project_root=PROJECT_ROOT)
    output_name = str(experiment["output_name"])
    json_path = paths["reports_dir"] / f"{output_name}.json"
    markdown_path = paths["reports_dir"] / f"{output_name}.md"
    html_path = paths["reports_dir"] / f"{output_name}.html"
    json_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    markdown_path.write_text(render_benchmark_report(report), encoding="utf-8")
    html_path.write_text(render_benchmark_html(report), encoding="utf-8")
    print(f"Wrote eval report to {json_path}")
    return 0


def handle_eval_sample_triplets(args: argparse.Namespace) -> int:
    _, rows, vectors = _load_song_artifacts(args.model, args.profile)
    samples = sample_hard_triplets(rows, vectors, count=args.count)
    base_config = load_base_config(PROJECT_ROOT)
    paths = ensure_project_directories(base_config, project_root=PROJECT_ROOT)
    output_path = (
        resolve_path(args.output, root=PROJECT_ROOT)
        if args.output
        else paths["reports_dir"] / f"triplets_{args.model}_{args.profile or 'baseline_v1'}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_path, index=False)
    print(f"Wrote {len(samples)} triplets to {output_path}")
    return 0


def handle_eval_compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(resolve_path(args.left, root=PROJECT_ROOT)).read_text(encoding="utf-8"))
    right = json.loads(Path(resolve_path(args.right, root=PROJECT_ROOT)).read_text(encoding="utf-8"))
    comparison = compare_reports(left, right)
    output_path = resolve_path(args.output, root=PROJECT_ROOT) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(json_ready(comparison), indent=2), encoding="utf-8")
        print(f"Wrote comparison report to {output_path}")
    else:
        _print_json(comparison)
    return 0


def handle_report_build(args: argparse.Namespace) -> int:
    store, rows, vectors = _load_song_artifacts(args.model, args.profile)
    query_set = load_optional_csv(args.query_set)
    results = run_named_queries(rows, vectors, query_set, store=store)
    base_config = load_base_config(PROJECT_ROOT)
    paths = ensure_project_directories(base_config, project_root=PROJECT_ROOT)
    output_stem = paths["reports_dir"] / (args.output_name or "query_report")
    report_payload = {
        "title": "ThatSoundsLike Query Report",
        "model": args.model,
        "profile": args.profile or "baseline_v1",
        "query_set": str(args.query_set),
        "results": results,
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(json_ready(report_payload), indent=2),
        encoding="utf-8",
    )
    output_stem.with_suffix(".md").write_text(
        render_query_report("ThatSoundsLike Query Report", results),
        encoding="utf-8",
    )
    output_stem.with_suffix(".html").write_text(
        render_query_html("ThatSoundsLike Query Report", results),
        encoding="utf-8",
    )
    print(f"Wrote query report to {output_stem.with_suffix('.md')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsl")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest_subparsers = ingest.add_subparsers(dest="ingest_command", required=True)

    ingest_validate = ingest_subparsers.add_parser("validate")
    ingest_validate.add_argument("--manifest", required=True)
    ingest_validate.add_argument("--output")
    ingest_validate.add_argument("--probe-duration", action="store_true")
    ingest_validate.set_defaults(func=handle_ingest_validate)

    ingest_scan_raw = ingest_subparsers.add_parser("scan-raw")
    ingest_scan_raw.add_argument("--input-dir")
    ingest_scan_raw.add_argument("--output")
    ingest_scan_raw.add_argument("--audit-output")
    ingest_scan_raw.add_argument("--skip-duration", action="store_true")
    ingest_scan_raw.set_defaults(func=handle_ingest_scan_raw)

    ingest_audit_raw = ingest_subparsers.add_parser("audit-raw")
    ingest_audit_raw.add_argument("--input-dir")
    ingest_audit_raw.add_argument("--output")
    ingest_audit_raw.set_defaults(func=handle_ingest_audit_raw)

    ingest_canonicalize = ingest_subparsers.add_parser("canonicalize")
    ingest_canonicalize.add_argument("--manifest", required=True)
    ingest_canonicalize.add_argument("--output")
    ingest_canonicalize.add_argument("--overwrite", action="store_true")
    ingest_canonicalize.set_defaults(func=handle_ingest_canonicalize)

    embed = subparsers.add_parser("embed")
    embed_subparsers = embed.add_subparsers(dest="embed_command", required=True)
    embed_run = embed_subparsers.add_parser("run")
    embed_run.add_argument("--manifest", required=True)
    embed_run.add_argument("--model", required=True)
    embed_run.add_argument("--profile", default="baseline_v1")
    embed_run.add_argument("--overwrite", action="store_true")
    embed_run.set_defaults(func=handle_embed_run)

    query = subparsers.add_parser("query")
    query_subparsers = query.add_subparsers(dest="query_command", required=True)

    query_nearest = query_subparsers.add_parser("nearest")
    query_nearest.add_argument("--model", required=True)
    query_nearest.add_argument("--profile", default="baseline_v1")
    query_nearest.add_argument("--song-id", required=True)
    query_nearest.add_argument("--top-k", type=int, default=10)
    query_nearest.set_defaults(func=handle_query_nearest)

    query_centroid = query_subparsers.add_parser("centroid")
    query_centroid.add_argument("--model", required=True)
    query_centroid.add_argument("--profile", default="baseline_v1")
    query_centroid.add_argument("--source-artist", required=True)
    query_centroid.add_argument("--target-artist", required=True)
    query_centroid.add_argument("--top-k", type=int, default=10)
    query_centroid.set_defaults(func=handle_query_centroid)

    query_pair = query_subparsers.add_parser("pair")
    query_pair.add_argument("--model", required=True)
    query_pair.add_argument("--profile", default="baseline_v1")
    query_pair.add_argument("--artist-a", required=True)
    query_pair.add_argument("--artist-b", required=True)
    query_pair.set_defaults(func=handle_query_pair)

    query_artist_like = query_subparsers.add_parser("artist-like")
    query_artist_like.add_argument("--model", required=True)
    query_artist_like.add_argument("--profile", default="baseline_v1")
    query_artist_like.add_argument("--artist", required=True)
    query_artist_like.add_argument("--top-k", type=int, default=1)
    query_artist_like.set_defaults(func=handle_query_artist_like)

    benchmark = subparsers.add_parser("benchmark")
    benchmark_subparsers = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_run = benchmark_subparsers.add_parser("run")
    benchmark_run.add_argument("--experiment", required=True)
    benchmark_run.set_defaults(func=handle_benchmark_run)

    eval_parser = subparsers.add_parser("eval")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_score = eval_subparsers.add_parser("score")
    eval_score.add_argument("--manifest", required=True)
    eval_score.add_argument("--model", required=True)
    eval_score.add_argument("--profile", default="baseline_v1")
    eval_score.add_argument("--named-queries")
    eval_score.add_argument("--triplets")
    eval_score.add_argument("--output-name")
    eval_score.set_defaults(func=handle_eval_score)

    eval_sample_triplets = eval_subparsers.add_parser("sample-triplets")
    eval_sample_triplets.add_argument("--model", required=True)
    eval_sample_triplets.add_argument("--profile", default="baseline_v1")
    eval_sample_triplets.add_argument("--count", type=int, default=100)
    eval_sample_triplets.add_argument("--output")
    eval_sample_triplets.set_defaults(func=handle_eval_sample_triplets)

    eval_compare = eval_subparsers.add_parser("compare")
    eval_compare.add_argument("--left", required=True)
    eval_compare.add_argument("--right", required=True)
    eval_compare.add_argument("--output")
    eval_compare.set_defaults(func=handle_eval_compare)

    report = subparsers.add_parser("report")
    report_subparsers = report.add_subparsers(dest="report_command", required=True)
    report_build = report_subparsers.add_parser("build")
    report_build.add_argument("--model", required=True)
    report_build.add_argument("--profile", default="baseline_v1")
    report_build.add_argument("--query-set", required=True)
    report_build.add_argument("--output-name")
    report_build.set_defaults(func=handle_report_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return int(args.func(args))
