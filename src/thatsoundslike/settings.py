from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_base_config(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_yaml(project_root / "configs" / "base.yaml")


def load_model_config(name_or_path: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config_path = Path(name_or_path)
    if not config_path.suffix:
        config_path = project_root / "configs" / "models" / f"{name_or_path}.yaml"
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    return deep_merge(load_base_config(project_root), load_yaml(config_path))


def load_profile_config(name_or_path: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config_path = Path(name_or_path)
    if not config_path.suffix:
        config_path = project_root / "configs" / "profiles" / f"{name_or_path}.yaml"
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    return load_yaml(config_path)


def load_runtime_config(
    model_name_or_path: str,
    profile_name_or_path: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    runtime = load_model_config(model_name_or_path, project_root=project_root)
    model_name = str(runtime.get("name", model_name_or_path))
    profile_name = str(profile_name_or_path or runtime.get("default_profile", "baseline_v1"))
    profile = load_profile_config(profile_name, project_root=project_root)
    merged = deep_merge(runtime, profile)
    merged["name"] = model_name
    merged["profile_name"] = str(profile.get("name", profile_name))
    return merged


def load_experiment_config(name_or_path: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config_path = Path(name_or_path)
    if not config_path.suffix:
        config_path = project_root / "configs" / "experiments" / f"{name_or_path}.yaml"
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    return deep_merge(load_base_config(project_root), load_yaml(config_path))


def ensure_project_directories(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    paths = config.get("paths", {})
    resolved: dict[str, Path] = {}
    for name, value in paths.items():
        path = resolve_path(value, project_root)
        path.mkdir(parents=True, exist_ok=True)
        resolved[name] = path
    return resolved
