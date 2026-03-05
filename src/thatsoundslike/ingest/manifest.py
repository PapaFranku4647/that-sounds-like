from __future__ import annotations

import re
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from .metadata import probe_audio

REQUIRED_COLUMNS = {"artist", "title", "source_path"}
DEFAULT_COLUMNS = [
    "song_id",
    "artist",
    "album",
    "title",
    "track_number",
    "year",
    "source_path",
    "source_rel_path",
    "file_ext",
    "duration_sec",
    "notes",
    "metadata_source",
    "canonical_path",
]
TEXT_COLUMNS = {
    "song_id",
    "artist",
    "album",
    "title",
    "track_number",
    "year",
    "source_rel_path",
    "file_ext",
    "notes",
    "metadata_source",
}
PATH_COLUMNS = {"source_path", "canonical_path"}


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def normalize_relative_path(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    parts = [part for part in PurePosixPath(text.replace("\\", "/")).parts if part not in {"", "."}]
    if parts and parts[0] == "/":
        parts = parts[1:]
    return "/".join(parts)


def stable_song_id(row: pd.Series) -> str:
    source_hint = normalize_relative_path(row.get("source_rel_path")) or normalize_text(
        Path(str(row.get("source_path", ""))).name
    )
    key = "|".join(
        [
            normalize_text(row.get("artist")),
            normalize_text(row.get("album")),
            normalize_text(row.get("title")),
            normalize_text(row.get("track_number")),
            source_hint,
        ]
    ).lower()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    dataframe = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_csv}")
    return dataframe


def normalize_manifest(
    dataframe: pd.DataFrame,
    manifest_path: str | Path,
    probe_missing_duration: bool = False,
) -> pd.DataFrame:
    manifest_dir = Path(manifest_path).resolve().parent
    normalized = dataframe.copy()
    for column in DEFAULT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    for column in TEXT_COLUMNS:
        if column in normalized.columns and normalized[column].dtype == object:
            normalized[column] = normalized[column].map(normalize_text)
    for column in PATH_COLUMNS:
        if column in normalized.columns and normalized[column].dtype == object:
            normalized[column] = normalized[column].map(
                lambda value: "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
            )
    normalized["source_rel_path"] = normalized["source_rel_path"].map(normalize_relative_path)
    normalized["source_path"] = normalized.apply(
        lambda row: str(
            resolve_source_path(
                value=row.get("source_path", ""),
                manifest_dir=manifest_dir,
                source_rel_path=row.get("source_rel_path", ""),
            )
        ),
        axis=1,
    )
    if normalized["source_rel_path"].eq("").any():
        normalized["source_rel_path"] = normalized.apply(
            lambda row: normalize_relative_path(row.get("source_rel_path"))
            or normalize_relative_path(_extract_music_raw_relative(str(row.get("source_path", "")))),
            axis=1,
        )
    if "song_id" not in normalized.columns or normalized["song_id"].eq("").any():
        normalized["song_id"] = normalized.apply(stable_song_id, axis=1)
    else:
        normalized["song_id"] = normalized["song_id"].replace("", pd.NA)
        missing_mask = normalized["song_id"].isna()
        if missing_mask.any():
            normalized.loc[missing_mask, "song_id"] = normalized.loc[missing_mask].apply(
                stable_song_id, axis=1
            )
    if probe_missing_duration:
        missing_duration = normalized["duration_sec"].isna() | normalized["duration_sec"].eq("")
        for index, row in normalized.loc[missing_duration].iterrows():
            info = probe_audio(row["source_path"])
            normalized.at[index, "duration_sec"] = info["duration_sec"]
    return normalized[DEFAULT_COLUMNS]


WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")


def _is_windows_absolute_text(value: str) -> bool:
    return bool(WINDOWS_DRIVE_PATTERN.match(value))


def _is_posix_absolute_text(value: str) -> bool:
    return PurePosixPath(value).is_absolute()


def _find_local_raw_path(relative_hint: str, manifest_dir: Path) -> Path | None:
    if not relative_hint:
        return None
    relative = Path(*PurePosixPath(relative_hint.replace("\\", "/")).parts)
    for root in [manifest_dir, *manifest_dir.parents]:
        for base in ("music_raw", Path("data") / "music_raw"):
            candidate = (root / base / relative).resolve()
            if candidate.exists():
                return candidate
    return None


def _extract_music_raw_relative(path_text: str) -> str:
    normalized = path_text.replace("\\", "/")
    marker = "/music_raw/"
    index = normalized.casefold().find(marker)
    if index == -1:
        return ""
    return normalized[index + len(marker) :].strip("/")


def resolve_source_path(
    value: str | Path,
    manifest_dir: Path,
    source_rel_path: str | Path | None = None,
) -> Path:
    path_text = str(value).strip()
    path = Path(path_text) if path_text else Path()
    if path_text:
        if path.is_absolute() and path.exists():
            return path.resolve()
        if _is_windows_absolute_text(path_text) or _is_posix_absolute_text(path_text):
            inferred_relative = _extract_music_raw_relative(path_text)
            local_from_foreign = _find_local_raw_path(inferred_relative, manifest_dir)
            if local_from_foreign is not None:
                return local_from_foreign
            return Path(path_text)
    if source_rel_path:
        local_from_rel = _find_local_raw_path(str(source_rel_path), manifest_dir)
        if local_from_rel is not None:
            return local_from_rel
    return (manifest_dir / path).resolve()


def validate_manifest(dataframe: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if dataframe["song_id"].duplicated().any():
        duplicates = dataframe.loc[dataframe["song_id"].duplicated(), "song_id"].tolist()
        errors.append(f"Duplicate song_id values found: {duplicates}")
    for row in dataframe.itertuples(index=False):
        if not Path(row.source_path).exists():
            errors.append(f"Missing source_path: {row.source_path}")
        if not normalize_text(row.artist):
            errors.append(f"Missing artist for {row.source_path}")
        if not normalize_text(row.title):
            errors.append(f"Missing title for {row.source_path}")
    return errors


def save_manifest(dataframe: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path
