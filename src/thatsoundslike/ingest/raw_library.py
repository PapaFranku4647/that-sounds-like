from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .metadata import probe_media

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}

SAFE_ASCII_PATTERN = re.compile(r"^[A-Za-z0-9 _.,()\-'\[\]&!]+$")
TRACK_PREFIX_PATTERN = re.compile(r"^(?P<track>\d{1,3})(?:\s*[-._)\]]\s*|\s+)(?P<title>.+)$")


def discover_raw_library_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "music_raw",
        project_root / "data" / "music_raw",
        project_root / "src" / "thatsoundslike" / "audio" / "music_raw",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No raw music library found. Searched: {searched}")


def iter_audio_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def clean_metadata_text(value: Any, replace_underscores: bool = False) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    if replace_underscores:
        text = text.replace("_", " ")
    return " ".join(text.strip().split())


def parse_track_number(value: Any) -> str:
    text = clean_metadata_text(value)
    if not text:
        return ""
    digits = re.match(r"^(\d{1,3})", text)
    return digits.group(1) if digits else ""


def parse_year(value: Any) -> str:
    text = clean_metadata_text(value)
    match = re.search(r"(\d{4})", text)
    return match.group(1) if match else ""


def infer_title_and_track(path: Path) -> tuple[str, str]:
    stem = clean_metadata_text(path.stem, replace_underscores=True)
    match = TRACK_PREFIX_PATTERN.match(stem)
    if not match:
        return "", stem
    return match.group("track"), clean_metadata_text(match.group("title"))


def _path_fallbacks(root: Path, audio_path: Path) -> dict[str, str]:
    relative = audio_path.relative_to(root)
    parts = list(relative.parts)
    artist = clean_metadata_text(parts[0], replace_underscores=True) if len(parts) >= 2 else ""
    album = clean_metadata_text(parts[1], replace_underscores=True) if len(parts) >= 3 else ""
    track_number, title = infer_title_and_track(audio_path)
    return {
        "artist": artist,
        "album": album,
        "title": title,
        "track_number": track_number,
    }


def metadata_sources(tags: dict[str, Any], fallbacks: dict[str, str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for field in ("artist", "album", "title", "track_number", "year"):
        tag_key = "track" if field == "track_number" else "date" if field == "year" else field
        value = clean_metadata_text(tags.get(tag_key), replace_underscores=(field == "title"))
        if field == "track_number":
            value = parse_track_number(tags.get(tag_key))
        elif field == "year":
            value = parse_year(tags.get(tag_key))
        fallback = clean_metadata_text(fallbacks.get(field, ""), replace_underscores=(field == "title"))
        sources[field] = "tag" if value else ("path" if fallback else "missing")
    return sources


def scan_raw_library(
    input_dir: str | Path,
    probe_duration: bool = True,
    ffprobe_bin: str = "ffprobe",
) -> pd.DataFrame:
    root = Path(input_dir).resolve()
    records: list[dict[str, object]] = []
    for audio_path in iter_audio_files(root):
        media = probe_media(audio_path, ffprobe_bin=ffprobe_bin)
        tags = media.get("tags", {})
        fallbacks = _path_fallbacks(root, audio_path)
        artist = clean_metadata_text(tags.get("artist")) or fallbacks["artist"]
        album = clean_metadata_text(tags.get("album")) or fallbacks["album"]
        title = clean_metadata_text(tags.get("title"), replace_underscores=True) or fallbacks["title"]
        track_number = parse_track_number(tags.get("track")) or fallbacks["track_number"]
        year = parse_year(tags.get("date"))
        source_flags = metadata_sources(tags, {**fallbacks, "year": year})
        notes: list[str] = []
        relative = audio_path.relative_to(root)
        if any(ord(ch) > 127 for ch in str(relative)):
            notes.append("non_ascii_path")
        if any(not SAFE_ASCII_PATTERN.match(part) for part in relative.parts):
            notes.append("special_chars_in_path")
        records.append(
            {
                "artist": artist,
                "album": album,
                "title": title,
                "track_number": track_number,
                "year": year,
                "source_path": str(audio_path),
                "source_rel_path": relative.as_posix(),
                "file_ext": audio_path.suffix.lower(),
                "duration_sec": media["duration_sec"] if probe_duration else None,
                "notes": ";".join(sorted(set(notes))),
                "canonical_path": "",
                "metadata_source": json.dumps(source_flags, sort_keys=True),
            }
        )
    return pd.DataFrame(records)


def audit_raw_library(input_dir: str | Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    root = Path(input_dir).resolve()
    files = iter_audio_files(root)
    non_ascii_paths: list[str] = []
    trailing_issues: list[str] = []
    normalized_rel_paths: list[str] = []
    missing_counts = Counter({"artist": 0, "album": 0, "title": 0, "track": 0, "date": 0})
    for audio_path in files:
        relative = audio_path.relative_to(root)
        rel_text = str(relative)
        if any(ord(ch) > 127 for ch in rel_text):
            non_ascii_paths.append(rel_text)
        if any(part != part.strip() or part.endswith(".") or part.endswith(" ") for part in relative.parts):
            trailing_issues.append(rel_text)
        normalized_rel_paths.append(
            "/".join(
                clean_metadata_text(part, replace_underscores=True).casefold().strip(" .")
                for part in relative.parts
            )
        )
        tags = probe_media(audio_path, ffprobe_bin=ffprobe_bin).get("tags", {})
        for key in missing_counts:
            if not clean_metadata_text(tags.get(key)):
                missing_counts[key] += 1
    duplicate_paths = sorted(
        name for name, count in Counter(normalized_rel_paths).items() if count > 1
    )
    return {
        "root": str(root),
        "file_count": len(files),
        "extensions": Counter(path.suffix.lower() for path in files),
        "non_ascii_paths": non_ascii_paths,
        "trailing_or_bad_end_paths": trailing_issues,
        "normalized_duplicate_paths": duplicate_paths,
        "missing_tag_counts": dict(missing_counts),
    }
