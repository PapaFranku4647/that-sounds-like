from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

SONG_OUTPUT_FIELDS = [
    "song_id",
    "artist",
    "album",
    "title",
    "track_number",
    "year",
    "duration_sec",
    "source_rel_path",
    "score",
]
FLOAT_PRECISION = {
    "score": 6,
    "duration_sec": 3,
    "start_sec": 3,
    "end_sec": 3,
}


def normalize_relative_path(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parts = [part for part in PurePosixPath(text.replace("\\", "/")).parts if part not in {"", "."}]
    if parts and parts[0] == "/":
        parts = parts[1:]
    return "/".join(parts)


def json_ready(value: Any, field_name: str | None = None) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for key, nested in value.items():
            converted = json_ready(nested, field_name=str(key))
            if converted not in (None, "", [], {}):
                payload[str(key)] = converted
        return payload
    if isinstance(value, (list, tuple)):
        return [item for item in (json_ready(item) for item in value) if item not in (None, "", [], {})]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item(), field_name=field_name)
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        normalized = normalize_relative_path(value) if field_name == "source_rel_path" else value
        return normalized or None
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if field_name in FLOAT_PRECISION:
            return round(value, FLOAT_PRECISION[field_name])
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def lookup_song(rows: pd.DataFrame, song_id: str) -> dict[str, Any]:
    mask = rows["song_id"].astype(str) == str(song_id)
    if not mask.any():
        raise ValueError(f"Song not found for song_id '{song_id}'")
    return rows.loc[mask].iloc[0].to_dict()


def song_brief(record: Mapping[str, Any], *, rank: int | None = None, score: float | None = None) -> dict[str, Any]:
    payload = {field: record.get(field) for field in SONG_OUTPUT_FIELDS if field in record}
    if score is not None:
        payload["score"] = score
    if not payload.get("source_rel_path") and record.get("source_path"):
        marker = "/music_raw/"
        source_path = str(record["source_path"]).replace("\\", "/")
        marker_index = source_path.casefold().find(marker)
        if marker_index != -1:
            payload["source_rel_path"] = source_path[marker_index + len(marker) :]
    if rank is not None:
        payload["rank"] = rank
    return json_ready(payload)


def song_matches(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(frame.reset_index(drop=True).to_dict(orient="records"), start=1):
        records.append(song_brief(row, rank=index))
    return records


def segment_pair_payload(explanation: Mapping[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    song_a = song_brief(lookup_song(rows, str(explanation["song_id_a"])))
    song_b = song_brief(lookup_song(rows, str(explanation["song_id_b"])))
    return json_ready(
        {
            "song_a": song_a,
            "song_b": song_b,
            "segment_a": {
                "index": explanation.get("segment_index_a"),
                "start_sec": explanation.get("start_sec_a"),
                "end_sec": explanation.get("end_sec_a"),
            },
            "segment_b": {
                "index": explanation.get("segment_index_b"),
                "start_sec": explanation.get("start_sec_b"),
                "end_sec": explanation.get("end_sec_b"),
            },
            "score": explanation.get("score"),
        }
    )


def nearest_payload(
    rows: pd.DataFrame,
    query_song_id: str,
    matches: pd.DataFrame,
    explanation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": {
            "type": "nearest",
            "song": song_brief(lookup_song(rows, query_song_id)),
        },
        "matches": song_matches(matches),
    }
    if explanation is not None:
        payload["best_segment_pair"] = segment_pair_payload(explanation, rows)
    return json_ready(payload)


def centroid_payload(
    rows: pd.DataFrame,
    source_artist: str,
    target_artist: str,
    matches: pd.DataFrame,
) -> dict[str, Any]:
    return json_ready(
        {
            "query": {
                "type": "centroid",
                "source_artist": source_artist,
                "target_artist": target_artist,
            },
            "matches": song_matches(matches),
        }
    )


def artist_like_payload(rows: pd.DataFrame, artist: str, matches: pd.DataFrame) -> dict[str, Any]:
    return json_ready(
        {
            "query": {
                "type": "artist_like",
                "artist": artist,
            },
            "matches": song_matches(matches),
        }
    )


def pair_payload(
    rows: pd.DataFrame,
    artist_a: str,
    artist_b: str,
    result: Mapping[str, Any],
    explanation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": {
            "type": "pair",
            "artist_a": artist_a,
            "artist_b": artist_b,
        },
        "match": {
            "score": result.get("score"),
            "song_a": song_brief(
                lookup_song(rows, str(result["artist_a_song_id"])),
                score=result.get("score"),
            ),
            "song_b": song_brief(lookup_song(rows, str(result["artist_b_song_id"]))),
        },
    }
    if explanation is not None:
        payload["best_segment_pair"] = segment_pair_payload(explanation, rows)
    return json_ready(payload)
