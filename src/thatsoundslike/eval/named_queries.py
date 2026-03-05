from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from thatsoundslike.embeddings.storage import FeatureStore
from thatsoundslike.presentation import (
    artist_like_payload,
    centroid_payload,
    nearest_payload,
    pair_payload,
)
from thatsoundslike.retrieval.centroid import centroid_query, most_artist_like_within_artist
from thatsoundslike.retrieval.explanations import best_segment_pair
from thatsoundslike.retrieval.pairwise import best_cross_artist_pair, nearest_neighbors

SINGLE_EXPECTED_FIELDS = ("song_id", "artist", "album", "title")
PAIR_SIDES = ("a", "b")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).casefold()


def _query_spec(query: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "song_id",
        "source_artist",
        "target_artist",
        "artist",
        "artist_a",
        "artist_b",
        "top_k",
        "notes",
    )
    return {key: query.get(key) for key in keys if str(query.get(key, "")).strip()}


def _single_query_evaluation(query: Mapping[str, Any], top_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if top_result is None:
        return None
    expected = {
        field: query.get(f"expected_{field}", "")
        for field in SINGLE_EXPECTED_FIELDS
        if str(query.get(f"expected_{field}", "")).strip()
    }
    if not expected:
        return None
    checks: list[dict[str, Any]] = []
    for field, expected_value in expected.items():
        actual_value = top_result.get(field)
        checks.append(
            {
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
                "matched": _normalize_text(actual_value) == _normalize_text(expected_value),
            }
        )
    return {
        "matched": all(check["matched"] for check in checks),
        "checks": checks,
    }


def _pair_query_evaluation(query: Mapping[str, Any], top_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if top_result is None:
        return None
    checks: list[dict[str, Any]] = []
    for side in PAIR_SIDES:
        expected = {
            field: query.get(f"expected_{field}_{side}", "")
            for field in SINGLE_EXPECTED_FIELDS
            if str(query.get(f"expected_{field}_{side}", "")).strip()
        }
        if not expected:
            continue
        actual_song = top_result.get(f"song_{side}", {})
        for field, expected_value in expected.items():
            actual_value = actual_song.get(field)
            checks.append(
                {
                    "field": f"{field}_{side}",
                    "expected": expected_value,
                    "actual": actual_value,
                    "matched": _normalize_text(actual_value) == _normalize_text(expected_value),
                }
            )
    if not checks:
        return None
    return {
        "matched": all(check["matched"] for check in checks),
        "checks": checks,
    }


def evaluate_named_query(query: Mapping[str, Any], top_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if query.get("query_type") == "pair":
        return _pair_query_evaluation(query, top_result)
    return _single_query_evaluation(query, top_result)


def run_named_queries(
    rows: pd.DataFrame,
    vectors,
    named_queries: pd.DataFrame,
    *,
    store: FeatureStore | None = None,
) -> list[dict[str, Any]]:
    if named_queries.empty:
        return []
    results: list[dict[str, Any]] = []
    for query in named_queries.fillna("").to_dict(orient="records"):
        query_type = str(query.get("query_type", "")).strip()
        top_k = int(query.get("top_k") or 5)
        response: dict[str, Any]
        top_result: dict[str, Any] | None = None
        if query_type == "nearest" and query.get("song_id"):
            payload = nearest_neighbors(rows, vectors, str(query["song_id"]), top_k=top_k)
            explanation = None
            if store is not None and not payload.empty:
                explanation = best_segment_pair(str(query["song_id"]), str(payload.iloc[0]["song_id"]), store)
            response = nearest_payload(rows, str(query["song_id"]), payload, explanation)
            matches = response.get("matches", [])
            top_result = matches[0] if matches else None
        elif query_type == "centroid" and query.get("source_artist") and query.get("target_artist"):
            payload = centroid_query(
                rows,
                vectors,
                source_artist=str(query["source_artist"]),
                target_artist=str(query["target_artist"]),
                top_k=top_k,
            )
            response = centroid_payload(rows, str(query["source_artist"]), str(query["target_artist"]), payload)
            matches = response.get("matches", [])
            top_result = matches[0] if matches else None
        elif query_type == "most_artist_like" and query.get("source_artist"):
            payload = most_artist_like_within_artist(rows, vectors, str(query["source_artist"]), top_k=top_k)
            response = artist_like_payload(rows, str(query["source_artist"]), payload)
            matches = response.get("matches", [])
            top_result = matches[0] if matches else None
        elif query_type == "most_artist_like" and query.get("artist"):
            payload = most_artist_like_within_artist(rows, vectors, str(query["artist"]), top_k=top_k)
            response = artist_like_payload(rows, str(query["artist"]), payload)
            matches = response.get("matches", [])
            top_result = matches[0] if matches else None
        elif query_type == "pair" and query.get("artist_a") and query.get("artist_b"):
            payload = best_cross_artist_pair(rows, vectors, str(query["artist_a"]), str(query["artist_b"]))
            explanation = None
            if store is not None:
                explanation = best_segment_pair(str(payload["artist_a_song_id"]), str(payload["artist_b_song_id"]), store)
            response = pair_payload(rows, str(query["artist_a"]), str(query["artist_b"]), payload, explanation)
            top_result = response.get("match")
        else:
            response = {"query": {"type": query_type}, "matches": []}
        results.append(
            {
                "name": str(query.get("name", "")).strip() or query_type,
                "query_type": query_type,
                "query": _query_spec(query),
                "response": response,
                "top_result": top_result,
                "evaluation": evaluate_named_query(query, top_result),
            }
        )
    return results


def summarize_named_queries(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [result for result in results if result.get("evaluation") is not None]
    passed = [result for result in evaluated if result["evaluation"].get("matched")]
    summary: dict[str, Any] = {
        "total_queries": len(results),
        "evaluated_queries": len(evaluated),
        "passed_queries": len(passed),
        "failed_queries": max(0, len(evaluated) - len(passed)),
    }
    if evaluated:
        summary["match_rate"] = len(passed) / len(evaluated)
    return summary
