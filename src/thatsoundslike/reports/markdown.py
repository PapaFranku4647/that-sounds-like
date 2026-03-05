from __future__ import annotations

from typing import Any


def _format_score(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _song_line(song: dict[str, Any]) -> str:
    artist = song.get("artist", "")
    title = song.get("title", "")
    album = song.get("album", "")
    score = _format_score(song.get("score"))
    line = f"{artist} - {title}" if artist or title else str(song.get("song_id", ""))
    extras = [part for part in [album, f"score {score}" if score else ""] if part]
    if extras:
        line += f" [{'; '.join(extras)}]"
    return line


def _query_spec_line(query: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in query.items() if key != "notes"]
    if query.get("notes"):
        parts.append(f"notes={query['notes']}")
    return ", ".join(parts)


def _evaluation_line(evaluation: dict[str, Any] | None) -> str:
    if evaluation is None:
        return "not evaluated"
    status = "PASS" if evaluation.get("matched") else "FAIL"
    checks = ", ".join(
        f"{check['field']}: expected={check['expected']} actual={check['actual']}"
        for check in evaluation.get("checks", [])
    )
    return f"{status} ({checks})" if checks else status


def _render_named_query(lines: list[str], query: dict[str, Any]) -> None:
    lines.append(f"#### {query['name']}")
    lines.append("")
    lines.append(f"- Query type: {query['query_type']}")
    lines.append(f"- Query: {_query_spec_line(query.get('query', {}))}")
    lines.append(f"- Evaluation: {_evaluation_line(query.get('evaluation'))}")
    response = query.get("response", {})
    if query["query_type"] == "pair":
        match = response.get("match")
        if match:
            lines.append(f"- Match: {_song_line(match['song_a'])} <-> {_song_line(match['song_b'])}")
    else:
        matches = response.get("matches", [])
        if matches:
            lines.append(f"- Top result: {_song_line(matches[0])}")
            if len(matches) > 1:
                lines.append("- Additional results:")
                for match in matches[1:]:
                    lines.append(f"  - {_song_line(match)}")
    if response.get("best_segment_pair"):
        pair = response["best_segment_pair"]
        lines.append(
            "- Best segment pair: "
            f"{pair['song_a']['title']} {pair['segment_a']['start_sec']}-{pair['segment_a']['end_sec']}s"
            f" <-> {pair['song_b']['title']} {pair['segment_b']['start_sec']}-{pair['segment_b']['end_sec']}s"
            f" (score {_format_score(pair['score'])})"
        )
    lines.append("")


def render_benchmark_report(report: dict[str, object]) -> str:
    lines = [f"# {report['name']}", "", f"Manifest: `{report['manifest']}`", ""]
    for result in report.get("results", []):
        target = result.get("target", result["model"])
        lines.append(f"## Target: {target}")
        lines.append("")
        lines.append(f"- Model: {result['model']}")
        if result.get("profile"):
            lines.append(f"- Profile: {result['profile']}")
        lines.append(f"- Songs: {result['song_count']}")
        lines.append(f"- Embedding dim: {result['embedding_dim']}")
        for key, value in result.get("metrics", {}).items():
            lines.append(f"- {key}: {value:.4f}")
        summary = result.get("named_query_summary", {})
        if summary:
            lines.append(f"- Named queries: {summary.get('total_queries', 0)} total")
            if summary.get("evaluated_queries", 0):
                lines.append(
                    f"- Named query matches: {summary['passed_queries']}/{summary['evaluated_queries']}"
                )
        named_queries = result.get("named_queries", [])
        if named_queries:
            lines.append("")
            lines.append("### Named Queries")
            lines.append("")
            for query in named_queries:
                _render_named_query(lines, query)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_query_report(title: str, items: list[dict[str, object]]) -> str:
    lines = [f"# {title}", ""]
    for item in items:
        _render_named_query(lines, item)
    return "\n".join(lines).strip() + "\n"
