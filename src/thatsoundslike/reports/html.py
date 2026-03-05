from __future__ import annotations

from html import escape
from typing import Any


def _format_score(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return escape(str(value))


def _song_line(song: dict[str, Any]) -> str:
    artist = escape(str(song.get("artist", "")))
    title = escape(str(song.get("title", "")))
    album = escape(str(song.get("album", "")))
    score = _format_score(song.get("score"))
    line = f"{artist} - {title}" if artist or title else escape(str(song.get("song_id", "")))
    extras = [part for part in [album, f"score {score}" if score else ""] if part]
    if extras:
        line += f" [{'; '.join(extras)}]"
    return line


def _query_spec(query: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value}" for key, value in query.items() if value not in ("", None)
    )


def _evaluation(evaluation: dict[str, Any] | None) -> str:
    if evaluation is None:
        return "not evaluated"
    status = "PASS" if evaluation.get("matched") else "FAIL"
    checks = "; ".join(
        f"{escape(str(check['field']))}: expected={escape(str(check['expected']))} actual={escape(str(check['actual']))}"
        for check in evaluation.get("checks", [])
    )
    return f"{status} ({checks})" if checks else status


def _render_named_query(query: dict[str, Any]) -> str:
    parts = [
        f"<section><h4>{escape(str(query['name']))}</h4>",
        f"<p><strong>Query type:</strong> {escape(str(query['query_type']))}</p>",
        f"<p><strong>Query:</strong> {escape(_query_spec(query.get('query', {})))}</p>",
        f"<p><strong>Evaluation:</strong> {escape(_evaluation(query.get('evaluation')))}</p>",
    ]
    response = query.get("response", {})
    if query["query_type"] == "pair":
        match = response.get("match")
        if match:
            parts.append(
                "<p><strong>Match:</strong> "
                f"{_song_line(match['song_a'])} &lt;-&gt; {_song_line(match['song_b'])}</p>"
            )
    else:
        matches = response.get("matches", [])
        if matches:
            items = "".join(f"<li>{_song_line(match)}</li>" for match in matches)
            parts.append(f"<ul>{items}</ul>")
    if response.get("best_segment_pair"):
        pair = response["best_segment_pair"]
        parts.append(
            "<p><strong>Best segment pair:</strong> "
            f"{escape(str(pair['song_a']['title']))} {pair['segment_a']['start_sec']}-{pair['segment_a']['end_sec']}s "
            f"&lt;-&gt; {escape(str(pair['song_b']['title']))} {pair['segment_b']['start_sec']}-{pair['segment_b']['end_sec']}s "
            f"(score {_format_score(pair['score'])})</p>"
        )
    parts.append("</section>")
    return "\n".join(parts)


def render_benchmark_html(report: dict[str, object]) -> str:
    sections = [f"<h1>{escape(str(report['name']))}</h1>", f"<p>Manifest: {escape(str(report['manifest']))}</p>"]
    for result in report.get("results", []):
        metrics = "".join(
            f"<li>{escape(str(key))}: {float(value):.4f}</li>"
            for key, value in result.get("metrics", {}).items()
        )
        summary = result.get("named_query_summary", {})
        summary_bits = [
            f"<p>Model: {escape(str(result['model']))}</p>",
            f"<p>Profile: {escape(str(result.get('profile', '')))}</p>" if result.get("profile") else "",
            f"<p>Songs: {escape(str(result['song_count']))}</p>",
            f"<p>Embedding dim: {escape(str(result['embedding_dim']))}</p>",
        ]
        summary_bits = [value for value in summary_bits if value]
        if summary:
            summary_bits.append(f"<p>Named queries: {escape(str(summary.get('total_queries', 0)))}</p>")
            if summary.get("evaluated_queries", 0):
                summary_bits.append(
                    "<p>Named query matches: "
                    f"{escape(str(summary['passed_queries']))}/{escape(str(summary['evaluated_queries']))}</p>"
                )
        named_queries = "".join(_render_named_query(query) for query in result.get("named_queries", []))
        sections.append(
            "\n".join(
                [
                    f"<section><h2>{escape(str(result.get('target', result['model'])))}</h2>",
                    *summary_bits,
                    f"<ul>{metrics}</ul>",
                    named_queries,
                    "</section>",
                ]
            )
        )
    return "<html><body>" + "".join(sections) + "</body></html>\n"


def render_query_html(title: str, items: list[dict[str, object]]) -> str:
    sections = [f"<h1>{escape(title)}</h1>"]
    sections.extend(_render_named_query(item) for item in items)
    return "<html><body>" + "".join(sections) + "</body></html>\n"
