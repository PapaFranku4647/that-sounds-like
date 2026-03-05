from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def canonical_output_path(song_id: str, output_dir: str | Path, codec: str = "flac") -> Path:
    return Path(output_dir) / f"{song_id}.{codec}"


def canonicalize_song(
    source_path: str | Path,
    output_path: str | Path,
    sample_rate: int,
    channels: int,
    codec: str = "flac",
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        codec,
        str(target_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return target_path


def canonicalize_manifest(
    manifest_df: pd.DataFrame,
    output_dir: str | Path,
    sample_rate: int,
    channels: int,
    codec: str = "flac",
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> pd.DataFrame:
    updated = manifest_df.copy()
    if "canonical_path" not in updated.columns:
        updated["canonical_path"] = ""
    else:
        updated["canonical_path"] = updated["canonical_path"].fillna("").astype(str)
    for index, row in updated.iterrows():
        target_path = canonical_output_path(row["song_id"], output_dir, codec=codec)
        if overwrite or not target_path.exists():
            canonicalize_song(
                source_path=row["source_path"],
                output_path=target_path,
                sample_rate=sample_rate,
                channels=channels,
                codec=codec,
                overwrite=overwrite,
                ffmpeg_bin=ffmpeg_bin,
            )
        updated.at[index, "canonical_path"] = str(target_path.resolve())
    return updated
