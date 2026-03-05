from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_table(dataframe: pd.DataFrame, path_stem: str | Path, prefer_parquet: bool = True) -> Path:
    stem = Path(path_stem)
    if prefer_parquet:
        parquet_path = stem.with_suffix(".parquet")
        try:
            dataframe.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception:
            pass
    csv_path = stem.with_suffix(".csv")
    dataframe.to_csv(csv_path, index=False)
    return csv_path


def read_table(path_stem: str | Path) -> pd.DataFrame:
    stem = Path(path_stem)
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No table found for stem: {stem}")


class FeatureStore:
    def __init__(
        self,
        features_root: str | Path,
        model_name: str,
        profile_name: str = "baseline_v1",
        prefer_parquet: bool = True,
    ) -> None:
        self.model_name = model_name
        self.profile_name = profile_name
        self.prefer_parquet = prefer_parquet
        self.features_root = Path(features_root)
        self.model_dir = self.features_root / model_name / profile_name
        self.legacy_model_dir = self.features_root / model_name
        self.segment_vector_dir = self.model_dir / "segment_vectors"
        self.segment_metadata_dir = self.model_dir / "segment_metadata"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.segment_vector_dir.mkdir(parents=True, exist_ok=True)
        self.segment_metadata_dir.mkdir(parents=True, exist_ok=True)

    @property
    def legacy_song_vectors_path(self) -> Path:
        return self.legacy_model_dir / "song_vectors.npy"

    @property
    def legacy_song_rows_stem(self) -> Path:
        return self.legacy_model_dir / "song_rows"

    @property
    def legacy_segment_vector_dir(self) -> Path:
        return self.legacy_model_dir / "segment_vectors"

    @property
    def legacy_segment_metadata_dir(self) -> Path:
        return self.legacy_model_dir / "segment_metadata"

    def _use_legacy_layout(self) -> bool:
        return self.profile_name == "baseline_v1"

    @property
    def song_vectors_path(self) -> Path:
        return self.model_dir / "song_vectors.npy"

    @property
    def song_rows_stem(self) -> Path:
        return self.model_dir / "song_rows"

    def save_song_vectors(self, rows: pd.DataFrame, vectors: np.ndarray) -> None:
        np.save(self.song_vectors_path, vectors.astype(np.float32))
        write_table(rows, self.song_rows_stem, prefer_parquet=self.prefer_parquet)

    def load_song_vectors(self) -> tuple[pd.DataFrame, np.ndarray]:
        song_vectors_path = self.song_vectors_path
        song_rows_stem = self.song_rows_stem
        if self._use_legacy_layout() and not song_vectors_path.exists() and self.legacy_song_vectors_path.exists():
            song_vectors_path = self.legacy_song_vectors_path
            song_rows_stem = self.legacy_song_rows_stem
        rows = read_table(song_rows_stem)
        vectors = np.load(song_vectors_path)
        return rows, vectors.astype(np.float32)

    def has_song_vectors(self) -> bool:
        current = self.song_vectors_path.exists() and (
            self.song_rows_stem.with_suffix(".parquet").exists()
            or self.song_rows_stem.with_suffix(".csv").exists()
        )
        legacy = self._use_legacy_layout() and self.legacy_song_vectors_path.exists() and (
            self.legacy_song_rows_stem.with_suffix(".parquet").exists()
            or self.legacy_song_rows_stem.with_suffix(".csv").exists()
        )
        return bool(current or legacy)

    def save_segment_vectors(self, song_id: str, vectors: np.ndarray, metadata: pd.DataFrame) -> None:
        np.save(self.segment_vector_dir / f"{song_id}.npy", vectors.astype(np.float32))
        write_table(metadata, self.segment_metadata_dir / song_id, prefer_parquet=self.prefer_parquet)

    def load_segment_vectors(self, song_id: str) -> tuple[np.ndarray, pd.DataFrame]:
        vector_path = self.segment_vector_dir / f"{song_id}.npy"
        metadata_stem = self.segment_metadata_dir / song_id
        if self._use_legacy_layout() and not vector_path.exists():
            legacy_path = self.legacy_segment_vector_dir / f"{song_id}.npy"
            if legacy_path.exists():
                vector_path = legacy_path
                metadata_stem = self.legacy_segment_metadata_dir / song_id
        vectors = np.load(vector_path).astype(np.float32)
        metadata = read_table(metadata_stem)
        return vectors, metadata

    def has_segment_vectors(self, song_id: str) -> bool:
        current = (self.segment_vector_dir / f"{song_id}.npy").exists()
        legacy = self._use_legacy_layout() and (self.legacy_segment_vector_dir / f"{song_id}.npy").exists()
        return bool(current or legacy)
