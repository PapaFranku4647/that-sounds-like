# ThatSoundsLike

ThatSoundsLike is a local Python project for song-level similarity search over full tracks.
It canonicalizes your audio library, embeds songs into a shared vector space, and answers
queries like:

- nearest neighbors for a song
- "most Beatles" Pink Floyd song
- "most Zeppelin" Zeppelin song
- best cross-artist song pair between two artists

The repository ships with:

- a deterministic local `stats` embedding baseline that works without PyTorch
- pluggable adapters for `MERT` and `music2vec`
- profile-scoped embedding artifacts, so you can compare retrieval recipes safely
- an artifact layout for canonical audio, segment embeddings, song embeddings, and reports
- a CLI for ingest, embedding, retrieval, benchmarking, evaluation, and report generation

## Environment

The intended runtime is WSL2 Ubuntu with Python 3.11 and a virtual environment:

If you copy this repo from Windows into WSL, exclude transient local directories like `.tmp/`
and `.venv/`. Example:

```bash
mkdir -p ~/code
rsync -a --exclude '.tmp' --exclude '.venv' \
  /mnt/c/Users/pipin/Desktop/Coding/Personal/2026/ThatSoundsLike_v2/ \
  ~/code/ThatSoundsLike_v2/
cd ~/code/ThatSoundsLike_v2
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip install -e ".[ml]"
```

For Windows-native development, `ffmpeg` must be on `PATH`. The baseline pipeline works
with the standard library, `numpy`, `pandas`, and `PyYAML`. The pretrained model adapters
lazy-load `torch` and `transformers` and only require them when those models are selected.

## Quick Start

1. Put raw songs under `music_raw/<artist>/<album>/<track>.mp3`.
2. Scan the raw library and build a manifest:

```bash
tsl ingest scan-raw
```

3. Validate a manifest manually if you need to edit or regenerate it:

```bash
tsl ingest validate --manifest data/manifests/library.csv
```

4. Canonicalize audio into `artifacts/canonical`:

```bash
tsl ingest canonicalize --manifest artifacts/indexes/library_manifest.csv
```

5. Build embeddings with the deterministic baseline:

```bash
tsl embed run --manifest artifacts/indexes/library_manifest.csv --model stats --profile baseline_v1
```

6. Query the corpus:

```bash
tsl query nearest --model stats --profile baseline_v1 --song-id my-song-id --top-k 5
tsl query centroid --model stats --profile baseline_v1 --source-artist "The Beatles" --target-artist "Pink Floyd"
tsl query artist-like --model stats --profile baseline_v1 --artist "Led Zeppelin"
tsl query pair --model stats --profile baseline_v1 --artist-a "Pink Floyd" --artist-b "Daft Punk"
```

Query output is intentionally compact:

- song rows are reduced to the fields you actually need for review
- `NaN` values are converted to `null`
- `source_rel_path` is normalized to forward slashes across Windows and WSL
- nearest and pair queries include a summarized best-segment explanation

## Manifest Format

Required columns:

- `artist`
- `title`
- `source_path`

Recommended columns:

- `song_id`
- `album`
- `track_number`
- `year`
- `duration_sec`
- `notes`

If `song_id` is omitted, `tsl ingest validate` generates a stable UUID5 based on normalized
metadata and writes a normalized manifest to `artifacts/indexes/library_manifest.csv`.

## Raw Library

- Preferred location: `music_raw/`
- Preferred layout: `music_raw/<artist>/<album>/<track>.mp3`
- `tsl ingest scan-raw` prefers embedded tags for `artist`, `album`, `title`, and `year`
- If tags are missing, it falls back to folder names and sanitized filename stems
- Raw files do not need to be renamed just because they contain spaces or punctuation
- Canonical output names are based on `song_id`, not the raw filename

## Artifact Layout

- `artifacts/canonical/`: canonical FLAC files
- `artifacts/features/<model>/segment_vectors/`: per-song segment vectors
- `artifacts/features/<model>/segment_metadata/`: per-song segment timing tables
- `artifacts/features/<model>/song_vectors.npy`: song embedding matrix
- `artifacts/features/<model>/song_rows.(parquet|csv)`: row metadata aligned to the matrix
- `artifacts/reports/`: benchmark and query reports

## Named Query Evaluation

Use `data/eval/named_queries.csv` to save the queries you actually care about and turn them
into repeatable evaluation runs.

Supported query types:

- `nearest`
- `centroid`
- `most_artist_like`
- `pair`

Useful columns:

- `name`
- `query_type`
- `song_id`
- `source_artist`
- `target_artist`
- `artist`
- `artist_a`
- `artist_b`
- `top_k`
- `expected_song_id`
- `expected_artist`
- `expected_album`
- `expected_title`
- `expected_song_id_a`
- `expected_artist_a`
- `expected_album_a`
- `expected_title_a`
- `expected_song_id_b`
- `expected_artist_b`
- `expected_album_b`
- `expected_title_b`
- `notes`

You can leave the expectation columns blank while exploring. Once you know what “good” looks
like, fill them in and use the benchmark report to track whether model changes improve or
regress those named queries.

Build a saved query report with:

```bash
tsl report build --model mert95 --query-set data/eval/named_queries.csv --output-name mert95_queries
```

This writes:

- `artifacts/reports/mert95_queries.json`
- `artifacts/reports/mert95_queries.md`
- `artifacts/reports/mert95_queries.html`

## Profiles

Profiles define the retrieval recipe independently from the encoder.

Included profiles:

- `baseline_v1`
  - one segment grid: `8s / 4s`
  - one hidden layer: `-1`
  - song pooling: `mean`
- `multiscale_v1`
  - three segment grids: `5s / 2.5s`, `15s / 7.5s`, `30s / 15s`
  - three hidden layers: `-1, -3, -5`
  - song pooling: `section_mean`

Artifacts are stored under:

- `artifacts/features/<model>/<profile>/...`

This lets you compare the same encoder under multiple retrieval recipes without overwriting
existing vectors.

## Evaluation Commands

Score one model/profile pair against your saved query and triplet sets:

```bash
tsl eval score \
  --manifest artifacts/indexes/library_manifest.csv \
  --model mert95 \
  --profile multiscale_v1 \
  --named-queries data/eval/named_queries.csv \
  --triplets data/eval/human_triplets.csv
```

Sample hard triplets for labeling from existing song vectors:

```bash
tsl eval sample-triplets --model mert95 --profile multiscale_v1 --count 100
```

Compare two saved benchmark or eval JSON reports:

```bash
tsl eval compare \
  --left artifacts/reports/eval_mert95_baseline_v1.json \
  --right artifacts/reports/eval_mert95_multiscale_v1.json
```

## Notes

- Use `stats` for smoke tests and local development.
- Use `mert95`, `music2vec`, or `mert330` after installing the `ml` extras and confirming the
  correct CUDA runtime in WSL.
- The benchmark pipeline can compare model/profile targets and produce Markdown and HTML reports.
