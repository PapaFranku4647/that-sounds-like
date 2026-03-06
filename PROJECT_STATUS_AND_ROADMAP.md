# ThatSoundsLike: Status, Implementation Plan, and Roadmap

Last updated: 2026-03-05

## Purpose

This document is the operational source of truth for the project.

It exists to keep:

- the codebase direction stable
- Codex sessions consistent across machines
- implementation work aligned with the actual current state
- future experimentation structured instead of ad hoc

Use this file as the first reference when resuming work on the project.

## Project Goal

ThatSoundsLike is a local music-similarity system for full songs.

It should answer queries such as:

- nearest-neighbor song search
- "most Beatles" Pink Floyd song
- "most Zeppelin" Zeppelin song
- "Beatles song most similar to Zeppelin sound"
- "which Pink Floyd song is most similar to which Daft Punk song"

The core product objective is not generic genre classification. It is high-quality similarity retrieval over a personal library, with enough structure and explainability that the results are inspectable and improvable.

## Product Scope

### In scope

- local-first operation
- private music library ingestion
- canonical audio preprocessing
- song and segment embeddings
- song-to-song retrieval
- artist-centroid retrieval
- cross-artist pair retrieval
- benchmarking and evaluation
- profile-based experiments
- reproducible artifacts and reports

### Out of scope for v1/v2

- streaming-service integration
- distributed training infrastructure
- large-scale public deployment
- commercial licensing workflows
- generative music features

## High-Level Architecture

The system has five main layers:

1. Ingest
   - scan `music_raw/`
   - infer metadata from tags or paths
   - produce a normalized manifest

2. Canonicalization
   - convert raw files to a stable internal audio format
   - write canonical files into `artifacts/canonical/`

3. Embedding
   - segment songs into windows
   - embed segments using a selected model
   - pool segment embeddings into song embeddings
   - store profile-scoped artifacts

4. Retrieval
   - nearest-neighbor search
   - artist-centroid scoring
   - within-artist "most artist-like" search
   - cross-artist pair search
   - segment-level explanation payloads

5. Evaluation and Reporting
   - named-query scoring
   - benchmark reports
   - hard-triplet sampling
   - report comparison between model/profile runs

## Current Repository Layout

Key directories:

- `src/thatsoundslike/`
  - application code
- `configs/models/`
  - model descriptors and defaults
- `configs/profiles/`
  - retrieval/embedding experiment profiles
- `configs/experiments/`
  - benchmark target sets
- `data/eval/`
  - named queries and future human labels
- `artifacts/`
  - generated manifests, canonical audio, embeddings, reports
- `tests/`
  - unit and integration tests

Key project entrypoints:

- `python -m thatsoundslike ingest scan-raw`
- `python -m thatsoundslike ingest canonicalize`
- `python -m thatsoundslike embed run`
- `python -m thatsoundslike query nearest`
- `python -m thatsoundslike query centroid`
- `python -m thatsoundslike query artist-like`
- `python -m thatsoundslike query pair`
- `python -m thatsoundslike benchmark run`
- `python -m thatsoundslike report build`
- `python -m thatsoundslike eval score`
- `python -m thatsoundslike eval sample-triplets`
- `python -m thatsoundslike eval compare`

## Current Technical Direction

### Chosen baseline model strategy

The project uses pretrained music encoders rather than training from scratch.

Current primary model:

- `mert95` (`MERT-v1-95M`)

Supporting baselines:

- `stats`
- `music2vec`

Rationale:

- the current library size is enough for retrieval and evaluation, but not enough for a strong from-scratch encoder
- pretrained music encoders already capture useful timbral, structural, and stylistic information
- the practical next gain is better retrieval logic and better evaluation, not base-model retraining

### Current profile strategy

Two retrieval/embedding profiles are now part of the codebase:

1. `baseline_v1`
   - single segment grid
   - simple mean pooling or baseline-compatible pooling
   - compatible with legacy artifact layout

2. `multiscale_v1`
   - multiple segment grids
   - layer fusion for supported encoders
   - section-aware song pooling
   - intended to improve style similarity quality over the basic baseline

## What Has Already Been Implemented

### Core pipeline

Implemented:

- raw library scanning from `music_raw/`
- manifest normalization and validation
- canonical audio generation
- deterministic baseline embeddings
- MERT and music2vec model adapters
- segment-level and song-level embedding storage
- exact cosine retrieval
- compact query output formatting
- segment explanation payloads

### Artifact and experiment management

Implemented:

- profile-scoped artifact layout under `artifacts/features/<model>/<profile>/`
- backward-compatible reads for legacy `baseline_v1` artifacts
- resumable embedding generation
- profile-aware runtime configuration

### Evaluation and reporting

Implemented:

- named-query execution and structured report output
- benchmark experiment config support
- report comparison helpers
- hard-triplet sampling workflow
- markdown, HTML, and JSON reports

### Migration and environment support

Implemented or established:

- git-backed repo for cross-machine use
- WSL2 Ubuntu workflow on Windows machines
- Python `venv` setup
- desktop migration path for the 4080 machine
- raw library transfer workflow

## Current Confirmed State

### Source/laptop machine

Confirmed:

- the codebase exists in `ThatSoundsLike_v2`
- git repo exists and has been pushed
- full personal library was available locally
- earlier full-corpus `mert95` baseline runs on the smaller GPU produced plausible retrieval output

### Desktop/4080 machine

Confirmed:

- repo cloned into `~/code/that-sounds-like`
- virtual environment created
- project dependencies installed
- tests were run
- full `music_raw/` transfer is now complete
- authoritative library count on the desktop is `559` songs

### Important operational detail

The desktop should run:

- `git pull`
- `python -m unittest discover -s tests -v`

before long jobs, to ensure the latest workflow fixes are present.

## Current Quality Assessment

Based on earlier full-library `mert95` baseline queries, the system is already producing plausible results.

Examples that looked strong or at least defensible:

- "most Zeppelin Zeppelin song" returning `Dazed and Confused`
- Beatles-to-Pink-Floyd centroid results spreading across multiple Floyd albums instead of collapsing to one album
- Pink Floyd to Daft Punk pairings that read like texture/production matches rather than random noise

This means the system has passed the "pipeline works and retrieval is nontrivial" threshold.

It has not yet passed the "evaluated and trusted" threshold.

## Current Weaknesses

The project is working, but the quality ceiling is still limited by several things:

1. Evaluation is still underdeveloped.
   - `named_queries.csv` is only a seed set
   - `human_triplets.csv` is not populated enough to make model decisions rigorous

2. Retrieval is still mostly single-stage.
   - there is no learned reranker yet
   - duplicate/remaster/live-version suppression is not implemented yet

3. Pooling can still improve.
   - `multiscale_v1` is an upgrade, but it is not yet a true hierarchical song encoder

4. Library hygiene matters.
   - duplicates, remasters, alternate masters, and near-identical tracks can distort nearest-neighbor quality

5. Explainability is not yet complete.
   - segment timestamps are available
   - automatic clip extraction and review tooling are not yet implemented

6. Long-song handling can still improve.
   - section-aware pooling helps, but the system is not yet fully form-aware

## Immediate Next Steps

These are the current next actions on the desktop 4080 machine.

### Step 1: sync repo and verify tests

Run:

```bash
cd ~/code/that-sounds-like
git pull
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Expected result:

- tests pass cleanly

### Step 2: rebuild the raw-library manifest on the desktop

Run:

```bash
python -m thatsoundslike ingest scan-raw
```

Expected result:

- `Scanned 559 audio files`

This step is important because manifests are environment-specific and should be regenerated where the files actually live.

### Step 3: verify GPU

Run:

```bash
nvidia-smi
python -c 'import torch; print("cuda:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")'
```

Expected result:

- CUDA available
- correct GPU name shown

### Step 4: canonicalize the full library

Run:

```bash
python -m thatsoundslike ingest canonicalize --manifest artifacts/indexes/library_manifest.csv
```

Expected result:

- canonical FLACs generated under `artifacts/canonical/`

### Step 5: run the main embedding job on the 4080

Run:

```bash
python -m thatsoundslike embed run --manifest artifacts/indexes/library_manifest.csv --model mert95 --profile multiscale_v1
```

Expected result:

- full profile-scoped segment and song embeddings for `mert95/multiscale_v1`

### Step 6: inspect retrieval quality

Run:

```bash
python -m thatsoundslike query artist-like --model mert95 --profile multiscale_v1 --artist "Led Zeppelin"
python -m thatsoundslike query centroid --model mert95 --profile multiscale_v1 --source-artist "The Beatles" --target-artist "Pink Floyd"
python -m thatsoundslike query pair --model mert95 --profile multiscale_v1 --artist-a "Pink Floyd" --artist-b "Daft Punk"
```

### Step 7: run evaluation and reports

Run:

```bash
python -m thatsoundslike eval score --manifest artifacts/indexes/library_manifest.csv --model mert95 --profile multiscale_v1 --named-queries data/eval/named_queries.csv --triplets data/eval/human_triplets.csv --output-name eval_mert95_multiscale_v1
python -m thatsoundslike report build --model mert95 --profile multiscale_v1 --query-set data/eval/named_queries.csv --output-name mert95_multiscale_queries
```

### Step 8: sample labels for human review

Run:

```bash
python -m thatsoundslike eval sample-triplets --model mert95 --profile multiscale_v1 --count 100
```

## Expected Runtime on the Desktop

From the current state:

- tests + scan: about `5-10 min`
- canonicalization: about `10-25 min`
- full `mert95 multiscale_v1`: about `35-90 min`

Total expected time:

- about `50 min to 2 hr`

## Progress Check During Embedding

Use a second terminal:

```bash
cd ~/code/that-sounds-like
source .venv/bin/activate
python -c 'import pandas as pd; from pathlib import Path; df=pd.read_csv("artifacts/indexes/library_manifest.csv"); done={p.stem for p in Path("artifacts/features/mert95/multiscale_v1/segment_vectors").glob("*.npy")}; secs=df["duration_sec"].fillna(0).astype(float); mask=df["song_id"].astype(str).isin(done); progress=secs[mask].sum()/secs.sum() if secs.sum() else 0.0; print(f"{mask.sum()}/{len(df)} songs, {progress:.1%} of audio processed")'
```

## Current Execution Plan

### Phase A: stabilize and benchmark the current retrieval stack

Objective:

- get the full desktop `mert95/multiscale_v1` run complete
- inspect real outputs
- compare against `baseline_v1`

Deliverables:

- complete desktop embeddings
- query output samples
- evaluation report
- first model/profile comparison report

Exit criteria:

- the system can be run end-to-end on the desktop without intervention
- evaluation reports exist for at least `mert95/baseline_v1` and `mert95/multiscale_v1`

### Phase B: build a real judged evaluation set

Objective:

- stop making quality decisions from anecdotes alone

Required work:

- expand `data/eval/named_queries.csv` to at least `40-50` real queries
- populate `data/eval/human_triplets.csv` with at least `150` labeled triplets
- optionally add pairwise relevance labels for stronger comparison

Suggested query categories:

- within-artist prototypes
- artist A to artist B centroid matches
- nearest-neighbor sanity checks
- cross-artist pair discovery
- known expected failures
- duplicate/remaster edge cases

Exit criteria:

- quality discussions are grounded in saved labels, not only intuition

### Phase C: improve retrieval without retraining the base encoder

Objective:

- improve ranking quality on top of frozen embeddings

Priority items:

1. duplicate suppression
2. remaster/live-version filtering
3. heuristic reranking
4. richer segment-coverage scoring
5. better long-song balancing

Implementation direction:

- keep song-level cosine as stage-one retrieval
- rerank the top candidate set using segment-level evidence
- suppress near-duplicate variants by default

Exit criteria:

- judged metrics improve over raw cosine retrieval

### Phase D: add a learned reranker

Objective:

- train a lightweight model on top of frozen embeddings and human labels

Model concept:

- input features:
  - song cosine
  - best segment cosine
  - top-k segment statistics
  - coverage score
  - section-count difference
  - duration ratio
  - duplicate flag
- output:
  - scalar ranking score

Training data:

- human triplets
- pairwise relevance labels

Exit criteria:

- learned reranker beats the heuristic pipeline on the judged set

### Phase E: add review tooling

Objective:

- make labeling and result inspection faster

Target:

- a small local review UI or report browser
- fast playback of anchor and candidate clips
- one-click match quality labeling

Exit criteria:

- a labeling session can generate meaningful supervision quickly

## Future Directions: Better Regardless of Compute

If compute is not the limiting factor, the next-level improvements are:

### 1. Better base representation learning

Move from flat pooled segment embeddings to a hierarchical song encoder:

- local segment encoder
- section encoder
- full-song context encoder

This would preserve more information about:

- song structure
- pacing
- transitions
- arrangement evolution
- long-form texture development

### 2. Multi-objective training

Use a combination of:

- audio-audio contrastive learning
- playlist/co-listen weak supervision
- tag and mood prediction
- audio-text alignment
- preference ranking from human labels

This is stronger than pure self-supervised audio training if the target is human-perceived similarity.

### 3. Dual embeddings

Split the representation into:

- style / sonic texture embedding
- musical identity / songcraft embedding

Then allow query-time weighting between them.

This would help distinguish:

- "same production vibe"
- "same songwriting feel"
- "same groove"
- "same psychedelic atmosphere"

### 4. Text-conditioned retrieval

Add a second retrieval path that supports prompts such as:

- "most Beatles-like Pink Floyd song"
- "most futuristic but still warm track"
- "closest to late-night space rock"

This likely needs a better audio-text aligned system than the current pure audio retrieval path.

### 5. Better explainability

Future explainability upgrades:

- export best-matching clips automatically
- surface section-to-section alignments
- show why a result matched:
  - texture
  - rhythm
  - harmonic density
  - ambience
  - vocal presence

### 6. Library hygiene and entity awareness

A better system should understand and suppress:

- remasters
- alternate mixes
- live versions
- duplicate encodes
- short edits vs album versions

Without this, retrieval quality is often overstated.

## Future Directions: Unlimited-Scale R&D

If the project later moves into serious training research, the recommended order is:

1. collect better evaluation labels first
2. expand public training data only if legally and practically appropriate
3. train or adapt larger encoders only after the evaluation harness is mature

Potential data sources for research:

- FMA
- MTG-Jamendo
- MusicCaps-like captioned data
- licensed internal or private data if available

Strong R&D direction:

- initialize from a strong music encoder
- train a hierarchical song model
- fine-tune on preference labels from this project

This should only happen after the current evaluation loop is solid.

## Decision Log

### Decision: do not train from scratch now

Reason:

- library size is too small for a competitive from-scratch music encoder
- stronger gains are available from retrieval design and evaluation

### Decision: use WSL + Python venv

Reason:

- reproducible
- compatible with the ML stack
- works across both Windows machines

### Decision: keep raw files outside git

Reason:

- binary assets are large
- transfer is operational, not version-control material

### Decision: use profiles

Reason:

- model quality work needs reproducible experiments
- embeddings and reports must not overwrite one another

## Resume Instructions for Future Codex Sessions

When starting a fresh Codex session on the desktop machine, start from:

```bash
cd ~/code/that-sounds-like
source .venv/bin/activate
codex
```

Recommended session handoff prompt:

```text
Read PROJECT_STATUS_AND_ROADMAP.md first. Repo is at ~/code/that-sounds-like. Venv is active. music_raw contains 559 songs. Current goal: run tests, rebuild manifest, canonicalize full library, run full embeddings with `python -m thatsoundslike embed run --manifest artifacts/indexes/library_manifest.csv --model mert95 --profile multiscale_v1`, then continue evaluation and reports.
```

## Short-Term Definition of Done

The next meaningful milestone is complete when:

- desktop tests pass
- desktop manifest confirms `559` songs
- full desktop `mert95/multiscale_v1` embeddings complete
- post-run queries look plausible
- evaluation reports are generated
- a first batch of human triplets is collected

## Medium-Term Definition of Done

The next major milestone after that is complete when:

- duplicate suppression exists
- reranking exists
- evaluation labels are substantial
- model/profile comparisons are grounded in judged metrics

## Long-Term Definition of Done

This project reaches the "next level" when:

- it retrieves musically convincing results consistently
- it is explainable
- it is robust to duplicates and remasters
- it has a real evaluation loop
- it has a credible path to larger-scale representation learning

