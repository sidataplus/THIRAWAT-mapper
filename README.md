# thirawat_mapper_beta CLI Guide

The `thirawat_mapper_beta` package ships a minimal toolchain for building a LanceDB index and running retrieval + reranking without any LLM/RAG components. This guide covers the three entry points:

- `thirawat_mapper_beta.index.build`
- `thirawat_mapper_beta.infer.bulk`
- `thirawat_mapper_beta.infer.query`

All commands below assume you are using [`uv`](https://github.com/astral-sh/uv) with the provided `pyproject.toml`.


## Setup with uv

```bash
# 1. Install dependencies into a local virtual environment (creates .venv/)
uv sync

# 2. (Optional) Activate the environment for interactive shells
source .venv/bin/activate

# 3. Or just run commands directly via uv
uv run python -m thirawat_mapper_beta.index.build --help
```

`uv sync` reads the project metadata and installs the required packages (PyTorch, LanceDB, transformers, etc.) against Python 3.11.x. Subsequent `uv run …` invocations will reuse the same environment. Replace paths in the examples below to match your workspace.


## 1. Build a LanceDB Index

```bash
uv run python -m thirawat_mapper_beta.index.build \
  --duckdb data/derived/concepts.duckdb \
  --profiles-table concept_profiles \
  --concepts-table concept \
  --domain-id Drug \
  --concept-class-id "Clinical Drug,Quant Clinical Drug,Clinical Drug Comp,Clinical Drug Form,Ingredient" \
  --extra-column concept_name --extra-column domain_id --extra-column vocabulary_id \
  --out-db data/lancedb/db \
  --table concepts_drug \
  --batch-size 256 \
  --device cuda
```

Key options:

- `--duckdb` – DuckDB file produced by [`sidataplus/athena2duckdb`](https://github.com/sidataplus/athena2duckdb).
- `--profiles-table` – Table containing `concept_id` and `profile_text` columns.
- `--concepts-table` – OMOP concept table for filtering by `domain_id` and/or `concept_class_id`.
- `--domain-id`, `--concept-class-id` – Optional filters; accept comma‑separated lists or repeated flags.
- `--extra-column` – Carry additional columns from the profiles table into LanceDB (repeat flag).
- `--out-db` / `--table` – Target LanceDB directory and table name.

The command will:

1. Load profiles (and apply filters if provided).
2. Encode `profile_text` with SapBERT CLS vectors (via `transformers`).
3. Write a LanceDB table where `vector` is a `FixedSizeList<float32>[768]` column.
4. Emit a `<table>_manifest.json` manifest describing the build (model id, filters, counts).


## 2. Bulk Inference

```bash
uv run python -m thirawat_mapper_beta.infer.bulk \
  --db data/lancedb/db \
  --table concepts_drug \
  --input data/eval/tmt_to_rxnorm.csv \
  --out runs/thirawat_beta_drug \
  --candidate-topk 100 \
  --device cuda
```

Input formats: CSV, TSV, Parquet, or Excel. By default the CLI expects the following columns (override via flags):

- `sourceName` (required)
- `sourceCode` (optional)
- `conceptId` (optional ground truth)
- `mappingStatus` (used for Usagi detection)

Pipeline steps per row:

1. Build query text (`sourceName` with `sourceCode` appended in parentheses when present).
2. Embed with SapBERT.
3. Vector search (cosine) against the LanceDB table to gather `--candidate-topk` entries.
4. Rerank with the THIRAWAT reranker (BMS). Beta is vector‑only; no FTS/BM25/hybrid.
5. Apply the strength+Jaccard post‑scorer per query (min‑max normalized) and blend with the reranker score.

Outputs (written to `--out`):

- `results.csv` – Classic relabel-wide format: `top1_*` … `topK_*` columns.
- `results_with_input.csv` – Original input row with candidate columns appended.
- `results_usagi.csv` – Only when the input looks like Usagi (`sourceName`, `mappingStatus`, `matchScore` present). For each row with at least one candidate, the first candidate populates `matchScore`, `conceptId`, `conceptName`, `domainId`, and marks `mappingStatus=UNCHECKED`, `statusSetBy=THIRAWAT-mapper`, `mappingType=MAPS_TO`.
- `metrics.json` – When ground-truth IDs are available (either via `conceptId` or Usagi rows with `mappingStatus == APPROVED`) the file reports Hit@{1,2,5,10,20,50,100}, MRR@100, coverage, and counts.

Selected flags:

- `--source-name-column`, `--source-code-column` – Override input headers.
- `--label-column` – Column containing gold concept IDs (default `conceptId`).
- `--status-column`, `--approved-value` – Configure Usagi approval detection.
- `--batch-size` – Query embedding batch size.


## 3. Interactive Query (REPL)

```bash
uv run python -m thirawat_mapper_beta.infer.query \
  --db data/lancedb/db \
  --table concepts_drug \
  --device cuda
```

Type a query and press Enter to see the post-scored top results:

```
query> amoxicillin clavulanate 875 mg
concept_id   | score  | s_sim | name
--------------------------------------------------------------------------------
123456       | 0.841  | 0.990 | Amoxicillin / Clavulanate 875 MG Oral Tablet
...
```

Commands:

- Type `:q`, `:quit`, or `:exit` to leave.
- Use `--candidate-topk` to change the candidate pool and `--show-topk` to limit display rows.

The REPL shares the same SapBERT encoder, THIRAWAT reranker, and post-scorer used in bulk mode.


## Notes & Requirements

- PyTorch with GPU is optional but recommended for large batches. If `--device` is omitted, the embedder auto-detects CUDA.
- The reranker model `na399/THIRAWAT-reranker-beta` is downloaded via Hugging Face on first use; ensure you have access tokens configured if required.
- LanceDB tables must expose a float32 fixed-size vector column (named `vector` when built with the provided index CLI).
- Contact `max@sidata.plus` for prebuilt LanceDB archives if you need a shortcut.

### Quick smoke (first 50 rows)

```bash
head -n 51 data/eval/tmt_to_rxnorm.csv > runs/smoke_input_50.csv
uv run python -m thirawat_mapper_beta.infer.bulk \
  --db data/lancedb/db \
  --table concepts_drug \
  --input runs/smoke_input_50.csv \
  --out runs/smoke_50 \
  --candidate-topk 100 \
  --device cuda
```
