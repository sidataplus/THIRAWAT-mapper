# thirawat_mapper_beta CLI Guide


**T**erminology **H**armonization: **I**ntelligent **R**etrieval with **A**lignment and **W**eighting reranking via **A**utomated **T**ransformers

## Prerequisites

All commands below assume you are using [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

### Model access

1. Request access on Hugging Face: https://huggingface.co/na399/THIRAWAT-reranker-beta  (click "Access request" / accept terms)
2. Install Hugging Face CLI following https://huggingface.co/docs/huggingface_hub/en/guides/cli
3. Login via CLI so downloads work from code: `hf auth login`


## Setup with uv

```bash
# 1. Install dependencies into a local virtual environment (creates .venv/)
uv sync

# 2. (Optional) Activate the environment for interactive shells
source .venv/bin/activate

# 3. Or just run commands directly via uv
uv run python -m thirawat_mapper_beta.index.build --help
```

`uv sync` reads the project metadata and installs the required packages (PyTorch, LanceDB, transformers, etc.) against Python 3.11.x. Subsequent `uv run …` invocations will reuse the same environment. Replace paths in the examples below to match your workspace. All text used for indexing and inference is normalized (lower‑cased, whitespace collapsed) for stable matching.


## 1. Build a LanceDB Index

```bash
uv run python -m thirawat_mapper_beta.index.build \
  --duckdb data/derived/concepts.duckdb \
  --profiles-table concept_profiles \
  --concepts-table concept \
  --domain-id Drug \
  --concept-class-id "Clinical Drug,Quant Clinical Drug,Clinical Drug Comp,Clinical Drug Form,Ingredient" \
  --extra-column "concept_name,domain_id,vocabulary_id,concept_class_id" \
  --out-db data/lancedb/db \
  --table concepts_drug \
  --batch-size 256 \
  --device cuda
```

Key options:

- `--duckdb` – DuckDB file produced by [`sidataplus/athena2duckdb`](https://github.com/sidataplus/athena2duckdb).
- `--profiles-table` – Table containing `concept_id` and `profile_text` columns.
- `--concepts-table` – OMOP concept table (defaults to `concept`). The builder always joins to this table and keeps only standard, valid concepts (`standard_concept = 'S' AND invalid_reason IS NULL`).
- `--domain-id`, `--concept-class-id` – Optional filters; accept comma‑separated lists or repeated flags.
- `--extra-column` – Carry additional columns from the profiles table into LanceDB (repeat flag).
- `--out-db` / `--table` – Target LanceDB directory and table name.

The command will:

1. Load profiles (and apply filters if provided).
2. Normalize `profile_text` and embed with SapBERT CLS vectors (via `transformers`).
3. Write a LanceDB table where `vector` is a `FixedSizeList<float32>[768]` column.
4. Emit a `<table>_manifest.json` manifest describing the build (model id, filters, counts).


## 2. Bulk Inference

```bash
uv run python -m thirawat_mapper_beta.infer.bulk \
  --db data/lancedb/db \
  --table concepts_drug \
  --input data/usagi.csv \
  --out runs/mapping \
  --candidate-topk 100 \
  --device auto
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
5. Optionally apply the strength+Jaccard post‑scorer per query (disabled by default via `--post-weight 0.0`).

Outputs (written to `--out`):

- `results.csv` – Classic relabel layout (wide, block‑per‑query). Columns: leading `rank` 1..K, then for each query three adjacent columns `[match_rank_or_unmatched, source_concept_name, source_concept_code]` with K rows beneath.
- `results_with_input.csv` – Original input row with candidate columns appended.
- `results_usagi.csv` – Only when the input looks like Usagi (`sourceName`, `mappingStatus`, `matchScore` present). For each row with at least one candidate, the first candidate populates `matchScore`, `conceptId`, `conceptName`, `domainId`, and marks `mappingStatus=UNCHECKED`, `statusSetBy=THIRAWAT-mapper`, `mappingType=MAPS_TO`.
- `metrics.json` – When ground-truth IDs are available (either via `conceptId` or Usagi rows with `mappingStatus == APPROVED`) the file reports Hit@{1,2,5,10,20,50,100}, MRR@100, coverage, and counts.

Selected flags:

- `--source-name-column`, `--source-code-column` – Override input headers.
- `--label-column` – Column containing gold concept IDs (optional, default `conceptId`).
- `--status-column`, `--approved-value` – Configure Usagi approval detection.
- `--batch-size` – Query embedding batch size (increase for better GPU throughput).
- `--n-limit` – Limit to the first N rows (smoke runs).
- `--where` – Optional LanceDB filter, e.g., `vocabulary_id = 'RxNorm' AND concept_class_id != 'Ingredient'` (when those columns exist in the index).
- `--device` – `auto|cuda|mps|cpu` (default `auto` with safe fallback and fast matmul).
- `--post-weight` – Weight for simple post‑score blend (default `0.3`).


## 3. Interactive Query (REPL)

```bash
uv run python -m thirawat_mapper_beta.infer.query \
  --db data/lancedb/db \
  --table concepts_drug \
  --device auto
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


## Notes & Requirements

- Vector‑only retrieval + BMS reranking (no FTS/BM25/hybrid in beta).
- Text is normalized (lowercase + collapsed whitespace) for indexing and inference.
- PyTorch device: if `--device auto`, the runner prefers CUDA → MPS → CPU and enables fast matmul/TF32 where safe. Use `--batch-size` to increase GPU throughput.
- The reranker model `na399/THIRAWAT-reranker-beta` is a gated model on Hugging Face. You must request access on the model page (web) and login via the CLI before running.
- LanceDB tables must expose a float32 fixed‑size vector column (named `vector` when built with this CLI).
- Index build keeps only standard, valid OMOP concepts (`standard_concept='S' AND invalid_reason IS NULL`).
