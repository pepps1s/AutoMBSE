# AutoMBSE

AutoMBSE is an engineering toolkit for **SysML v2 / MBSE**. Its core capability is to turn “requirements → view planning → SysML code generation → structured parsing → rule verification → evaluation” into a reproducible pipeline, with a single CLI entrypoint: `autombse`.

## Feature Overview

- **Pipeline**: end-to-end SysML v2 generation (optional RAG/Qdrant), supports `--record/--replay` for reproducible runs; also supports an incremental long-period mode via `--long-period`
- **Verify**: build a structure tree from SysML text and run rule checks (local rules + cross-view consistency)
- **Eval**: view-level evaluation for `AutoMBSE/out/views/res.json`, outputs `#Elements / Recall / Precision / F1 / Similarity / BLEU / BLEU_mbse`, and can incrementally write back derived fields
- **Validate**: JSON Schema validation for key artifacts (v1 contract)
- **Ingest**: ingest examples / knowledge / parts into Qdrant (optional)

## Install & Run

### 1) Run directly (no install)

```bash
cd AutoMBSE
python3 -m autombse --help
```

### 2) Install as a command (recommended)

```bash
python3 -m pip install -e AutoMBSE
autombse --help
```

### 3) Optional extras (install by feature)

```bash
python3 -m pip install -e 'AutoMBSE[rag]'
```

## Quick Start

### A. Long-period incremental generation (recommended for many views/diagrams)

```bash
autombse pipeline run --long-period \
  --input-path AutoMBSE/resource/knowledge/tmt_long_period_input.md
```

Semantic Entropy parameters (optional):

```bash
autombse pipeline run --long-period \
  --semantic-entropy-samples 5 \
  --semantic-entropy-temperature 0.7 \
  --semantic-entropy-threshold 0.85
```

Default output directory: `AutoMBSE/out/long_period/<run-id>/` (includes `state.json/index.json/events.jsonl/token_trace.csv/diagrams/*.sysml`, etc.).

### B. View-level evaluation

```bash
autombse eval views
```

Evaluate only specific methods (repeatable or comma-separated):

```bash
autombse eval views --method MBSE_code_ds --method MBSE_code_gpt
```

Incrementally write back derived fields (e.g., `*_bleu`; enabled by default, can be explicitly disabled):

```bash
autombse eval views --update
autombse eval views --no-update
```

Force recomputation of derived fields (e.g., you updated outputs or want to refresh cache):

```bash
autombse eval views --force
```

## Configuration & Environment Variables

Most commands automatically load the default config (`AutoMBSE/resource/autombse.params.v1.yaml`) and deep-merge it with the YAML/JSON provided by `--config <path>`.

Common environment variables:

- `OPENAI_API_KEY` / `AUTOMBSE_API_KEY`: LLM API Key
- `AUTOMBSE_BASE_URL`: LLM Base URL (OpenAI-compatible API)
- `AUTOMBSE_CACHE_DIR`: embedding cache directory for evaluation (default: `AutoMBSE/cache/`)
- `AUTOMBSE_PROMPTS_FILE`: override prompts library path (default: `AutoMBSE/resource/autombse.prompts.v1.yaml`)

