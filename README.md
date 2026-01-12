# AutoMBSE

AutoMBSE is an experimental and engineering-oriented scaffold project for **SysML v2 / MBSE**. It unifies end-to-end capabilities for generation, evaluation, validation, extraction, ingestion, and baseline management under the `autombse` command-line interface, and provides stable, composable configuration and contract mechanisms.

The repository is designed with reproducibility, configurability, and verifiability as core objectives.

---

## Overview

* **Generation (Pipeline)**: Generates SysML v2 model code from requirements and retrieval-augmented context using LLMs, with support for long-period incremental generation.
* **Evaluation (Eval)**: Performs view-level and project-level metric evaluation across different methods or stages, including BLEU, semantic similarity, and element-level Precision / Recall / F1.
* **Validation (Validate / Verify)**:

  * `validate`: Uses JSON Schema to validate the structure and field completeness of key artifacts.
  * `verify`: Parses SysML model structures and applies rule checks on model trees or textual representations.
* **Extraction (Extract)**: Extracts SysML fenced code blocks from generated outputs and produces structured results suitable for downstream processing.
* **Ingestion (Ingest)**: Vectorizes examples, domain knowledge, or part data for retrieval and evaluation acceleration.
* **Baseline Snapshot (Baseline)**: Generates hash manifests for key artifacts and caches to support experiment reproducibility and regression comparison.

---

## Installation and Environment

### Prerequisites

* Python `>= 3.8`
* Use of a virtual environment (venv or conda) is recommended

### Local Installation (Editable)

Execute the following commands at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Install CLI and core dependencies
python -m pip install -e AutoMBSE
```

To enable retrieval augmentation, evaluation, BERT embeddings, or vector ingestion, install optional dependencies:

```bash
# RAG / evaluation / vector retrieval
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e "AutoMBSE[rag]"

# Data ingestion capabilities
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e "AutoMBSE[ingest]"
```

---

## Quick Start

### 1) Set LLM-related Environment Variables

AutoMBSE reads credentials and service endpoints from environment variables:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"        # or AUTOMBSE_API_KEY
export AUTOMBSE_BASE_URL="https://api.xxx/v1"   # optional
```

Optional template variable:

```bash
export AUTOMBSE_TARGET_SYSTEM="Target system name"
```

### 2) View CLI Help

```bash
autombse --help
python -m autombse --help
```

### 3) Run the End-to-End Generation Pipeline

```bash
autombse pipeline run
```

Common optional parameters:

* `--out-dir <path>`: Output directory
* `--log-history <path>`: Textual generation log path
* `--log-code <path>`: Generated SysML code log path
* `--record <path>`: Record prompt / context / response rounds to JSONL
* `--replay <path>`: Replay generation results from JSONL without LLM calls

---

## Configuration Mechanism

### Default Configuration Sources

* Default parameters: `AutoMBSE/resource/autombse.params.v1.yaml`
* Prompt template library: `AutoMBSE/resource/autombse.prompts.v1.yaml`

The CLI accepts YAML or JSON configuration files via `--config` and performs deep merging with defaults.

### Configuration Priority

1. Explicit CLI parameters
2. Environment variables
3. Configuration files specified by `--config`
4. `defaults` in the built-in parameter file

### Common Environment Variables

* `OPENAI_API_KEY` / `AUTOMBSE_API_KEY`
* `AUTOMBSE_BASE_URL`
* `AUTOMBSE_TARGET_SYSTEM`
* `AUTOMBSE_PARAMS_FILE` / `AUTOMBSE_RESOURCE_PARAMS`
* `AUTOMBSE_PROMPTS_FILE` / `AUTOMBSE_RESOURCE_PROMPTS`
* `AUTOMBSE_CACHE_DIR`

### Prompt References and Variable Substitution

Configuration files support the following mechanisms:

* `@prompt:<id>`: Reference a template from the prompt library
* `{{var}}` / `{{vars.var}}`: Template variable substitution

Example:

```yaml
template:
  vars:
    target_system: "Example system"

llm:
  model: "gpt-4.1"
  system_prompt: "@prompt:system.sysml_general"

pipeline:
  stages:
    - id: "BDD"
      task: "@prompt:pipeline.tasks.bdd"
```

---

## CLI Usage Guide

All commands support the following global optional parameters:

* `--config <path>`: Configuration file path
* `--cwd <path>`: Working directory
* `--version`: Print version and exit
* `--verbose`: Enable verbose logging

### Path Resolution Rules

* When a path parameter is explicitly provided:

  * Absolute paths are resolved as-is
  * Relative paths are resolved against the current working directory
* When no path parameter is provided, default paths are used

### `pipeline run`

```bash
autombse pipeline run
```

Optional parameters:

* `--api-key`
* `--base-url`
* `--model`
* `--long-period / --no-long-period`
* `--out-dir`
* `--log-history <path>`
* `--log-code <path>`
* `--record <path>`
* `--replay <path>`

### Long-Period Incremental Generation

```bash
autombse pipeline run --long-period \
  --input-path <path> \
  --run-id <id>
```

Optional parameters:

* `--run-id <id>`
* `--resume / --no-resume`
* `--input-path <path>` (repeatable)
* `--diagram-plan <path>`
* `--max-diagrams <n>`
* `--max-context-tokens <n>`
* `--semantic-entropy-samples <n>`
* `--semantic-entropy-temperature <float>`
* `--semantic-entropy-threshold <float>`

### `validate`

```bash
autombse validate res
autombse validate sysml-tree
autombse validate rule-states
autombse validate example
```

Optional parameters (all `validate <artifact>` commands):

* `--input <path>`
* `--max-errors <n>` (0 = print none)

### `verify`

```bash
autombse verify tree
autombse verify rules --sysml <path>
autombse verify rules --tree <path>
```

Optional parameters:

* `--input <path>` (for `verify tree`)
* `--output <path>` (for `verify tree`)
* `--sysml <path>` (for `verify rules`; exclusive with `--tree`)
* `--tree <path>` (for `verify rules`; exclusive with `--sysml`)
* `--rule-type <bdd|cross|all>`
* `--state-file <path>`

### `eval`

```bash
autombse eval views
autombse eval projects
```

Optional parameters (`eval views`):

* `--input <path>`
* `--method/--methods <id>[,<id>...]` (repeatable; supports comma-separated list)
* `--threshold <float>`
* `--cache-dir <path>`
* `--bleu-mbse-weight <float>`
* `--update / --no-update`
* `--force`
* `--generate-missing` (requires `--update`)
* `--regenerate` (requires `--update`)
* `--max-generate <n>`
* `--element-extractor <auto|strict>`
* `--element-extractor-by-method <method>=<mode>` (repeatable)

Optional parameters (`eval projects`):

* `--input <path>`
* `--method/--methods <id>[,<id>...]` (repeatable; supports comma-separated list)
* `--threshold <float>`
* `--cache-dir <path>`
* `--use-qdrant / --no-use-qdrant`
* `--created-flag / --no-created-flag`
* `--update / --no-update`
* `--bleu-mbse-weight <float>`

### `ingest`

```bash
autombse ingest examples
autombse ingest knowledge
autombse ingest pump-parts --input <csv_path>
```

Optional parameters:

* `--input <path>`

### `extract`

```bash
autombse extract code-blocks
```

Optional parameters:

* `--input <path>`
* `--output <path>`

### `baseline snapshot`

```bash
autombse baseline snapshot --output <path>
```

Optional parameters:

* `--output <path>` (default: `AutoMBSE/artifacts/baseline_manifest.json`)

---

## Artifacts and Directory Structure

* `AutoMBSE/out/`

  * `log-history.md`
  * `log-code.md`
  * Generated views and model artifacts
* `AutoMBSE/cache/`

  * Embedding and similarity caches
* `AutoMBSE/resource/`

  * Default parameters, prompts, schemas, and example resources

---

## Development and Testing

```bash
cd AutoMBSE
python -m unittest discover -s tests
```
