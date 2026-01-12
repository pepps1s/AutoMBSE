# `autombse/` Package Notes

`autombse` is AutoMBSE’s Python package and CLI implementation. It covers pipeline orchestration, SysML parsing and rule verification, evaluation, and optional RAG integration.

## Directory Layout

- `autombse/cli/`: argparse-based CLI (command registration and subcommand entrypoints)
- `autombse/pipeline/`: pipelines and long-period incremental generation (event logs, resumable runs, semantic-entropy confidence)
- `autombse/sysml/`: SysML v2 code block extraction, lightweight parse tree, domain knowledge templates, element extraction
- `autombse/verification/`: rule engine (local rules + cross-view consistency)
- `autombse/evaluation/`: view-level / project-level evaluation (BERT CLS embedding cache, BLEU, element-level PRF)
- `autombse/rag/`: Qdrant retrieval and knowledge stitching (optional)

## Common Commands

```bash
cd AutoMBSE
python3 -m autombse --help

python3 -m autombse pipeline run --help
python3 -m autombse eval views --help
```

## Tests

```bash
cd AutoMBSE
python3 -m unittest discover -s tests
```
