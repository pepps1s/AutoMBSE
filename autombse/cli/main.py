from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from .commands.baseline import baseline_snapshot
from .commands.extract import extract_code_blocks_cmd
from .commands.eval import eval_projects_cmd, eval_views_cmd
from .commands.ingest import (
    ingest_examples_cmd,
    ingest_knowledge_cmd,
    ingest_pump_parts_cmd,
)
from .commands.pipeline import pipeline_run_cmd
from .commands.stub import stub_cmd
from .commands.verify import verify_rules_cmd, verify_tree_cmd
from .commands.validate import (
    validate_example_cmd,
    validate_res_cmd,
    validate_rule_states_cmd,
    validate_sysml_tree_cmd,
)
from ..config.loaders import load_config_for_command
from ..utils.paths import find_repo_root


class _AppendMethodsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        current = getattr(namespace, self.dest, None)
        if current is None:
            current = []
        if not isinstance(current, list):
            current = list(current)

        raw = values if isinstance(values, str) else str(values)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        current.extend(parts)
        setattr(namespace, self.dest, current)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autombse",
        description="AutoMBSE CLI (legacy-compatible migration).",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--version", action="version", version="autombse 0.1.0")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file.")
    parser.add_argument(
        "--cwd",
        type=str,
        help="Working directory (legacy scripts rely on relative paths).",
    )
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        dest="legacy_layout",
        help="Enable legacy paths/layout (default: disabled).",
    )
    parser.add_argument(
        "--no-legacy-layout",
        action="store_false",
        dest="legacy_layout",
        help="Disable legacy paths/layout.",
    )
    parser.add_argument("--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="End-to-end pipeline commands.",
        argument_default=argparse.SUPPRESS,
    )
    pipeline_sub = pipeline_parser.add_subparsers(dest="pipeline_cmd", metavar="<subcommand>")
    pipeline_run = pipeline_sub.add_parser(
        "run",
        help="Run the legacy-compatible pipeline.",
        argument_default=argparse.SUPPRESS,
    )
    pipeline_run.add_argument("--api-key", type=str)
    pipeline_run.add_argument("--base-url", type=str)
    pipeline_run.add_argument("--model", type=str)
    pipeline_run.add_argument(
        "--long-period",
        action="store_true",
        dest="long_period",
        help="Enable incremental long-period pipeline mode (writes to out/long_period/<run-id>/).",
    )
    pipeline_run.add_argument(
        "--no-long-period",
        action="store_false",
        dest="long_period",
        help="Disable long-period mode (default).",
    )
    pipeline_run.add_argument("--run-id", type=str, help="Long-period run id (default: auto).")
    pipeline_run.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help="Resume from the latest long-period state if available (default).",
    )
    pipeline_run.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start a fresh long-period run.",
    )
    pipeline_run.add_argument(
        "--input-path",
        action="append",
        dest="input_paths",
        help="Requirement/spec input path (file or dir); repeatable (long-period mode).",
    )
    pipeline_run.add_argument("--diagram-plan", type=str, help="Path to a diagram plan JSON (long-period mode).")
    pipeline_run.add_argument("--max-diagrams", type=int, help="Max diagrams to build in this run (long-period mode).")
    pipeline_run.add_argument(
        "--max-context-tokens",
        type=int,
        help="Max context tokens per LLM call (long-period mode).",
    )
    pipeline_run.add_argument(
        "--semantic-entropy-samples",
        type=int,
        help="Total samples for semantic entropy confidence (long-period mode; 1 disables extra sampling).",
    )
    pipeline_run.add_argument(
        "--semantic-entropy-temperature",
        type=float,
        help="Sampling temperature for semantic entropy probes (long-period mode).",
    )
    pipeline_run.add_argument(
        "--semantic-entropy-threshold",
        type=float,
        help="Jaccard threshold for semantic clustering in semantic entropy (long-period mode).",
    )
    pipeline_run.add_argument("--record", type=str, help="Record LLM responses to a JSONL file.")
    pipeline_run.add_argument("--replay", type=str, help="Replay LLM responses from a JSONL file.")
    pipeline_run.add_argument("--out-dir", type=str)
    pipeline_run.add_argument("--log-history", type=str)
    pipeline_run.add_argument("--log-code", type=str)
    pipeline_run.set_defaults(handler=pipeline_run_cmd, _command_path=("pipeline", "run"))

    eval_parser = subparsers.add_parser("eval", help="Evaluation commands.", argument_default=argparse.SUPPRESS)
    eval_sub = eval_parser.add_subparsers(dest="eval_cmd", metavar="<subcommand>")
    eval_views = eval_sub.add_parser("views", help="View-level evaluation.", argument_default=argparse.SUPPRESS)
    eval_views.add_argument("--input", type=str)
    eval_views.add_argument(
        "--method",
        "--methods",
        action=_AppendMethodsAction,
        help="Baseline method id(s) (repeatable; supports comma-separated list).",
    )
    eval_views.add_argument("--threshold", type=float, help="Semantic similarity threshold.")
    eval_views.add_argument("--cache-dir", type=str, help="Similarity cache directory.")
    eval_views.add_argument("--bleu-mbse-weight", type=float, help="Blend weight for BLEU_mbse = (1-w)*BLEU + w*Similarity.")
    eval_views.add_argument(
        "--force",
        action="store_true",
        help="Recompute derived fields (e.g. *_similarity, *_bleu) even if they already exist in res.json.",
    )
    eval_views.add_argument(
        "--generate-missing",
        action="store_true",
        dest="generate_missing",
        help="Generate missing method outputs for supported methods and write back to res.json (requires --update).",
    )
    eval_views.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regenerate selected method outputs for supported methods (requires --update).",
    )
    eval_views.add_argument(
        "--max-generate",
        type=int,
        help="Stop after generating at most this many missing examples (useful for long runs).",
    )
    eval_views.add_argument(
        "--element-extractor",
        type=str,
        choices=["auto", "strict", "legacy_semicolon"],
        help="Element extractor used for #Elements and element-level PRF (default: auto).",
    )
    eval_views.add_argument(
        "--element-extractor-by-method",
        action="append",
        dest="element_extractor_by_method",
        help="Override element extractor per method, e.g. MBSE_code_ds=legacy_semicolon (repeatable).",
    )
    eval_views.add_argument("--update", action="store_true", help="Update res.json with derived metric fields (default).")
    eval_views.add_argument("--no-update", action="store_false", dest="update", help="Do not modify res.json.")
    eval_views.set_defaults(handler=eval_views_cmd, _command_path=("eval", "views"))

    eval_projects = eval_sub.add_parser("projects", help="Project-level evaluation.", argument_default=argparse.SUPPRESS)
    eval_projects.add_argument("--input", type=str)
    eval_projects.add_argument(
        "--method",
        "--methods",
        action=_AppendMethodsAction,
        help="Baseline method id(s) (repeatable; supports comma-separated list).",
    )
    eval_projects.add_argument("--threshold", type=float, help="Semantic similarity threshold.")
    eval_projects.add_argument("--cache-dir", type=str, help="Similarity cache directory.")
    eval_projects.add_argument("--use-qdrant", action="store_true", dest="use_qdrant", help="Use Qdrant acceleration.")
    eval_projects.add_argument("--no-use-qdrant", action="store_false", dest="use_qdrant", help="Disable Qdrant.")
    eval_projects.add_argument("--created-flag", action="store_true", dest="created_flag", help="Skip recreate_collection.")
    eval_projects.add_argument(
        "--no-created-flag", action="store_false", dest="created_flag", help="Allow recreate_collection."
    )
    eval_projects.add_argument("--bleu-mbse-weight", type=float, help="Blend weight for BLEU_mbse = (1-w)*BLEU + w*Similarity.")
    eval_projects.add_argument("--update", action="store_true", help="Update res.json with derived metric fields (default).")
    eval_projects.add_argument("--no-update", action="store_false", dest="update", help="Do not modify res.json.")
    eval_projects.set_defaults(handler=eval_projects_cmd, _command_path=("eval", "projects"))

    verify_parser = subparsers.add_parser("verify", help="Verification commands.", argument_default=argparse.SUPPRESS)
    verify_sub = verify_parser.add_subparsers(dest="verify_cmd", metavar="<subcommand>")

    verify_tree = verify_sub.add_parser(
        "tree",
        help="Generate/validate sysml_tree.json (legacy wrapper).",
        argument_default=argparse.SUPPRESS,
    )
    verify_tree.add_argument("--input", type=str)
    verify_tree.add_argument("--output", type=str)
    verify_tree.set_defaults(handler=verify_tree_cmd, _command_path=("verify", "tree"))

    verify_rules = verify_sub.add_parser("rules", help="Run SysML rules (legacy wrapper).", argument_default=argparse.SUPPRESS)
    verify_rules.add_argument("--sysml", type=str)
    verify_rules.add_argument("--tree", type=str)
    verify_rules.add_argument("--rule-type", type=str, choices=["bdd", "cross", "all"])
    verify_rules.add_argument("--state-file", type=str)
    verify_rules.set_defaults(handler=verify_rules_cmd, _command_path=("verify", "rules"))

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest data into vector stores (legacy wrapper).",
        argument_default=argparse.SUPPRESS,
    )
    ingest_sub = ingest_parser.add_subparsers(dest="ingest_cmd", metavar="<subcommand>")

    ingest_examples = ingest_sub.add_parser(
        "examples",
        help="Ingest examples into Qdrant (legacy wrapper).",
        argument_default=argparse.SUPPRESS,
    )
    ingest_examples.add_argument("--input", type=str, help="Path to example.json.")
    ingest_examples.set_defaults(handler=ingest_examples_cmd, _command_path=("ingest", "examples"))

    ingest_knowledge = ingest_sub.add_parser(
        "knowledge",
        help="Ingest knowledge into Qdrant (legacy wrapper).",
        argument_default=argparse.SUPPRESS,
    )
    ingest_knowledge.add_argument("--input", type=str, help="Path to knowledge text file.")
    ingest_knowledge.set_defaults(handler=ingest_knowledge_cmd, _command_path=("ingest", "knowledge"))

    ingest_pump_parts = ingest_sub.add_parser(
        "pump-parts",
        help="Ingest pump parts into Qdrant (legacy wrapper).",
        argument_default=argparse.SUPPRESS,
    )
    ingest_pump_parts.add_argument("--input", type=str, help="Path to pump parts CSV.")
    ingest_pump_parts.set_defaults(handler=ingest_pump_parts_cmd, _command_path=("ingest", "pump-parts"))

    extract_parser = subparsers.add_parser("extract", help="Extraction utilities.", argument_default=argparse.SUPPRESS)
    extract_sub = extract_parser.add_subparsers(dest="extract_cmd", metavar="<subcommand>")
    extract_code_blocks = extract_sub.add_parser(
        "code-blocks",
        help="Extract SysML code blocks from res.json (legacy wrapper).",
        argument_default=argparse.SUPPRESS,
    )
    extract_code_blocks.add_argument("--input", type=str)
    extract_code_blocks.add_argument("--output", type=str)
    extract_code_blocks.set_defaults(handler=extract_code_blocks_cmd, _command_path=("extract", "code-blocks"))

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate artifacts against legacy v1 schemas.",
        argument_default=argparse.SUPPRESS,
    )
    validate_sub = validate_parser.add_subparsers(dest="validate_cmd", metavar="<artifact>")

    validate_res = validate_sub.add_parser("res", help="Validate res.json.", argument_default=argparse.SUPPRESS)
    validate_res.add_argument("--input", type=str)
    validate_res.add_argument(
        "--max-errors",
        type=int,
        help="Maximum number of schema errors to print (0 = print none).",
    )
    validate_res.set_defaults(handler=validate_res_cmd, _command_path=("validate", "res"))

    validate_tree = validate_sub.add_parser(
        "sysml-tree",
        help="Validate sysml_tree.json.",
        argument_default=argparse.SUPPRESS,
    )
    validate_tree.add_argument("--input", type=str)
    validate_tree.add_argument(
        "--max-errors",
        type=int,
        help="Maximum number of schema errors to print (0 = print none).",
    )
    validate_tree.set_defaults(handler=validate_sysml_tree_cmd, _command_path=("validate", "sysml-tree"))

    validate_states = validate_sub.add_parser(
        "rule-states",
        help="Validate rule_states.json.",
        argument_default=argparse.SUPPRESS,
    )
    validate_states.add_argument("--input", type=str)
    validate_states.add_argument(
        "--max-errors",
        type=int,
        help="Maximum number of schema errors to print (0 = print none).",
    )
    validate_states.set_defaults(handler=validate_rule_states_cmd, _command_path=("validate", "rule-states"))

    validate_example = validate_sub.add_parser("example", help="Validate example.json.", argument_default=argparse.SUPPRESS)
    validate_example.add_argument("--input", type=str)
    validate_example.add_argument(
        "--max-errors",
        type=int,
        help="Maximum number of schema errors to print (0 = print none).",
    )
    validate_example.set_defaults(handler=validate_example_cmd, _command_path=("validate", "example"))

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Baseline artifacts helpers.",
        argument_default=argparse.SUPPRESS,
    )
    baseline_sub = baseline_parser.add_subparsers(dest="baseline_cmd", metavar="<subcommand>")
    baseline_snapshot_parser = baseline_sub.add_parser(
        "snapshot",
        help="Create baseline manifest snapshot.",
        argument_default=argparse.SUPPRESS,
    )
    baseline_snapshot_parser.add_argument("--output", type=str)
    baseline_snapshot_parser.set_defaults(handler=baseline_snapshot, _command_path=("baseline", "snapshot"))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "cwd", None):
        os.chdir(args.cwd)

    repo_root = find_repo_root(Path.cwd())

    if not hasattr(args, "handler"):
        parser.print_help()
        return 2

    config = load_config_for_command(
        repo_root=repo_root,
        command_path=getattr(args, "_command_path", ()),
        config_path=getattr(args, "config", None),
        legacy_layout=bool(getattr(args, "legacy_layout", False)),
    )

    return int(args.handler(args=args, config=config, repo_root=repo_root))


__all__ = ["main"]
