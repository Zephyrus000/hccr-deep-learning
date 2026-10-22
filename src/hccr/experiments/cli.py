"""Command-line interface for seed sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from hccr.experiments.runner import run_experiments
from hccr.experiments.spec import load_experiment_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ordered HCCR seed sweeps and one-factor ablations."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--set",
        dest="base_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--variant",
        dest="variant_overrides",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "JSON/YAML mapping with name and args; use the YAML config for "
            "complex variants."
        ),
    )
    parser.add_argument("--profile-devices", choices=("cpu", "cuda"), nargs="+")
    parser.add_argument("--full-class-num-classes", type=int)
    parser.add_argument("--torchrun-nproc-per-node", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--show-output", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    spec = load_experiment_spec(arguments, project_root)
    run_experiments(
        spec,
        dry_run=arguments.dry_run,
        resume=arguments.resume,
        show_output=arguments.show_output,
        continue_on_error=arguments.continue_on_error,
    )
    return 0
