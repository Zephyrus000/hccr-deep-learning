"""Experiment comparison command parser and handler."""

from __future__ import annotations

import argparse
from pathlib import Path

from hccr.experiments import compare_runs
from hccr.utils.experiment import write_json

NAME = "compare-runs"
HELP = "Compare a candidate experiment with a baseline quality gate."


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--min-top1-gain", type=float, default=0.0)
    parser.add_argument("--max-p95-latency-ratio", type=float, default=1.0)


def run(arguments: argparse.Namespace) -> int:
    verdict = compare_runs(
        arguments.summary,
        arguments.baseline,
        arguments.candidate,
        arguments.min_top1_gain,
        arguments.max_p95_latency_ratio,
    )
    write_json(arguments.summary.with_name("comparison.json"), verdict)
    print(verdict)
    return 0 if verdict["accepted"] else 1
