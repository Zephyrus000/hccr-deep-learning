"""Thin command-line router for HCCR workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hccr import __version__
from hccr.commands import benchmark, compare_runs, deploy, train, validate

COMMAND_MODULES = (train, validate, deploy, benchmark, compare_runs)
LEGACY_SCAFFOLDS = ("prepare-data", "evaluate", "predict", "tune")
COMMANDS = tuple(command.NAME for command in COMMAND_MODULES) + LEGACY_SCAFFOLDS


def build_parser() -> argparse.ArgumentParser:
    """Build the stable top-level parser shared by the CLI and sweep runner."""
    parser = argparse.ArgumentParser(prog="hccr")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMAND_MODULES:
        command_parser = subparsers.add_parser(command.NAME, help=command.HELP)
        command.configure_parser(command_parser)
        command_parser.set_defaults(command_handler=command.run)
    for name in LEGACY_SCAFFOLDS:
        command_parser = subparsers.add_parser(
            name, help=f"Run the {name} workflow scaffold."
        )
        command_parser.set_defaults(command_handler=_run_legacy_scaffold)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a command and dispatch it to its owning module."""
    arguments = build_parser().parse_args(argv)
    return arguments.command_handler(arguments)


def _run_legacy_scaffold(arguments: argparse.Namespace) -> int:
    print(f"hccr {arguments.command}: workflow scaffold is ready for implementation.")
    return 0
