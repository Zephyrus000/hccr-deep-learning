"""Compatibility wrapper for ``hccr benchmark``."""

from hccr.commands.benchmark import benchmark, build_parser, main

__all__ = ["benchmark", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
