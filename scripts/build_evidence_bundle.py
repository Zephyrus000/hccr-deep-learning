"""Build or validate a versioned paper-evidence bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from hccr.evidence import build_evidence_bundle, validate_evidence_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--sweep-id", action="append", default=[])
    parser.add_argument("--derived-root", type=Path, action="append", default=[])
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.validate is not None:
        if (
            arguments.output is not None
            or arguments.run_id
            or arguments.sweep_id
            or arguments.derived_root
        ):
            raise ValueError("--validate cannot be combined with build options")
        result = validate_evidence_bundle(arguments.validate)
    else:
        if arguments.output is None:
            raise ValueError("--output is required when building a bundle")
        manifest = build_evidence_bundle(
            arguments.experiments,
            arguments.output,
            run_ids=arguments.run_id or None,
            sweep_ids=arguments.sweep_id or None,
            derived_roots=arguments.derived_root or None,
        )
        result = {
            "bundle": str(arguments.output.resolve()),
            "runs": len(manifest["runs"]),
            "sweeps": len(manifest["sweeps"]),
            "derived_collections": len(manifest["derived_collections"]),
            "standalone_artifacts": len(manifest["standalone_artifacts"]),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
