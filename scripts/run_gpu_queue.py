"""Plan independent GPU jobs; launch them only with --execute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hccr.ops.gpu_queue import run_gpu_queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stage-configs",
        action="store_true",
        help="Finish every job from each config before starting the next config",
    )
    arguments = parser.parse_args(argv)
    state = run_gpu_queue(
        arguments.config,
        arguments.gpus,
        arguments.output_dir,
        execute=arguments.execute,
        resume=arguments.resume,
        stage_configs=arguments.stage_configs,
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "jobs": len(state["jobs"]),
                "status_file": str(arguments.output_dir / "status.json"),
            }
        )
    )
    return 1 if state["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
