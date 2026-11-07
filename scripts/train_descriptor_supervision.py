"""Prepare image topology targets or fine-tune a matched supervision variant."""

import argparse
import json
from pathlib import Path

from hccr.training.descriptor_supervision import (
    prepare_supervision_targets,
    read_supervision_config,
    run_descriptor_training,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-tda", "train"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/descriptor_supervision.yaml"),
    )
    parser.add_argument(
        "--method", choices=("none", "hog", "tda", "both"), required=True
    )
    parser.add_argument("--source-run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command != "train" and args.resume:
        parser.error("--resume applies only to training")
    config = read_supervision_config(
        args.config,
        args.method,
        seed=args.seed,
        source_run=args.source_run,
        output_dir=args.output_dir,
        device=args.device,
    )
    result = (
        prepare_supervision_targets(config)
        if args.command == "prepare-tda"
        else run_descriptor_training(config, resume=args.resume)
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
