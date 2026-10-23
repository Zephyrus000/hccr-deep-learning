"""Export frozen-pilot calibration predictions and optional allocation controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from hccr.training.proposal_predictions import export_calibration_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--storage-backend", choices=("auto", "filesystem", "lmdb"), default="auto"
    )
    parser.add_argument("--lmdb-path", type=Path)
    parser.add_argument("--fit-frequency-output", type=Path)
    parser.add_argument("--directional-output", type=Path)
    parser.add_argument("--directional-rank", type=int)
    args = parser.parse_args()
    metadata = export_calibration_predictions(
        args.run_dir,
        args.manifest,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        storage_backend=args.storage_backend,
        lmdb_path=args.lmdb_path,
        fit_frequency_output=args.fit_frequency_output,
        directional_output=args.directional_output,
        directional_rank=args.directional_rank,
    )
    print(f"Exported {metadata['model_role']} calibration predictions to {args.output}")


if __name__ == "__main__":
    main()
