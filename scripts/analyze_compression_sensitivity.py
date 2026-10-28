"""Analyze a calibration-selected fixed mask on validation writers only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hccr.evaluation.compression_sensitivity import analyze_compression_sensitivity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-run", type=Path, required=True)
    parser.add_argument("--low-rank-run", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--storage-backend", choices=("lmdb", "auto", "filesystem"), default="lmdb"
    )
    parser.add_argument("--lmdb-path", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261024)
    args = parser.parse_args(argv)
    report = analyze_compression_sensitivity(
        args.dense_run,
        args.low_rank_run,
        args.mask,
        args.manifest,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        storage_backend=args.storage_backend,
        lmdb_path=args.lmdb_path,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "report": str(args.output),
                "split": report["split"],
                "signed_net_damage_gap": report["signed_net_damage_gap"],
                "mask_reselected": False,
                "training_started": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
