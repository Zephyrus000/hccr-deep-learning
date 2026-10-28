"""Export and benchmark a CPU dense-head INT8 control with an unchanged backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

from hccr.benchmarking.quantization import benchmark_classifier_quantization


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--engine")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="0 means all validation images",
    )
    args = parser.parse_args(argv)
    report = benchmark_classifier_quantization(**vars(args))
    print(f"Saved {report['scope']} report to {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
