"""Evaluate a validation-selected trained HOG fusion on full HWDB and ICDAR."""

from __future__ import annotations

import argparse

from hccr.evaluation.hog_fusion import run_hog_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.cpu_threads) < 1 or args.num_workers < 0:
        parser.error("batch size/CPU threads must be positive, workers nonnegative")
    run_hog_evaluation(args)


if __name__ == "__main__":
    main()
