"""Run one official 3,755-class frozen CNN + HOG fusion pilot."""

from __future__ import annotations

import argparse

from hccr.training.hog_fusion import run_hog_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--min-baseline-top1", type=float, default=0.95)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.extract_batch_size, args.cpu_threads) < 1:
        parser.error("epochs, batch sizes and CPU threads must be positive")
    if args.learning_rate <= 0 or args.num_workers < 0:
        parser.error("learning rate must be positive and workers nonnegative")
    if not 0 <= args.min_baseline_top1 <= 1:
        parser.error("minimum baseline accuracy must be in [0, 1]")
    run_hog_fusion(args)


if __name__ == "__main__":
    main()
