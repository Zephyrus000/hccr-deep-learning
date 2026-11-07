"""Continue a completed HOG fusion pilot to a larger total epoch budget."""

import argparse

from hccr.training.hog_continuation import continue_hog_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.total_epochs < 1 or args.cpu_threads < 1:
        parser.error("epoch budget and CPU threads must be positive")
    continue_hog_training(args)


if __name__ == "__main__":
    main()
