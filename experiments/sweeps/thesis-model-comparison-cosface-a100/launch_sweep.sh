#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/hccr-deep-learning
mkdir -p experiments/sweeps/thesis-model-comparison-cosface-a100
exec .venv/bin/python scripts/run_experiments.py --config configs/experiment/thesis_model_comparison.yaml --resume --continue-on-error --show-output >> experiments/sweeps/thesis-model-comparison-cosface-a100/runner.log 2>&1
