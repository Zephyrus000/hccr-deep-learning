#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/hccr-deep-learning
exec env CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_experiments.py --config configs/experiment/thesis_model_comparison_gpu1.yaml --resume --continue-on-error --show-output >> experiments/sweeps/thesis-model-comparison-single-gpu1/runner.log 2>&1
