#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/hccr-deep-learning
exec env CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_experiments.py --config configs/experiment/thesis_model_comparison_gpu0.yaml --resume --continue-on-error --show-output >> experiments/sweeps/thesis-model-comparison-single-gpu0/runner.log 2>&1
