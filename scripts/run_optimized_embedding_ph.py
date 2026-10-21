"""Benchmark/parity gate, then restart the full-class study and final TEST."""

from __future__ import annotations

import argparse
import os
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from scripts.benchmark_embedding_ph import benchmark
from scripts.run_embedding_ph_study import exclusive_worker
from scripts.run_embedding_ph_study import main as run_study

from hccr.tda.embedding.study import load_config
from hccr.tda.features import digest_file, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    arguments = parser.parse_args(argv)
    # Spawned PH processes inherit these before importing NumPy/PyTorch.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = load_config(arguments.config)
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_worker(root / "dispatch.lock"):
        if (root / "dispatch_status.json").exists():
            raise FileExistsError("optimized dispatch already exists; inspect it first")
        state = {
            "job_id": root.name,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "requested_config": str(arguments.config.resolve()),
            "requested_config_sha256": digest_file(arguments.config),
        }

        def notify(stage):
            state["stage"] = stage
            state["updated_at"] = datetime.now(UTC).isoformat()
            write_json(root / "dispatch_status.json", state)
            print(stage, flush=True)

        try:
            notify("benchmark_initializing")
            report = benchmark(
                config, arguments.bank, root / "benchmark", notify=notify
            )
            resolved = asdict(config)
            resolved["search_backend"] = report["chosen_backend"]
            resolved_path = root / "resolved_config.json"
            write_json(resolved_path, resolved)
            state["chosen_backend"] = report["chosen_backend"]
            state["subset_benchmark_speedup"] = report["chosen_speedup_vs_reference"]
            notify("full_class_study")
            code = run_study(["--config", str(resolved_path), "--threads", "12"])
            if code:
                raise RuntimeError(
                    "full-class worker failed; inspect seed-7/status.json"
                )
            state["status"] = "completed"
            notify("completed")
            return 0
        except Exception as error:
            state["status"] = "failed"
            state["error"] = f"{type(error).__name__}: {error}"
            state["failed_stage"] = state["stage"]
            notify("failed")
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
