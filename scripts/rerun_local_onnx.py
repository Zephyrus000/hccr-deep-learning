"""Rerun only frozen dense-reference ONNX controls in a fresh evidence folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time
import traceback

import run_local_post_training as analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, default=Path("artifacts/downloads/vastai-20261026"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hwdb1_0-1_1_official_five_splits/manifest.csv"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=5)
    args = parser.parse_args()
    args.output, args.downloads, args.manifest = args.output.resolve(), args.downloads.resolve(), args.manifest.resolve()
    args.panel, args.calibration_panel = 4096, 1024
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory; prior evidence is never overwritten")
    args.output.mkdir(parents=True)
    # ORT creates calibration files through tempfile. Keep them inside the
    # selected writable evidence directory, including in Windows sandboxes.
    temporary = args.output / "temporary"
    temporary.mkdir()
    tempfile.tempdir = str(temporary)
    import torch
    import onnx
    import onnxruntime
    from hccr.models.artifacts import load_model_from_run
    torch.set_num_threads(args.threads)
    state = {"status": "running", "pid": os.getpid(), "started": time.time(), "training": False,
             "runtime": {"torch": str(torch.__version__), "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
             "temporary_directory": str(temporary),
             "optimizer_updates": 0, "test_images_read": 0, "icdar_images_read": 0, "seeds": {}}
    status_path = args.output / "status.json"
    analysis.write_json(status_path, state)
    try:
        rows = analysis.inventory(args)
        dense = [r for r in rows if r["group"] == "final_core" and r["variant"] == "dense_reference"]
        if sorted(int(r["seed"]) for r in dense) != [7, 17, 29]:
            raise ValueError("Expected exactly the three frozen dense-reference seeds")
        panels = analysis.reconcile_panel(args, rows)
        candidates = {}
        for row in dense:
            name = analysis.run_identity(row)
            state["seeds"][str(row["seed"])] = {"status": "running"}
            analysis.write_json(status_path, state)
            before = analysis.digest(Path(row["run"]) / "checkpoint.pt")
            model = load_model_from_run(Path(row["run"]), "cpu").float().eval()
            try:
                analysis.onnx_dense_control(args, row, model, panels)
                after = analysis.digest(Path(row["run"]) / "checkpoint.pt")
                if before != after:
                    raise ValueError("Frozen checkpoint changed during ONNX analysis")
                state["seeds"][str(row["seed"])] = {"status": "completed", "checkpoint_sha256_before": before,
                                                        "checkpoint_sha256_after": after}
                candidates[name] = model
            except Exception as exc:
                state["seeds"][str(row["seed"])] = {"status": "failed", "error": str(exc),
                                                        "traceback": traceback.format_exc()}
            analysis.write_json(status_path, state)
            print(f"ONNX {name}: {state['seeds'][str(row['seed'])]['status']}", flush=True)
        if all(s["status"] == "completed" for s in state["seeds"].values()):
            state["benchmark_status"] = "running"
            analysis.write_json(status_path, state)
            analysis.cpu_benchmarks(args, dense, panels, candidates)
            state["benchmark_status"] = "completed"
            state["status"] = "completed"
        else:
            state["status"] = "completed_with_failures"
    except Exception as exc:
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    state["finished"] = time.time()
    state["runner_sha256"] = analysis.digest(__file__)
    state["analysis_sha256"] = analysis.digest(analysis.__file__)
    analysis.write_json(status_path, state)
    print(f"{state['status']}: {args.output}", flush=True)
    return 0 if state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
