"""Run cached-embedding development, final TEST and matched accuracy comparison."""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch

from hccr.tda.embedding.study import EmbeddingStudy, load_config
from hccr.tda.features import digest_file, write_json


@contextmanager
def exclusive_worker(path: Path):
    """OS releases this nonblocking lock even when the worker crashes."""
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def accuracy_comparison(selection: dict, heldout: dict) -> dict:
    metrics = heldout["held_out"]["test"]
    counts = {item["samples"] for item in metrics.values()}
    if len(counts) != 1 or set(metrics) != {"cnn", "cnn_knn", "distance", "ph"}:
        raise ValueError(
            "TEST controls have different sample counts or missing results"
        )
    ph = metrics["ph"]["top1"]
    return {
        "fingerprint": heldout["fingerprint"],
        "seed": heldout["seed"],
        "classes": heldout["classes"],
        "test_samples": counts.pop(),
        "top1": {name: item["top1"] for name, item in metrics.items()},
        "ph_delta_percentage_points": {
            name: 100 * (ph - metrics[name]["top1"])
            for name in ("cnn", "cnn_knn", "distance")
        },
        "ph_beats_all_matched_controls": all(
            ph > metrics[name]["top1"] for name in ("cnn", "cnn_knn", "distance")
        ),
        "selection_signature": selection["selection_signature"],
        "scope": "one_seed_frozen_backbone_refitted_heads_exploratory_test",
        "statistical_significance_established": False,
        "writer_disjoint_verified": heldout["writer_disjoint_verified"],
        "historical_checkpoint_score_used_as_reference": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    config = load_config(arguments.config)
    root = Path(config.output_dir).resolve() / f"seed-{config.seed}"
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    with exclusive_worker(root / "runner.lock"):
        config_hash = digest_file(arguments.config)
        if status_path.exists():
            prior = json.loads(status_path.read_text(encoding="utf-8"))
            if not arguments.resume:
                raise FileExistsError("job exists; inspect status and use --resume")
            if prior["config_sha256"] != config_hash:
                raise ValueError("config changed; use a separate output_dir")
            if prior["status"] == "completed":
                print(json.dumps({"already_completed": str(status_path)}))
                return 0
        logger = logging.getLogger("embedding_ph_worker")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(root / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        state = {
            "job_id": root.parent.name,
            "status": "running",
            "worker_pid": os.getpid(),
            "config": str(arguments.config.resolve()),
            "config_sha256": config_hash,
            "runner_sha256": digest_file(Path(__file__)),
            "started_at": datetime.now(UTC).isoformat(),
            "classes": config.max_classes,
            "seed": config.seed,
            "device": config.device,
            "threads": arguments.threads,
            "stage": "initializing_sources",
            "completed_stages": [],
        }

        def stage(name):
            state["stage"] = name
            state["updated_at"] = datetime.now(UTC).isoformat()
            write_json(status_path, state)
            logger.info("stage=%s", name)

        try:
            torch.set_num_threads(arguments.threads)
            stage("initializing_sources")
            with EmbeddingStudy(config) as study:
                state["fingerprint"] = study.fingerprint
                state["train_samples"] = len(study.train.targets)
                state["validation_samples"] = len(study.validation.targets)
                stage("prepare_bank")
                bank = study.prepare()
                state["bank_samples"] = len(bank.targets)
                state["completed_stages"].append("prepare_bank")
                for split in ("train", "validation"):
                    stage(f"extract_{split}")
                    with study.extract(split):
                        pass
                    state["completed_stages"].append(f"extract_{split}")
                stage("compare_validation")
                selection_path = study.output / "selection.json"
                if not selection_path.exists():
                    study.compare()
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                state["completed_stages"].append("compare_validation")
                stage("evaluate_test")
                heldout = study.evaluate(("test",))
                state["completed_stages"].append("evaluate_test")
                stage("accuracy_comparison")
                report = accuracy_comparison(selection, heldout)
                write_json(root / "accuracy_comparison.json", report)
                logger.info("TEST comparison: %s", json.dumps(report))
                state["completed_stages"].append("accuracy_comparison")
            state["status"] = "completed"
            stage("completed")
            return 0
        except Exception as error:
            state["status"] = "failed"
            state["error"] = f"{type(error).__name__}: {error}"
            state["failed_stage"] = state["stage"]
            stage("failed")
            logger.exception("Embedding PH worker failed")
            return 1
        finally:
            state["finished_at"] = datetime.now(UTC).isoformat()
            write_json(status_path, state)
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
