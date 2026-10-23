from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import yaml

from hccr.artifacts.checkpoints import file_digest
from hccr.ops.gpu_queue import run_gpu_queue


class GPUQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_text("audited-manifest", encoding="utf-8")
        self.output = self.root / "queue"

    def config(self, name="experiment", **values):
        path = self.root / f"{name}.yaml"
        payload = {
            "experiment_id": name,
            "manifest": str(self.manifest),
            "output_dir": str(self.root / f"unused-{name}"),
            "seeds": [7, 17, 29],
            "profile_devices": [],
            "base_args": {"num_classes": 3, "epochs": 1, "device": "cuda"},
            "variants": [{"name": name, "args": {}}],
        }
        payload.update(values)
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def run_queue(self, configs, **options):
        return run_gpu_queue(
            configs, [0, 1], self.output, project_root=self.root, **options
        )

    def fake_child(self, command, *, cwd, env, stdout, stderr, check):
        self.assertEqual(command[1:3], ["-m", "hccr.experiments"])
        self.assertIn("--resume", command)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["MKL_NUM_THREADS"], "1")
        config = yaml.safe_load(
            Path(command[command.index("--config") + 1]).read_text()
        )
        self.assertEqual(len(config["seeds"]), 1)
        self.assertEqual(len(config["variants"]), 1)
        destination = Path(config["output_dir"])
        run = destination / "20261023T000000Z-fake"
        run.mkdir(parents=True, exist_ok=True)
        (run / "checkpoint.pt").write_bytes(b"checkpoint")
        (run / "checkpoint_metadata.json").write_text(
            json.dumps(
                {
                    "training": {"seed": config["seeds"][0]},
                    "manifest_digest": file_digest(Path(config["manifest"])),
                }
            )
        )
        (run / "labels.json").write_text("{}")
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "validation": {
                        "best": {"top1": 0.5},
                        "selected_checkpoint": "checkpoint",
                    },
                    "test": {"status": "not_run"},
                }
            )
        )
        status = destination / "sweeps" / config["experiment_id"] / "status.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "records": [
                        {
                            "status": "completed",
                            "key": f"job/seed-{config['seeds'][0]}",
                            "seed": config["seeds"][0],
                            "run_path": str(run),
                        }
                    ],
                }
            )
        )
        stdout.write("child completed\n")
        return subprocess.CompletedProcess(command, 0)

    def test_plan_deduplicates_exact_resolved_jobs_without_launch_or_hardware(self):
        first, duplicate = self.config("first"), self.config("duplicate")
        with (
            patch("hccr.ops.gpu_queue._preflight") as hardware,
            patch("hccr.ops.gpu_queue.subprocess.run") as process,
        ):
            state = self.run_queue([first, duplicate, first])
        hardware.assert_not_called()
        process.assert_not_called()
        self.assertEqual(state["status"], "planned")
        self.assertEqual(len(state["jobs"]), 3)
        self.assertEqual(len(state["jobs"][0]["aliases"]), 3)
        self.assertEqual(len({job["child"]["output_dir"] for job in state["jobs"]}), 3)
        fingerprints = {job["fingerprint"] for job in state["jobs"]}
        state = self.run_queue([first, duplicate, first], resume=True)
        self.assertEqual({job["fingerprint"] for job in state["jobs"]}, fingerprints)

    def test_resolved_defaults_dedupe_but_scientific_changes_do_not(self):
        first = self.config("first", seeds=[7])
        duplicate = self.config(
            "duplicate",
            seeds=[7],
            base_args={
                "num_classes": 3,
                "epochs": 1,
                "device": "cuda",
                "learning_rate": 0.001,
            },
        )
        different = self.config(
            "different",
            seeds=[7],
            base_args={
                "num_classes": 3,
                "epochs": 1,
                "device": "cuda",
                "learning_rate": 0.002,
            },
        )
        state = self.run_queue([first, duplicate, different])
        self.assertEqual(len(state["jobs"]), 2)
        self.manifest.write_text("changed-manifest", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inputs/code changed"):
            self.run_queue([first, duplicate, different], resume=True)

    def test_scheduler_has_one_active_child_per_gpu_and_keeps_both_busy(self):
        config = self.config()
        barrier, lock = threading.Barrier(2), threading.Lock()
        active, peak, starts = {"0": 0, "1": 0}, {"0": 0, "1": 0}, []

        def child(command, **options):
            gpu = options["env"]["CUDA_VISIBLE_DEVICES"]
            with lock:
                active[gpu] += 1
                peak[gpu] = max(peak[gpu], active[gpu])
                starts.append(gpu)
                synchronize = len(starts) <= 2
            try:
                if synchronize:
                    barrier.wait(timeout=5)
                return self.fake_child(command, **options)
            finally:
                with lock:
                    active[gpu] -= 1

        with (
            patch("hccr.ops.gpu_queue._preflight", return_value=[]),
            patch("hccr.ops.gpu_queue.subprocess.run", side_effect=child),
        ):
            state = self.run_queue([config], execute=True)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(peak, {"0": 1, "1": 1})
        self.assertEqual(len(starts), 3)
        self.assertEqual(set(starts[:2]), {"0", "1"})
        self.assertTrue(all(job["artifacts"] for job in state["jobs"]))

    def test_resume_skips_only_artifact_hash_verified_completed_jobs(self):
        config = self.config(seeds=[7])
        with (
            patch("hccr.ops.gpu_queue._preflight", return_value=[]),
            patch(
                "hccr.ops.gpu_queue.subprocess.run", side_effect=self.fake_child
            ) as process,
        ):
            state = self.run_queue([config], execute=True)
            self.assertEqual(process.call_count, 1)
            resumed = self.run_queue([config], execute=True, resume=True)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(process.call_count, 1)
            artifact = next(
                Path(path)
                for path in state["jobs"][0]["artifacts"]
                if path.endswith("checkpoint.pt")
            )
            artifact.write_bytes(b"tampered")
            corrupted = self.run_queue([config], execute=True, resume=True)
            self.assertEqual(corrupted["status"], "failed")
            self.assertEqual(process.call_count, 1)
            self.assertTrue(corrupted["jobs"][0]["verification_failed"])
            artifact.write_bytes(b"checkpoint")
            restored = self.run_queue([config], execute=True, resume=True)
            self.assertEqual(restored["status"], "completed")
            self.assertEqual(process.call_count, 1)

    def test_failure_is_actionable_and_other_jobs_finish(self):
        config = self.config()

        def child(command, **options):
            payload = yaml.safe_load(
                Path(command[command.index("--config") + 1]).read_text()
            )
            if payload["seeds"] == [7]:
                return subprocess.CompletedProcess(command, 9)
            return self.fake_child(command, **options)

        with (
            patch("hccr.ops.gpu_queue._preflight", return_value=[]),
            patch("hccr.ops.gpu_queue.subprocess.run", side_effect=child),
        ):
            state = self.run_queue([config], execute=True)
        self.assertEqual(state["status"], "failed")
        failures = [job for job in state["jobs"] if job["status"] == "failed"]
        self.assertEqual(len(failures), 1)
        self.assertIn("code 9", failures[0]["error"])
        self.assertIn("inspect", failures[0]["error"])
        self.assertEqual(sum(job["status"] == "completed" for job in state["jobs"]), 2)

    def test_execute_preflight_fails_before_children_on_missing_manifest(self):
        config = self.config()
        self.manifest.unlink()
        with (
            patch("hccr.ops.gpu_queue.torch.cuda.device_count", return_value=0),
            patch("hccr.ops.gpu_queue.subprocess.run") as process,
            self.assertRaises(RuntimeError),
        ):
            self.run_queue([config], execute=True)
        process.assert_not_called()

    def test_ddp_and_duplicate_gpu_indices_are_rejected(self):
        config = self.config(base_args={"num_classes": 3, "distributed": True})
        with self.assertRaisesRegex(ValueError, "single-process"):
            self.run_queue([config])
        with self.assertRaisesRegex(ValueError, "unique"):
            run_gpu_queue([self.config()], [0, 0], self.output, project_root=self.root)

    def test_preflight_checks_bfloat16_names_and_manifest_without_starting_jobs(self):
        config = self.config(
            required_cuda_device_count=2,
            required_cuda_device_name="5090",
            base_args={"num_classes": 3, "precision": "bfloat16", "device": "cuda"},
        )
        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}),
            patch("hccr.ops.gpu_queue.torch.cuda.device_count", return_value=2),
            patch(
                "hccr.ops.gpu_queue.torch.cuda.device",
                side_effect=lambda _: nullcontext(),
            ),
            patch(
                "hccr.ops.gpu_queue.torch.cuda.get_device_name", return_value="RTX 5090"
            ) as names,
            patch(
                "hccr.ops.gpu_queue.torch.cuda.is_bf16_supported", return_value=False
            ) as bf16,
            patch(
                "hccr.ops.gpu_queue.subprocess.run", side_effect=self.fake_child
            ) as process,
        ):
            with self.assertRaisesRegex(RuntimeError, "bfloat16"):
                self.run_queue([config], execute=True)
            process.assert_not_called()
            bf16.return_value = True
            names.return_value = "other GPU"
            with self.assertRaisesRegex(RuntimeError, "names"):
                self.run_queue([config], execute=True)
            names.return_value = "RTX 5090"
            self.manifest.unlink()
            with self.assertRaisesRegex(ValueError, "Missing or changed"):
                self.run_queue([config], execute=True)
            process.assert_not_called()

    def test_changed_source_code_invalidates_a_saved_plan(self):
        source = self.root / "src" / "model.py"
        source.parent.mkdir()
        source.write_text("WEIGHTS = 1\n")
        config = self.config()
        self.run_queue([config])
        source.write_text("WEIGHTS = 2\n")
        with self.assertRaisesRegex(ValueError, "inputs/code changed"):
            self.run_queue([config], resume=True)


if __name__ == "__main__":
    unittest.main()
