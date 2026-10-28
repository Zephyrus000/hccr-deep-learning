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
from hccr.ops.gpu_queue import _plan, run_gpu_queue


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

    def test_stage_barrier_waits_for_last_teacher_before_objectives(self):
        teacher = self.config("teacher")
        objectives = self.config(
            "objectives", base_args={"num_classes": 3, "epochs": 2, "device": "cuda"}
        )
        last_teacher_finished = threading.Event()
        objective_started = threading.Event()
        lock = threading.Lock()
        completed_teachers = set()
        violations = []

        def child(command, **options):
            config = yaml.safe_load(
                Path(command[command.index("--config") + 1]).read_text()
            )
            seed = config["seeds"][0]
            if config["base_args"]["epochs"] == 1:
                if seed == 7:
                    self.assertTrue(last_teacher_finished.wait(timeout=3))
                    objective_started.wait(timeout=0.1)
                result = self.fake_child(command, **options)
                with lock:
                    completed_teachers.add(seed)
                if seed == 29:
                    last_teacher_finished.set()
                return result
            objective_started.set()
            with lock:
                if completed_teachers != {7, 17, 29}:
                    violations.append(seed)
            return self.fake_child(command, **options)

        with (
            patch("hccr.ops.gpu_queue._preflight", return_value=[]),
            patch("hccr.ops.gpu_queue.subprocess.run", side_effect=child),
        ):
            state = self.run_queue(
                [teacher, objectives], execute=True, stage_configs=True
            )
        self.assertEqual(state["status"], "completed")
        self.assertEqual(violations, [])
        self.assertEqual(
            [job["stage_index"] for job in state["jobs"]], [0, 0, 0, 1, 1, 1]
        )
        last_teacher_finished.clear()
        objective_started.clear()
        completed_teachers.clear()
        violations.clear()
        self.output = self.root / "without-barrier"
        with (
            patch("hccr.ops.gpu_queue._preflight", return_value=[]),
            patch("hccr.ops.gpu_queue.subprocess.run", side_effect=child),
        ):
            state = self.run_queue([teacher, objectives], execute=True)
        self.assertEqual(state["status"], "completed")
        self.assertTrue(
            violations, "The fixture must detect the default scheduler crossing stages"
        )

    def test_resume_rejects_changed_config_order_or_barrier_mode(self):
        first = self.config("first", seeds=[7])
        second = self.config("second", seeds=[17])
        state = self.run_queue([first, second], stage_configs=True)
        self.assertTrue(state["stage_configs"])
        self.assertEqual(
            state["config_order"], [str(first.resolve()), str(second.resolve())]
        )
        with self.assertRaisesRegex(ValueError, "order|barrier"):
            self.run_queue([second, first], resume=True, stage_configs=True)
        with self.assertRaisesRegex(ValueError, "order|barrier"):
            self.run_queue([first, second], resume=True, stage_configs=False)
        resumed = self.run_queue([first, second], resume=True, stage_configs=True)
        self.assertEqual(
            [job["fingerprint"] for job in resumed["jobs"]],
            [job["fingerprint"] for job in state["jobs"]],
        )

    def test_failed_stage_job_is_not_retried_and_next_stage_runs(self):
        teacher = self.config("teacher", seeds=[7, 17])
        objective = self.config(
            "objective",
            seeds=[29],
            base_args={"num_classes": 3, "epochs": 2, "device": "cuda"},
        )
        calls = []

        def child(command, **options):
            config = yaml.safe_load(
                Path(command[command.index("--config") + 1]).read_text()
            )
            seed = config["seeds"][0]
            calls.append(seed)
            if seed == 17:
                return subprocess.CompletedProcess(command, 1)
            return self.fake_child(command, **options)

        with (
            patch("hccr.ops.gpu_queue._preflight", return_value=[]),
            patch("hccr.ops.gpu_queue.subprocess.run", side_effect=child),
        ):
            state = self.run_queue(
                [teacher, objective], execute=True, stage_configs=True
            )
        self.assertEqual(state["status"], "failed")
        self.assertEqual(calls.count(17), 1)
        self.assertEqual(calls[-1], 29)
        self.assertEqual(
            [job["status"] for job in state["jobs"]],
            ["completed", "failed", "completed"],
        )

    def test_legacy_queue_defaults_to_no_barrier_and_cannot_enable_one_on_resume(self):
        config = self.config(seeds=[7])
        state = self.run_queue([config])
        state.pop("stage_configs")
        state.pop("config_order")
        (self.output / "status.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "barrier"):
            self.run_queue([config], resume=True, stage_configs=True)
        resumed = self.run_queue([config], resume=True)
        self.assertFalse(resumed["stage_configs"])

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

    def test_main_recipe_prioritizes_teachers_then_objectives_then_backbones(self):
        project = Path(__file__).resolve().parents[1]
        configs = []
        for group in ("teacher", "objectives", "backbones"):
            source = (
                project / "configs" / "experiment" / f"official_classifier_{group}.yaml"
            )
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            base = payload["base_args"]
            self.assertEqual(payload["seeds"], [7, 17, 29])
            self.assertEqual(base["epochs"], 20)
            self.assertEqual(base["batch_size"], 256)
            self.assertEqual(base["reference_batch_size"], 256)
            self.assertEqual(base["learning_rate"], 0.0001)
            self.assertEqual(base["scheduler_min_lr"], 0.000001)
            self.assertEqual(base["weight_decay"], 0.0001)
            self.assertEqual(base["lr_warmup_ratio"], 0.05)
            self.assertEqual(base["margin_warmup_ratio"], 0.1)
            self.assertEqual(base["precision"], "bfloat16")
            self.assertEqual(base["augmentation_backend"], "gpu")
            self.assertEqual(base["training_cache"], "auto")
            self.assertEqual(base["num_workers"], 12)
            payload["manifest"] = str(self.manifest)
            destination = self.root / f"{group}.yaml"
            destination.write_text(yaml.safe_dump(payload), encoding="utf-8")
            configs.append(destination)
        jobs, _ = _plan(configs, self.output, self.root)
        self.assertEqual(len(jobs), 30)
        aliases = [job["aliases"][0] for job in jobs]
        expected = [("dense_cosface_teacher", seed) for seed in (7, 17, 29)]
        expected += [
            (variant, seed)
            for variant in ("linear_ce", "normalized_softmax", "arcface")
            for seed in (7, 17, 29)
        ]
        expected += [
            (variant, seed)
            for variant in (
                "resnet18",
                "mobilenet_v3_small",
                "shufflenet_v2_x1_0",
                "efficientnet_b0",
                "mobilenetv4_conv_small",
                "repvit_m0_9",
            )
            for seed in (7, 17, 29)
        ]
        self.assertEqual(
            [(alias["variant"], alias["seed"]) for alias in aliases], expected
        )
        for job in jobs[:3]:
            self.assertEqual(
                [alias["variant"] for alias in job["aliases"]],
                ["dense_cosface_teacher", "cosface", "efficient_hccr"],
            )
        teacher_jobs, _ = _plan(configs[:1], self.output, self.root)
        self.assertEqual(
            [job["fingerprint"] for job in jobs[:3]],
            [job["fingerprint"] for job in teacher_jobs],
        )


if __name__ == "__main__":
    unittest.main()
