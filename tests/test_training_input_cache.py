from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision.transforms.v2 import functional as vision_functional

from hccr.preprocessing import EvalPreprocessor, TrainPreprocessor
from hccr.training.input_cache import (
    build_training_cache,
    estimate_trainable_state_bytes,
    plan_cache_admission,
)


class PreparedFitDataset(Dataset):
    def __init__(self, count=11):
        self.transform = EvalPreprocessor(16)
        self.rows = [dict(split="train", sample_id=str(i)) for i in range(count)]
        self.reads = []

    def prepared(self, index):
        image = Image.new("L", (11, 13), color=255)
        image.putpixel((3, 3), 10 + index)
        image.putpixel((7, 9), 130 + index)
        return vision_functional.to_dtype(
            vision_functional.to_image(self.transform(image)), torch.float32, scale=True
        )

    def __getitem__(self, index):
        if self.rows[index]["split"] != "train":
            raise AssertionError("holdout image was decoded")
        self.reads.append(index)
        return self.prepared(index), index, ""

    def __len__(self):
        return len(self.rows)


class TrainingInputCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def build(self, dataset, **options):
        with patch(
            "hccr.training.input_cache._available_cpu_memory_bytes", return_value=2**30
        ):
            return build_training_cache(
                dataset, 4, "cpu", placement="cpu", reserve_gib=0, **options
            )

    def test_fit_images_decode_once_and_eval_pixels_remain_exact_across_epochs(self):
        dataset = PreparedFitDataset()
        cache, info = self.build(dataset)
        self.assertIsNotNone(cache)
        self.assertEqual(info["placement"], "cpu")
        self.assertEqual(info["status"], "cached")
        self.assertEqual(cache.images.dtype, torch.uint8)
        self.assertEqual(cache.targets.dtype, torch.int64)
        self.assertEqual(info["payload_bytes"], 11 * (16 * 16 + 8))
        self.assertEqual(len(cache), 3)
        self.assertEqual(dataset.reads, list(range(11)))
        for epoch in (1, 2):
            cache.set_epoch(epoch)
            seen = []
            for images, targets, tags in cache:
                self.assertEqual(images.dtype, torch.float32)
                self.assertEqual(images.device.type, "cpu")
                self.assertEqual(targets.dtype, torch.int64)
                self.assertEqual(tags, [""] * len(targets))
                for image, target in zip(images, targets, strict=True):
                    self.assertTrue(torch.equal(image, dataset.prepared(int(target))))
                    seen.append(int(target))
            self.assertEqual(sorted(seen), list(range(11)))
        self.assertEqual(dataset.reads, list(range(11)))

    def test_private_seed_plus_epoch_order_replays_without_global_rng_dependence(self):
        cache, _ = self.build(PreparedFitDataset())
        cache.set_epoch(4)
        first = torch.cat([targets for _, targets, _ in cache])
        expected = torch.randperm(11, generator=torch.Generator().manual_seed(7 + 4))
        self.assertEqual(first.tolist(), expected.tolist())
        torch.manual_seed(999)
        cache.set_epoch(4)
        self.assertEqual(torch.cat([t for _, t, _ in cache]).tolist(), first.tolist())
        cache.set_epoch(5)
        self.assertNotEqual(
            torch.cat([t for _, t, _ in cache]).tolist(), first.tolist()
        )

    def test_admission_boundary_accounts_for_cap_reserve_and_unknown_memory(self):
        self.assertTrue(plan_cache_admission(100, 130, 100, 30)["admitted"])
        self.assertFalse(plan_cache_admission(101, 130, 100, 30)["admitted"])
        self.assertEqual(
            plan_cache_admission(100, 129, 200, 30)["reason"],
            "memory_reserve_would_be_used",
        )
        self.assertEqual(
            plan_cache_admission(100, None, 200, 30)["reason"],
            "available_memory_unknown",
        )

    def test_cache_preparation_does_not_consume_global_torch_rng(self):
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        self.build(PreparedFitDataset())
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_refusal_precedes_decoding_and_auto_reports_streaming_fallback(self):
        dataset = PreparedFitDataset()
        with patch(
            "hccr.training.input_cache._available_cpu_memory_bytes", return_value=0
        ):
            with self.assertRaisesRegex(MemoryError, "refused"):
                build_training_cache(dataset, 4, "cpu", placement="cpu")
            cache, info = build_training_cache(dataset, 4, "cpu", placement="auto")
        self.assertIsNone(cache)
        self.assertEqual(info["status"], "stream_fallback")
        self.assertFalse(info["attempts"][0]["admitted"])
        self.assertEqual(dataset.reads, [])

    def test_hard_byte_cap_refuses_even_when_physical_memory_is_available(self):
        dataset = PreparedFitDataset()
        with self.assertRaisesRegex(MemoryError, "cache_budget_exceeded"):
            self.build(dataset, max_gib=1 / 1024**3)
        self.assertEqual(dataset.reads, [])

    def test_trainable_state_estimate_excludes_frozen_parameters(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2, bias=False))
        model[0].requires_grad_(False)
        self.assertEqual(estimate_trainable_state_bytes(model), 3 * 4 * 2 * 4)
        cache, info = self.build(PreparedFitDataset(), model=model)
        self.assertIsNotNone(cache)
        self.assertEqual(info["future_trainable_state_bytes"], 96)
        self.assertEqual(estimate_trainable_state_bytes(None), 0)

    def test_cache_rejects_holdouts_and_random_transforms_before_decoding(self):
        for split in ("validation", "calibration", "test", "icdar"):
            dataset = PreparedFitDataset()
            dataset.rows[-1]["split"] = split
            with (
                self.subTest(split=split),
                self.assertRaisesRegex(ValueError, "fit/train"),
            ):
                self.build(dataset)
            self.assertEqual(dataset.reads, [])
        dataset = PreparedFitDataset()
        dataset.transform = TrainPreprocessor(16)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            self.build(dataset)
        self.assertEqual(dataset.reads, [])

    def test_subset_uses_inferred_size_and_decodes_only_selected_fit_images(self):
        dataset = PreparedFitDataset()
        cache, info = self.build(Subset(dataset, [8, 2, 5]))
        self.assertEqual(info["image_size"], 16)
        self.assertEqual(info["sample_count"], 3)
        self.assertEqual(dataset.reads, [8, 2, 5])
        self.assertEqual(cache.targets.tolist(), [8, 2, 5])
        with self.assertRaisesRegex(ValueError, "image_size"):
            self.build(dataset, image_size=32)

    def test_cgroup_remaining_bytes_limit_host_memory_admission(self):
        values = {
            "/proc/meminfo": "MemAvailable: 1000 kB\n",
            "/sys/fs/cgroup/memory.max": "2000",
            "/sys/fs/cgroup/memory.current": "1400",
        }

        def read_text(path, *args, **kwargs):
            if path.as_posix() not in values:
                raise FileNotFoundError(path)
            return values[path.as_posix()]

        from hccr.training.input_cache import _available_cpu_memory_bytes

        with patch.object(Path, "read_text", read_text):
            self.assertEqual(_available_cpu_memory_bytes(), 600)

    def test_auto_allocation_failure_falls_back_without_cached_partial_tensors(self):
        dataset = PreparedFitDataset()
        with (
            patch(
                "hccr.training.input_cache._available_cpu_memory_bytes",
                return_value=2**30,
            ),
            patch(
                "hccr.training.input_cache._fill_cache", side_effect=MemoryError("full")
            ),
        ):
            cache, info = build_training_cache(
                dataset, 4, "cpu", placement="auto", reserve_gib=0
            )
        self.assertIsNone(cache)
        self.assertEqual(info["attempts"][0]["reason"], "allocation_failed")
        self.assertEqual(dataset.reads, [])

    def test_spawn_staging_workers_prepare_cpu_cache(self):
        cache, info = self.build(PreparedFitDataset(3), num_workers=1)
        self.assertEqual(info["build_workers"], 1)
        self.assertEqual(cache.targets.tolist(), [0, 1, 2])
        self.assertEqual(tuple(cache.images.shape), (3, 1, 16, 16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_auto_cuda_allocation_failure_recovers_into_host_cache(self):
        from hccr.training.input_cache import _fill_cache

        dataset = PreparedFitDataset(3)

        def fill(*args):
            if args[3].type == "cuda":
                raise torch.OutOfMemoryError("simulated cache allocation")
            return _fill_cache(*args)

        with (
            patch("hccr.training.input_cache._fill_cache", side_effect=fill),
            patch(
                "hccr.training.input_cache._available_cpu_memory_bytes",
                return_value=2**30,
            ),
            patch(
                "hccr.training.input_cache.torch.cuda.mem_get_info",
                return_value=(2**30, 2**30),
            ),
        ):
            cache, info = build_training_cache(
                dataset, 2, "cuda", placement="auto", reserve_gib=0
            )
        self.assertEqual(info["placement"], "cpu")
        self.assertEqual(info["attempts"][0]["reason"], "allocation_failed")
        self.assertEqual(dataset.reads, [0, 1, 2])
        images, targets, _ = next(iter(cache))
        self.assertEqual(images.device.type, "cuda")
        self.assertEqual(targets.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_cache_batches_stay_on_requested_device_with_identical_pixels(self):
        dataset = PreparedFitDataset()
        cache, info = build_training_cache(
            dataset, 4, "cuda", placement="cuda", max_gib=1, reserve_gib=0
        )
        self.assertEqual(info["placement"], "cuda")
        self.assertEqual(cache.images.device.type, "cuda")
        for images, targets, _ in cache:
            self.assertEqual(images.device.type, "cuda")
            self.assertEqual(targets.device.type, "cuda")
            for image, target in zip(images.cpu(), targets.cpu(), strict=True):
                self.assertTrue(torch.equal(image, dataset.prepared(int(target))))
