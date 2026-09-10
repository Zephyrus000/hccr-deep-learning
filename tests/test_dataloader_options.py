from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hccr.training.distributed import DistributedContext
from hccr.training.workflow import TrainingConfig, _data_loader_options, _train_sampler


class DataLoaderOptionsTests(unittest.TestCase):
    def _config(self, **overrides) -> TrainingConfig:
        values = {
            "manifest_path": Path("manifest.csv"),
            "output_dir": Path("experiments"),
            "num_classes": 2,
            **overrides,
        }
        return TrainingConfig(**values)

    def test_cuda_workers_use_spawn_and_bounded_prefetch(self) -> None:
        options = _data_loader_options(
            self._config(num_workers=16, worker_timeout_seconds=90), "cuda:0"
        )

        self.assertEqual(options["multiprocessing_context"], "spawn")
        self.assertEqual(options["prefetch_factor"], 1)
        self.assertTrue(options["persistent_workers"])
        self.assertEqual(options["timeout"], 90)
        self.assertTrue(options["pin_memory"])
        self.assertTrue(callable(options["worker_init_fn"]))

    def test_single_process_loader_omits_multiprocessing_only_options(self) -> None:
        options = _data_loader_options(self._config(num_workers=0), "cpu")

        self.assertEqual(options, {"num_workers": 0, "pin_memory": False})

    def test_cpu_auto_keeps_platform_default_start_method(self) -> None:
        options = _data_loader_options(self._config(num_workers=4), "cpu")

        self.assertNotIn("multiprocessing_context", options)

    def test_spawn_workers_can_iterate_a_real_loader(self) -> None:
        options = _data_loader_options(
            self._config(
                num_workers=2,
                dataloader_start_method="spawn",
                persistent_workers=False,
                worker_timeout_seconds=30,
            ),
            "cpu",
        )
        loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2, **options)

        values = torch.cat([batch[0] for batch in loader])

        self.assertEqual(values.tolist(), list(range(8)))

    def test_distributed_train_sampler_shards_one_rank_without_shuffle_conflict(
        self,
    ) -> None:
        dataset = TensorDataset(torch.arange(8))
        context = DistributedContext(
            enabled=True,
            rank=1,
            world_size=2,
            local_rank=1,
            device="cuda:1",
            backend="nccl",
        )

        sampler = _train_sampler(dataset, self._config(), context)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)

        self.assertIsNotNone(sampler)
        assert sampler is not None
        self.assertEqual(sampler.num_replicas, 2)
        self.assertEqual(sampler.rank, 1)
        self.assertEqual(len(list(loader)), 2)
