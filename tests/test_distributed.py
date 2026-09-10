from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from hccr.training.distributed import (
    destroy_distributed,
    initialize_distributed,
)


class DistributedContextTests(unittest.TestCase):
    def test_disabled_mode_is_a_no_op(self) -> None:
        context = initialize_distributed(False, "cpu")

        self.assertFalse(context.enabled)
        self.assertTrue(context.is_main_process)
        self.assertEqual(context.broadcast_object({"value": 1}), {"value": 1})
        self.assertEqual(context.all_gather_object("local"), ["local"])

    def test_torchrun_requires_explicit_distributed_flag(self) -> None:
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True),
            self.assertRaisesRegex(ValueError, "requires --distributed"),
        ):
            initialize_distributed(False, "cuda")

    def test_torchrun_context_uses_local_cuda_rank_and_cleans_up(self) -> None:
        environment = {
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "1",
            "LOCAL_WORLD_SIZE": "2",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("hccr.training.distributed.dist.is_available", return_value=True),
            patch(
                "hccr.training.distributed.dist.is_initialized",
                side_effect=[False, True],
            ),
            patch(
                "hccr.training.distributed.dist.is_nccl_available",
                return_value=True,
            ),
            patch("hccr.training.distributed.dist.init_process_group") as initialize,
            patch("hccr.training.distributed.dist.destroy_process_group") as destroy,
            patch(
                "hccr.training.distributed.torch.cuda.is_available",
                return_value=True,
            ),
            patch("hccr.training.distributed.torch.cuda.device_count", return_value=2),
            patch("hccr.training.distributed.torch.cuda.set_device") as set_device,
        ):
            context = initialize_distributed(True, "cuda")
            destroy_distributed(context)

        self.assertTrue(context.enabled)
        self.assertFalse(context.is_main_process)
        self.assertEqual(context.device, "cuda:1")
        self.assertEqual(context.backend, "nccl")
        set_device.assert_called_once_with(1)
        initialize.assert_called_once_with(backend="nccl")
        destroy.assert_called_once_with()
