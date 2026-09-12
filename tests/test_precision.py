from __future__ import annotations

import unittest

from hccr.training.precision import PRECISION_NAMES, resolve_precision


class PrecisionTests(unittest.TestCase):
    def test_auto_cpu_resolves_to_float32(self) -> None:
        precision = resolve_precision("auto", "cpu")
        self.assertEqual(precision.resolved, "float32")
        self.assertFalse(precision.autocast_enabled)
        self.assertFalse(precision.uses_grad_scaler)

    def test_reduced_precision_requires_cuda(self) -> None:
        for requested in ("float16", "bfloat16"):
            with (
                self.subTest(requested=requested),
                self.assertRaisesRegex(ValueError, "requires CUDA"),
            ):
                resolve_precision(requested, "cpu")

    def test_metadata_records_requested_and_resolved_precision(self) -> None:
        precision = resolve_precision("float32", "cpu")
        self.assertEqual(
            precision.metadata(),
            {
                "requested": "float32",
                "resolved": "float32",
                "autocast_enabled": False,
                "autocast_dtype": None,
                "grad_scaler": False,
            },
        )

    def test_precision_names_are_stable_cli_choices(self) -> None:
        self.assertEqual(PRECISION_NAMES, ("auto", "float32", "float16", "bfloat16"))
