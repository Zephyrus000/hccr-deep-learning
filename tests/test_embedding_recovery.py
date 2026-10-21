from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scripts.recover_embedding_ph_cache import (
    reuse_development_caches,
    verify_storage_patch,
)

from hccr.tda.embedding import study as study_module
from hccr.tda.embedding.smoke import create_synthetic_caches
from hccr.tda.embedding.study import EmbeddingConfig, EmbeddingStudy
from hccr.tda.features import digest_file, write_json


class EmbeddingRecoveryTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prior, self.target = self.root / "prior", self.root / "recovered"
        source = self.root / "source"
        create_synthetic_caches(source)
        self.snapshot = self.root / "features.before.py"
        features = Path(__file__).parents[1] / "src/hccr/tda/features.py"
        self.snapshot.write_text(
            features.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        self.before_hash = digest_file(self.snapshot)
        old = copy.deepcopy(study_module.implementation_identity())
        old["source"]["tda/features.py"] = self.before_hash
        config = EmbeddingConfig(
            str(source),
            str(self.prior),
            max_classes=3,
            bank_per_class=4,
            neighbors=3,
            query_batch_size=4,
            reference_batch_size=5,
            head_epochs=1,
            head_batch_size=8,
            alpha_grid=(1.0,),
            device="cpu",
        )
        write_json(self.prior / "resolved_config.json", asdict(config))
        with (
            patch.object(study_module, "implementation_identity", return_value=old),
            EmbeddingStudy(config) as study,
        ):
            with study.extract("train"):
                pass
            self.old_fingerprint = study.fingerprint
        self.config = replace(config, output_dir=str(self.target))

    def test_verified_reuse_keeps_array_bytes_and_old_provenance_unchanged(self):
        old_metadata = self.prior / "seed-7/features/train/metadata.json"
        metadata_hash = digest_file(old_metadata)
        report = reuse_development_caches(self.prior, self.config, self.snapshot)
        self.assertEqual(report["prior_fingerprint"], self.old_fingerprint)
        self.assertNotEqual(report["fingerprint"], self.old_fingerprint)
        self.assertEqual(digest_file(old_metadata), metadata_hash)
        source = self.prior / "seed-7/features/train/cnn.npy"
        target = self.target / "seed-7/features/train/cnn.npy"
        self.assertEqual(digest_file(target), digest_file(source))
        self.assertEqual(target.stat().st_ino, source.stat().st_ino)
        with EmbeddingStudy(self.config) as study:
            with (
                patch(
                    "hccr.tda.embedding.parallel.NeighborhoodExtractor.batches",
                    side_effect=AssertionError("TRAIN re-extracted"),
                ),
                study.extract("train") as cache,
            ):
                np.testing.assert_array_equal(cache["cnn"], study.train.features[:])
            with study.extract("validation") as cache:
                self.assertEqual(cache["ph"].shape, (12, 14))
        repeat = reuse_development_caches(self.prior, self.config, self.snapshot)
        self.assertEqual(repeat["fingerprint"], report["fingerprint"])

    def test_reuse_rejects_changed_scientific_config_and_corrupt_arrays(self):
        with self.assertRaisesRegex(ValueError, "scientific config"):
            reuse_development_caches(
                self.prior, replace(self.config, neighbors=4), self.snapshot
            )
        array = self.prior / "seed-7/features/train/cnn.npy"
        with array.open("r+b") as handle:
            handle.seek(-1, 2)
            previous = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([previous[0] ^ 1]))
        config = replace(self.config, output_dir=str(self.root / "corrupt"))
        with self.assertRaisesRegex(ValueError, "corrupt cache"):
            reuse_development_caches(self.prior, config, self.snapshot)

    def test_storage_guard_rejects_numerical_changes_and_wrong_snapshot(self):
        with self.assertRaisesRegex(ValueError, "pinned implementation"):
            verify_storage_patch(self.snapshot, "0" * 64)
        source = self.snapshot.read_text(encoding="utf-8").replace(
            "count * len(batch) / total", "count * len(batch) / (total + 1)"
        )
        self.snapshot.write_bytes(source.encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "not a storage-only change"):
            verify_storage_patch(
                self.snapshot, hashlib.sha256(source.encode()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
