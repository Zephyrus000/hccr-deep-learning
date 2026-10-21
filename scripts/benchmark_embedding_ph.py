"""Paired TRAIN-only benchmark of the production neighborhood extraction path."""

from __future__ import annotations

import cProfile
import json
import pstats
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from hccr.tda.embedding.neighborhood import ReferenceBank
from hccr.tda.embedding.parallel import NeighborhoodExtractor
from hccr.tda.embedding.source import CachedEmbeddings, id_digest
from hccr.tda.features import CacheWriter, digest_file, load_cache, write_json


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(
    config, bank_root: Path, output: Path, *, queries=1024, trials=5, notify=print
):
    if queries < 1 or trials < 5:
        raise ValueError(
            "benchmark needs positive queries and at least five paired trials"
        )
    if (output / "benchmark.json").exists():
        raise FileExistsError("benchmark already exists; use a fresh output")
    output.mkdir(parents=True, exist_ok=True)
    bank_metadata = json.loads(
        (bank_root / "metadata.json").read_text(encoding="utf-8")
    )
    if bank_metadata["identity"].get("fit_split") != "train":
        raise ValueError("benchmark bank must be TRAIN-only")
    with ExitStack() as resources:
        cache = resources.enter_context(
            load_cache(bank_root, bank_metadata["identity"])
        )
        source = resources.enter_context(
            CachedEmbeddings(Path(config.source_root), "train")
        )
        if len(source.class_mapping) != config.max_classes:
            raise ValueError(
                "benchmark source must include the configured full class set"
            )
        bank = ReferenceBank(cache["vectors"], cache["targets"], cache["sample_ids"])
        counts = np.bincount(bank.targets, minlength=config.max_classes)
        if (
            len(counts) != config.max_classes
            or np.any(counts < 1)
            or np.any(counts > config.bank_per_class)
        ):
            raise ValueError(
                "benchmark bank does not cover the full configured class set"
            )
        old_study = json.loads(
            (bank_root.parent / "study.json").read_text(encoding="utf-8")
        )
        old_identity = old_study["identity"]
        if (
            bank_metadata["identity"]["study"] != old_study["fingerprint"]
            or old_identity["source_provenance"] != source.shared_identity
            or any(
                old_identity["config"][key] != getattr(config, key)
                for key in (
                    "seed",
                    "class_subset_seed",
                    "max_classes",
                    "bank_per_class",
                    "neighbors",
                )
            )
        ):
            raise ValueError("benchmark bank/source/scientific protocol mismatch")
        selected = np.random.default_rng(17).choice(
            len(source.targets), size=min(queries, len(source.targets)), replace=False
        )
        view = SimpleNamespace(
            features=np.asarray(source.features[selected]),
            targets=np.asarray(source.targets[selected]),
            indices=selected,
            sample_ids=np.array([source.sample_ids[int(i)] for i in selected]),
        )
        # This diagnostic profile is not one of the reported timing trials.
        notify("profiling_reference_search")
        profiler = cProfile.Profile()
        profiler.runcall(
            bank.neighbors, view.features[:32], view.sample_ids[:32], config.neighbors
        )
        profiler.dump_stats(str(output / "reference_search.pstats"))
        stats = pstats.Stats(profiler)
        profile = [
            {
                "function": str(key),
                "calls": value[1],
                "self_seconds": value[2],
                "cumulative_seconds": value[3],
            }
            for key, value in sorted(
                stats.stats.items(), key=lambda item: item[1][3], reverse=True
            )[:12]
        ]
        configs = {
            "reference": replace(
                config,
                search_backend="reference",
                ph_workers=0,
                query_batch_size=32,
                reference_batch_size=2048,
            ),
            "cpu_exact": replace(config, search_backend="cpu_exact"),
            "cuda_exact": replace(config, search_backend="cuda_exact"),
        }
        engines = {
            name: resources.enter_context(
                NeighborhoodExtractor(bank, bank_root / "vectors.npy", candidate)
            )
            for name, candidate in configs.items()
        }
        for name, engine in engines.items():
            notify(f"warmup_{name}")
            warm = (
                view
                if name != "reference"
                else SimpleNamespace(
                    **{key: values[:32] for key, values in vars(view).items()}
                )
            )
            list(engine.batches(warm))
        for engine in engines.values():
            engine.timings = {key: 0.0 for key in engine.timings}
            diagnostics = getattr(engine.search, "diagnostics", {})
            for key in ("queries", "reranked_candidates", "oracle_fallback_queries"):
                if key in diagnostics:
                    diagnostics[key] = 0
        samples, reference_hashes = {name: [] for name in engines}, None
        for repeat in range(trials):
            order = list(engines) if repeat % 2 == 0 else list(reversed(engines))
            for name in order:
                notify(f"trial_{repeat + 1}_{name}")
                target = output / f"trial-{repeat + 1}-{name}"
                _synchronize()
                started = perf_counter()
                with CacheWriter(
                    target, {"benchmark": name, "trial": repeat}, len(selected)
                ) as writer:
                    for batch in engines[name].batches(view):
                        writer.append(batch.arrays)
                        if batch.inspection:
                            writer.add_inspection(batch.inspection)
                _synchronize()
                elapsed = perf_counter() - started
                metadata = json.loads(
                    (target / "metadata.json").read_text(encoding="utf-8")
                )
                hashes = {
                    key: spec["sha256"] for key, spec in metadata["arrays"].items()
                }
                if reference_hashes is None:
                    reference_hashes = hashes
                elif hashes != reference_hashes:
                    raise ValueError(f"numerical/row-order parity failed in {name}")
                samples[name].append(elapsed)
        summary = {
            name: {
                "seconds": values,
                "median_seconds": median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "samples_per_second": len(selected) / median(values),
            }
            for name, values in samples.items()
        }
        chosen = min(
            ("cpu_exact", "cuda_exact"),
            key=lambda name: summary[name]["median_seconds"],
        )
        speedup = (
            summary["reference"]["median_seconds"] / summary[chosen]["median_seconds"]
        )
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "scope": (
                "TRAIN full-bank search + Rips + ordered cache writes and hash "
                "publication; not model training or deployment latency"
            ),
            "queries": len(selected),
            "paired_trials": trials,
            "alternating_order": True,
            "warmup_included": False,
            "full_bank_samples": len(bank.targets),
            "classes": config.max_classes,
            "neighbors": config.neighbors,
            "ordered_query_ids": id_digest(view.sample_ids),
            "source_metadata_sha256": source.pin["metadata_sha256"],
            "bank_metadata_sha256": digest_file(bank_root / "metadata.json"),
            "gpu": torch.cuda.get_device_name(0),
            "search_threads": config.search_threads,
            "ph_workers_per_optimized_engine": config.ph_workers,
            "pool_note": "two warm pools resident; only one engine executes each trial",
            "cache_warmth": "bank/source warmed before alternating timed trials",
            "summary": summary,
            "chosen_backend": chosen,
            "chosen_speedup_vs_reference": speedup,
            "exact_array_hash_parity": True,
            "reference_array_hashes": reference_hashes,
            "reference_profile": profile,
            "stage_timings": {name: engine.timings for name, engine in engines.items()},
            "search_diagnostics": {
                name: getattr(engine.search, "diagnostics", {})
                for name, engine in engines.items()
            },
            "torch_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "test_or_icdar_opened": False,
        }
        write_json(output / "benchmark.json", report)
        if speedup <= 1.05:
            raise RuntimeError(
                "no clear measured throughput gain; refusing a long restart"
            )
        return report
