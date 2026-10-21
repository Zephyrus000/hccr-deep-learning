"""Independent cached-embedding comparison with TRAIN-only references."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from math import ceil
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

from hccr.config.loader import load_yaml
from hccr.models.heads import AngularMarginClassifier
from hccr.tda.cubical import PH_FEATURE_NAMES, STATISTIC_NAMES
from hccr.tda.experiment import classification_metrics, head_logits, train_common_head
from hccr.tda.features import (
    AuxiliaryRows,
    CacheWriter,
    FittedBlocks,
    FusedRows,
    digest_file,
    implementation_identity,
    load_cache,
    signature,
    write_json,
)
from hccr.training.precision import PRECISION_NAMES

from .accelerated import SCREENING_PROTOCOL
from .neighborhood import (
    MAX_NEIGHBORS,
    RIPS_PROTOCOL,
    ReferenceBank,
    mix_knn_scores,
    normalize_rows,
    select_reference_indices,
)
from .parallel import NeighborhoodExtractor
from .source import CachedEmbeddings, EmbeddingView, id_digest

CONTROLS = ("cnn", "cnn_knn", "distance", "ph")


@dataclass(frozen=True)
class EmbeddingConfig:
    source_root: str
    output_dir: str = "artifacts/embedding-ph"
    seed: int = 7
    class_subset_seed: int = 7
    max_classes: int = 200
    bank_per_class: int = 32
    neighbors: int = 32
    query_batch_size: int = 32
    reference_batch_size: int = 2048
    search_backend: str = "reference"
    search_threads: int = 6
    ph_workers: int = 0
    ph_chunk_size: int = 16
    head_epochs: int = 8
    head_batch_size: int = 2048
    head_learning_rate: float = 0.001
    head_weight_decay: float = 0.0001
    logit_scale: float = 32.0
    angular_margin: float = 0.1
    alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    knn_mix_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    device: str = "auto"
    precision: str = "float32"

    def __post_init__(self):
        integers = (
            "max_classes",
            "bank_per_class",
            "neighbors",
            "query_batch_size",
            "reference_batch_size",
            "ph_chunk_size",
            "search_threads",
            "head_epochs",
            "head_batch_size",
        )
        for name in integers:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.ph_workers, int)
            or isinstance(self.ph_workers, bool)
            or self.ph_workers < 0
            or self.search_backend not in {"reference", "cpu_exact", "cuda_exact"}
        ):
            raise ValueError("invalid PH workers or search backend")
        if self.max_classes < 2 or not 2 <= self.neighbors <= MAX_NEIGHBORS:
            raise ValueError("need at least two classes and neighbors in [2,64]")
        for name in ("seed", "class_subset_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "head_learning_rate",
            "head_weight_decay",
            "logit_scale",
            "angular_margin",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            self.head_learning_rate == 0
            or self.logit_scale == 0
            or self.angular_margin >= 1
        ):
            raise ValueError(
                "learning rate/scale must be positive and CosFace margin < 1"
            )
        if not self.alpha_grid or any(
            not np.isfinite(a) or a <= 0 for a in self.alpha_grid
        ):
            raise ValueError("alpha_grid must contain positive finite values")
        if (
            not self.knn_mix_grid
            or 0 not in self.knn_mix_grid
            or any(not np.isfinite(a) or not 0 <= a <= 1 for a in self.knn_mix_grid)
        ):
            raise ValueError(
                "knn_mix_grid must include zero and use finite values in [0,1]"
            )
        if (
            self.precision not in PRECISION_NAMES
            or not self.source_root
            or not self.output_dir
        ):
            raise ValueError("invalid precision or empty source/output path")


def load_config(path: Path) -> EmbeddingConfig:
    raw = load_yaml(path)
    unknown = set(raw) - {field.name for field in fields(EmbeddingConfig)}
    if unknown:
        raise ValueError(f"unknown embedding-PH config keys: {sorted(unknown)}")
    for key in ("alpha_grid", "knn_mix_grid"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return EmbeddingConfig(**raw)


def _implementation() -> dict:
    return {
        "embedding_source": {
            path.name: digest_file(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "shared": implementation_identity(),
        "rips": RIPS_PROTOCOL,
        "ph_features": list(PH_FEATURE_NAMES),
        "distance_features": list(STATISTIC_NAMES),
        "search": "exhaustive_float64_unit_euclidean_within_train_bank_ties_sample_id",
        "accelerated_search": SCREENING_PROTOCOL,
        "self_exclusion": "exact_sample_id_not_zero_distance",
        "knn_vote": "uniform_reference_labels_probability_mix_no_temperature_search",
    }


@dataclass(frozen=True)
class _VerifiedSelection:
    report: dict


class EmbeddingStudy:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.source_root = Path(config.source_root).resolve()
        self.output = Path(config.output_dir).resolve() / f"seed-{config.seed}"
        if self.output.is_relative_to(
            self.source_root
        ) or self.source_root.is_relative_to(self.output):
            raise ValueError(
                "source caches and embedding-study outputs must be separate"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else str(torch.device(config.device))
        )
        self._resources = ExitStack()
        self._bank = None
        try:
            training = self._resources.enter_context(
                CachedEmbeddings(self.source_root, "train")
            )
            available = sorted(map(int, np.unique(training.targets)))
            if config.max_classes > len(available):
                raise ValueError("max_classes exceeds classes with TRAIN samples")
            self.original_classes = tuple(
                sorted(
                    Random(config.class_subset_seed).sample(
                        available, config.max_classes
                    )
                )
            )
            self.shared_source = training.shared_identity
            validation = self._resources.enter_context(
                CachedEmbeddings(self.source_root, "validation", self.shared_source)
            )
            self.train = EmbeddingView(training, self.original_classes)
            self.validation = EmbeddingView(validation, self.original_classes)
            self._development_ids = set(map(str, self.train.sample_ids))
            if self._development_ids.intersection(map(str, self.validation.sample_ids)):
                raise ValueError("sample-ID leakage between TRAIN and validation")
            self._development_ids.update(map(str, self.validation.sample_ids))
            inverse = {value: key for key, value in training.class_mapping.items()}
            self.class_mapping = {
                inverse[old]: new for new, old in enumerate(self.original_classes)
            }
            self.identity = {
                "schema_version": 1,
                "config": {
                    key: value
                    for key, value in asdict(config).items()
                    if key != "output_dir"
                },
                "train": self.train.identity,
                "validation": self.validation.identity,
                "class_mapping": self.class_mapping,
                "source_provenance": self.shared_source,
                "implementation": _implementation(),
            }
            self.fingerprint = signature(self.identity)
            self.output.mkdir(parents=True, exist_ok=True)
            identity_path = self.output / "study.json"
            if identity_path.exists():
                prior = json.loads(identity_path.read_text(encoding="utf-8"))
                if prior.get("fingerprint") != self.fingerprint:
                    raise ValueError(
                        "study/source/config changed; use a new output_dir"
                    )
            else:
                write_json(
                    identity_path,
                    {"fingerprint": self.fingerprint, "identity": self.identity},
                )
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def close(self):
        self._resources.close()

    def prepare(self) -> ReferenceBank:
        if self._bank is not None:
            return self._bank
        indices = select_reference_indices(
            self.train.targets,
            self.train.sample_ids,
            per_class=self.config.bank_per_class,
            seed=self.config.seed,
        )
        if len(indices) <= self.config.neighbors:
            raise ValueError(
                "TRAIN bank is too small for configured neighbors after self exclusion"
            )
        self.bank_identity = {
            "study": self.fingerprint,
            "fit_split": "train",
            "quota_policy": "up_to_quota_per_class_seeded_id_sha256",
            "sample_ids": id_digest(self.train.sample_ids[indices]),
            "samples": len(indices),
            "geometry": "l2_normalized_float64_original_320d",
        }
        path = self.output / "bank"
        if not path.exists():
            with CacheWriter(path, self.bank_identity, len(indices)) as writer:
                for start in range(0, len(indices), self.config.reference_batch_size):
                    chosen = indices[start : start + self.config.reference_batch_size]
                    writer.append(
                        {
                            "vectors": normalize_rows(self.train.features[chosen]),
                            "targets": self.train.targets[chosen],
                            "sample_ids": self.train.sample_ids[chosen],
                        }
                    )
        cache = self._resources.enter_context(load_cache(path, self.bank_identity))
        self._bank = ReferenceBank(
            cache["vectors"], cache["targets"], cache["sample_ids"]
        )
        return self._bank

    def _features(self, view: EmbeddingView):
        bank = self.prepare()
        identity = {
            "study": self.fingerprint,
            "source_view": view.identity,
            "bank": signature(self.bank_identity),
        }
        path = self.output / "features" / view.split
        if not path.exists():
            started = perf_counter()
            write_seconds, last_progress, processed = 0.0, started, 0
            with (
                NeighborhoodExtractor(
                    bank, self.output / "bank" / "vectors.npy", self.config
                ) as extractor,
                CacheWriter(path, identity, len(view.targets)) as writer,
            ):
                for batch in tqdm(
                    extractor.batches(view),
                    total=ceil(len(view.targets) / self.config.query_batch_size),
                    desc=f"embedding PH {view.split}",
                ):
                    before_write = perf_counter()
                    writer.append(batch.arrays)
                    if batch.inspection:
                        writer.add_inspection(batch.inspection)
                    write_seconds += perf_counter() - before_write
                    processed += len(batch.arrays["targets"])
                    if perf_counter() - last_progress >= 30:
                        last_progress = perf_counter()
                        write_json(
                            self.output / f"extraction_{view.split}_progress.json",
                            {
                                "samples_written": processed,
                                "total_samples": len(view.targets),
                                "elapsed_seconds": last_progress - started,
                                "timings": extractor.timings,
                                "write_seconds": write_seconds,
                            },
                        )
                timings = dict(extractor.timings)
                search_diagnostics = getattr(extractor.search, "diagnostics", {})
            write_json(
                self.output / f"extraction_{view.split}.json",
                {
                    "samples": len(view.targets),
                    "elapsed_seconds": perf_counter() - started,
                    "timing_scope": (
                        "cached_embedding_search_and_ph_excludes_backbone_"
                        "not_deployment_latency"
                    ),
                    "bank_size": len(bank.targets),
                    "neighbors": self.config.neighbors,
                    "search_backend": self.config.search_backend,
                    "ph_workers": self.config.ph_workers,
                    "timings": timings,
                    "write_seconds": write_seconds,
                    "search_diagnostics": search_diagnostics,
                    "timing_note": (
                        "search overlaps PH; worker times are summed wall times"
                    ),
                },
            )
        return load_cache(path, identity)

    def extract(self, split: str):
        if split not in {"train", "validation"}:
            raise ValueError(
                "extract only accepts development splits; use evaluate after selection"
            )
        return self._features(self.train if split == "train" else self.validation)

    def _load_head(self, chosen: dict):
        path = self.output / chosen["head"]
        if digest_file(path) != chosen["head_sha256"]:
            raise ValueError("selected head checkpoint changed")
        head = AngularMarginClassifier(
            334,
            len(self.class_mapping),
            "cosface",
            self.config.logit_scale,
            self.config.angular_margin,
        ).to(self.device)
        head.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        return head.eval()

    def _rows(self, cache, name: str, blocks, alpha):
        auxiliary = (
            AuxiliaryRows(cache, "cnn") if name in {"cnn", "cnn_knn"} else cache[name]
        )
        return FusedRows(cache["cnn"], auxiliary, blocks, alpha)

    def _metrics(self, head, cache, name, chosen):
        blocks = FittedBlocks.from_state(chosen["transforms"])
        values = self._rows(cache, name, blocks, chosen["alpha"])
        logits = head_logits(head, values, self.device, self.config.head_batch_size)
        if name == "cnn_knn":
            bank = self.prepare()

            def probabilities():
                offset = 0
                for batch in logits:
                    targets = bank.targets[
                        cache["neighbors"][offset : offset + len(batch)]
                    ]
                    offset += len(batch)
                    yield mix_knn_scores(batch, targets, chosen["mix"])

            return classification_metrics(probabilities(), cache["targets"])
        return classification_metrics(logits, cache["targets"])

    def compare(self) -> dict:
        if (self.output / "selection.json").exists():
            raise FileExistsError(
                "comparison already selected; use evaluate or a new output_dir"
            )
        self.prepare()
        with self.extract("train") as train, self.extract("validation") as validation:
            selections = {}
            for name in ("cnn", "distance", "ph"):
                auxiliary = (
                    AuxiliaryRows(train, "cnn") if name == "cnn" else train[name]
                )
                blocks = FittedBlocks.fit(train["cnn"], auxiliary)
                best, trials = None, []
                for trial, alpha in enumerate(
                    (0.0,) if name == "cnn" else self.config.alpha_grid
                ):
                    head, evidence = train_common_head(
                        self._rows(train, name, blocks, alpha),
                        train["targets"],
                        self._rows(validation, name, blocks, alpha),
                        validation["targets"],
                        len(self.class_mapping),
                        self.config,
                        self.config.logit_scale,
                        self.config.angular_margin,
                        self.device,
                    )
                    chosen = {"alpha": alpha, "transforms": blocks.state()}
                    metrics = self._metrics(head, validation, name, chosen)
                    trials.append({**evidence, "alpha": alpha, "validation": metrics})
                    if best is None or metrics["top1"] > best["validation"]["top1"]:
                        path = self.output / "heads" / f"{name}-trial-{trial}.pt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                key: value.detach().cpu()
                                for key, value in head.state_dict().items()
                            },
                            path,
                        )
                        best = {
                            **chosen,
                            "validation": metrics,
                            "selected_epoch": evidence["selected_epoch"],
                            "parameters": evidence["parameters"],
                            "head": str(path.relative_to(self.output)),
                            "head_sha256": digest_file(path),
                        }
                selections[name] = {**best, "trials": trials, "head_family": "cosface"}
            cnn = selections["cnn"]
            head = self._load_head(cnn)
            best, trials = None, []
            for mix in self.config.knn_mix_grid:
                chosen = {**cnn, "mix": mix}
                metrics = self._metrics(head, validation, "cnn_knn", chosen)
                trials.append({"mix": mix, "validation": metrics})
                if best is None or metrics["top1"] > best["validation"]["top1"]:
                    best = {**chosen, "validation": metrics}
            selections["cnn_knn"] = {
                **best,
                "trials": trials,
                "head_reused_from": "cnn",
            }
        counts = np.bincount(self.prepare().targets, minlength=len(self.class_mapping))
        report = {
            "schema_version": 1,
            "status": "all_variants_selected",
            "fingerprint": self.fingerprint,
            "bank_metadata_sha256": digest_file(self.output / "bank" / "metadata.json"),
            "development_caches": {
                split: digest_file(self.output / "features" / split / "metadata.json")
                for split in ("train", "validation")
            },
            "variants": {name: selections[name] for name in CONTROLS},
            "selection_rule": "validation_top1_first_best_epoch_alpha_and_knn_mix",
            "bank": {
                "fit_split": "train",
                "samples": int(counts.sum()),
                "class_support": counts.tolist(),
                "nearest_search_scope": "sampled_train_bank_not_all_train",
            },
            "search_budget": {
                "head_trials": 1 + 2 * len(self.config.alpha_grid),
                "knn_mixtures": len(self.config.knn_mix_grid),
            },
            "comparison_scope": "refitted_heads_on_identical_frozen_cached_embeddings",
            "writer_disjoint_verified": None,
            "historical_original_score_discrepancy": (
                "unresolved_not_used_as_accuracy_reference"
            ),
            "research_accuracy_gain_established": False,
        }
        report["selection_signature"] = signature(report)
        write_json(self.output / "selection.json", report)
        return report

    def _selection(self) -> _VerifiedSelection:
        path = self.output / "selection.json"
        if not path.is_file():
            raise ValueError("finish all four comparisons before accessing holdouts")
        report = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            key: value for key, value in report.items() if key != "selection_signature"
        }
        if (
            report.get("status") != "all_variants_selected"
            or set(report.get("variants", {})) != set(CONTROLS)
            or report.get("fingerprint") != self.fingerprint
            or report.get("selection_signature") != signature(payload)
        ):
            raise ValueError(
                "incomplete or incompatible selection; holdout access denied"
            )
        if report["bank_metadata_sha256"] != digest_file(
            self.output / "bank" / "metadata.json"
        ):
            raise ValueError("reference bank changed; holdout access denied")
        # Recheck array bytes even if this instance already holds the bank maps.
        self.prepare()
        with load_cache(self.output / "bank", self.bank_identity):
            pass
        for split, expected in report["development_caches"].items():
            if expected != digest_file(
                self.output / "features" / split / "metadata.json"
            ):
                raise ValueError("development cache changed; holdout access denied")
            view = self.train if split == "train" else self.validation
            with self._features(view):
                pass
        for chosen in report["variants"].values():
            if digest_file(self.output / chosen["head"]) != chosen["head_sha256"]:
                raise ValueError("selected head changed; holdout access denied")
        return _VerifiedSelection(report)

    def _holdout(self, split: str, selection: _VerifiedSelection):
        if selection.report["fingerprint"] != self.fingerprint:
            raise ValueError("selection does not belong to this study")
        return CachedEmbeddings(self.source_root, split, self.shared_source)

    def evaluate(self, splits: tuple[str, ...] = ("test",)) -> dict:
        selection = self._selection()
        if not splits or any(split not in {"test", "icdar"} for split in splits):
            raise ValueError("held-out splits must be test or icdar")
        reports = {}
        heldout_ids = set()
        for split in splits:
            with self._holdout(split, selection) as source:
                view = EmbeddingView(source, self.original_classes)
                if self._development_ids.intersection(map(str, view.sample_ids)):
                    raise ValueError(
                        "sample-ID leakage between development and holdout"
                    )
                ids = set(map(str, view.sample_ids))
                if heldout_ids.intersection(ids):
                    raise ValueError("sample-ID overlap between requested holdouts")
                heldout_ids.update(ids)
                with self._features(view) as cache:
                    reports[split] = {
                        name: self._metrics(
                            self._load_head(chosen), cache, name, chosen
                        )
                        for name, chosen in selection.report["variants"].items()
                    }
        report = {
            "fingerprint": self.fingerprint,
            "held_out": reports,
            "selection_unchanged": True,
            "classes": len(self.class_mapping),
            "seed": self.config.seed,
            "writer_disjoint_verified": None,
            "holdout_role": (
                "exploratory_source_holdout_may_have_been_previously_reported"
            ),
        }
        write_json(self.output / "heldout.json", report)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "extract", "compare", "run", "evaluate", "smoke")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/embedding_ph_seed7.yaml"),
    )
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument(
        "--output-dir", type=Path, help="empty output path for smoke only"
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "smoke":
        from .smoke import synthetic_smoke

        print(json.dumps(synthetic_smoke(arguments.output_dir), indent=2))
        return 0
    if arguments.output_dir is not None:
        parser.error(
            "--output-dir is smoke-only; set output_dir in the experiment YAML"
        )
    with EmbeddingStudy(load_config(arguments.config)) as study:
        if arguments.command == "prepare":
            study.prepare()
        elif arguments.command == "extract":
            for split in ("train", "validation"):
                with study.extract(split):
                    pass
        elif arguments.command in {"compare", "run"}:
            study.compare()
        else:
            study.evaluate(tuple(arguments.splits))
        print(
            json.dumps({"output_dir": str(study.output), "command": arguments.command})
        )
    return 0
