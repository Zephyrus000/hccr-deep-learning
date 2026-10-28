"""CPU dense cosine-head dynamic INT8 control; the backbone stays FP32.

The eager quantization API is version-dependent and deprecated by PyTorch.
Artifacts record its runtime and engine; loading requires compatible support.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic
from torch.nn import functional as F

from hccr.artifacts.checkpoints import file_digest
from hccr.data.dataset import HCCRDataset
from hccr.models.artifacts import (
    build_model_from_metadata,
    load_model_from_run,
    read_checkpoint_metadata,
)
from hccr.models.heads import AngularMarginClassifier
from hccr.preprocessing import EvalPreprocessor


class NormalizedDenseDeploymentHead(nn.Module):
    """Target-free cosine scores with cached normalized weights in a Linear."""

    def __init__(self, source: AngularMarginClassifier) -> None:
        super().__init__()
        if not isinstance(source, AngularMarginClassifier):
            raise ValueError("Only normalized dense angular heads are supported")
        weight = source.weight.detach().cpu().float()
        if not torch.isfinite(weight).all() or not math.isfinite(source.scale):
            raise ValueError("Source classifier contains nonfinite values")
        self.linear = nn.Linear(source.in_features, source.out_features, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(F.normalize(weight, dim=1))
        self.register_buffer("logit_scale", torch.tensor(float(source.scale)))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings.float(), dim=1)
        return self.linear(normalized).clamp(-1 + 1e-7, 1 - 1e-7) * self.logit_scale


@contextmanager
def cpu_quantization_context(threads: int, engine: str | None = None):
    """Restore both process-global settings even when conversion/timing fails."""
    if threads < 1:
        raise ValueError("threads must be positive")
    engine = engine or torch.backends.quantized.engine
    if engine == "none" or engine not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Unsupported CPU quantization engine: {engine}")
    previous = torch.get_num_threads(), torch.backends.quantized.engine
    try:
        torch.set_num_threads(threads)
        torch.backends.quantized.engine = engine
        yield engine
    finally:
        torch.backends.quantized.engine = previous[1]
        torch.set_num_threads(previous[0])


@torch.inference_mode(False)
def build_dense_head_pair(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Copy the model; replace only its dense head, then quantize only Linear."""
    if not isinstance(getattr(model, "classifier", None), AngularMarginClassifier):
        raise ValueError("Only normalized dense angular heads are supported")
    fp32 = copy.deepcopy(model).cpu().float().eval()
    fp32.classifier = NormalizedDenseDeploymentHead(fp32.classifier).eval()
    int8 = copy.deepcopy(fp32)
    int8.classifier = quantize_dynamic(
        int8.classifier,
        {nn.Linear: per_channel_dynamic_qconfig},
        dtype=torch.qint8,
        inplace=False,
    )
    if not isinstance(int8.classifier.linear, DynamicLinear):
        raise RuntimeError("Conversion did not produce a real dynamic INT8 Linear")
    return fp32, int8


def _backbone_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("classifier."):
            value = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode(False)
def load_classifier_deployment(path: Path) -> nn.Module:
    """Load a self-contained full-model artifact, including packed INT8 state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1 or payload["kind"] not in {"fp32", "int8"}:
        raise ValueError("Unsupported classifier deployment artifact")
    model = build_model_from_metadata(payload["metadata"]).cpu().float().eval()
    model.classifier = NormalizedDenseDeploymentHead(model.classifier).eval()
    with cpu_quantization_context(torch.get_num_threads(), payload["engine"]):
        if payload["kind"] == "int8":
            model.classifier = quantize_dynamic(
                model.classifier,
                {nn.Linear: per_channel_dynamic_qconfig},
                dtype=torch.qint8,
            )
        model.load_state_dict(payload["state_dict"])
    return model.eval()


def _difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Deployment produced nonfinite scores")
    return {
        "samples": len(reference),
        "max_absolute_score_drift": float((reference - candidate).abs().max()),
        "mean_absolute_score_drift": float((reference - candidate).abs().mean()),
        "top1_agreement": float(
            (reference.argmax(1) == candidate.argmax(1)).float().mean()
        ),
    }


def _batch_one_difference(reference, candidate, samples: torch.Tensor) -> dict:
    return _difference(
        torch.cat([reference(sample) for sample in samples.split(1)]),
        torch.cat([candidate(sample) for sample in samples.split(1)]),
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _timings(models: dict, inputs: list, warmup: int, iterations: int, blocks: int):
    results = {name: [] for name in models}
    for block in range(blocks):
        for name in list(models)[:: 1 if block % 2 == 0 else -1]:
            model = models[name]
            for index in range(warmup):
                model(inputs[index % len(inputs)])
            times = []
            for index in range(iterations):
                sample = inputs[index % len(inputs)]
                started = time.perf_counter_ns()
                scores = model(sample)
                times.append((time.perf_counter_ns() - started) / 1e6)
            if not torch.isfinite(scores).all():
                raise ValueError("Timed inference produced nonfinite scores")
            results[name].append(
                {"p50_ms": _percentile(times, 0.5), "p95_ms": _percentile(times, 0.95)}
            )
    return {
        name: {
            "blocks": rows,
            "median_p50_ms": statistics.median(row["p50_ms"] for row in rows),
            "median_p95_ms": statistics.median(row["p95_ms"] for row in rows),
            "p50_range_ms": [
                min(r["p50_ms"] for r in rows),
                max(r["p50_ms"] for r in rows),
            ],
        }
        for name, rows in results.items()
    }


def _validation(manifest: Path, source: Path, metadata: dict, models: dict, limit: int):
    if file_digest(manifest) != metadata.get("manifest_digest"):
        raise ValueError("Validation manifest differs from source checkpoint")
    labels = json.loads((source / "labels.json").read_text(encoding="utf-8"))
    if isinstance(labels, dict):
        labels = labels.get("labels")
        if isinstance(labels, dict):
            expected_keys = {str(i) for i in range(metadata["model"]["num_classes"])}
            if set(labels) != expected_keys:
                raise ValueError("Source label IDs must be consecutive classifier rows")
            labels = [labels[str(i)] for i in range(len(labels))]
    if not isinstance(labels, list) or len(labels) != metadata["model"]["num_classes"]:
        raise ValueError("Source labels do not match classifier dimensions")
    preprocess = metadata["preprocess"]
    dataset = HCCRDataset(
        manifest,
        "validation",
        EvalPreprocessor(preprocess["image_size"], preprocess["margin"]),
    )
    count, correct, agreement = 0, dict.fromkeys(models, 0), 0
    try:
        if not len(dataset):
            raise ValueError("Manifest contains no validation images")
        for row in dataset.rows:
            class_id = int(row["class_id"])
            if (
                not 0 <= class_id < len(labels)
                or row["unicode_label"] != labels[class_id]
            ):
                raise ValueError("Validation class ordering differs from source labels")
        for index in range(min(limit or len(dataset), len(dataset))):
            image, target, _ = dataset[index]
            predictions = {name: model(image[None]) for name, model in models.items()}
            for scores in predictions.values():
                if not torch.isfinite(scores).all():
                    raise ValueError("Validation produced nonfinite scores")
            picks = {
                name: int(scores.argmax(1)) for name, scores in predictions.items()
            }
            for name, pick in picks.items():
                correct[name] += pick == target
            agreement += picks["fp32"] == picks["int8"]
            count += 1
    finally:
        dataset.close()
    return {
        "split": "validation",
        "samples": count,
        "sample_limit": limit,
        "selection": "manifest_order",
        "manifest_digest": file_digest(manifest),
        "top1_accuracy": {name: value / count for name, value in correct.items()},
        "top1_agreement": agreement / count,
        "labels_digest": file_digest(source / "labels.json"),
        "preprocessing": preprocess,
    }


@torch.inference_mode(False)
def benchmark_classifier_quantization(
    source_run: Path,
    output_dir: Path,
    *,
    threads: int = 1,
    warmup: int = 20,
    iterations: int = 100,
    blocks: int = 5,
    samples: int = 64,
    seed: int = 7,
    engine: str | None = None,
    validation_manifest: Path | None = None,
    validation_samples: int = 0,
) -> dict:
    """Export/reload exact artifacts and measure paired synthetic CPU workloads.

    Dataset accuracy is omitted unless validation_manifest is explicitly supplied.
    Dynamic activation quantization needs no activation calibration images.
    """
    if min(iterations, blocks, samples) < 1 or warmup < 0 or validation_samples < 0:
        raise ValueError("Invalid benchmark iteration/sample counts")
    source_run, output_dir = Path(source_run).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite deployment output directory")
    metadata = read_checkpoint_metadata(source_run)
    source_hashes = {
        name: file_digest(source_run / name)
        for name in ("checkpoint.pt", "checkpoint_metadata.json")
    }
    if (source_run / "labels.json").is_file():
        source_hashes["labels.json"] = file_digest(source_run / "labels.json")
    original = load_model_from_run(source_run, device="cpu").float().eval()
    with cpu_quantization_context(threads, engine) as actual_engine, torch.no_grad():
        fp32, int8 = build_dense_head_pair(original)
        backbone_digest = _backbone_digest(original)
        if any(_backbone_digest(model) != backbone_digest for model in (fp32, int8)):
            raise RuntimeError("Head conversion changed backbone state")
        generator = torch.Generator().manual_seed(seed)
        size = int(metadata["preprocess"]["image_size"])
        images = torch.rand(min(samples, 8), 1, size, size, generator=generator)
        embeddings = torch.randn(
            samples, original.classifier.in_features, generator=generator
        )
        expected, actual = original(images), fp32(images)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        fidelity = _difference(expected, actual)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
        )
        artifact_info, restored = {}, {}
        for kind, model in (("fp32", fp32), ("int8", int8)):
            path = staging / f"{kind}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "kind": kind,
                    "engine": actual_engine,
                    "metadata": metadata,
                    "state_dict": model.state_dict(),
                },
                path,
            )
            head_path = staging / f"{kind}-head.pt"
            torch.save(model.classifier.state_dict(), head_path)
            restored[kind] = load_classifier_deployment(path)
            torch.testing.assert_close(
                restored[kind](images), model(images), rtol=0, atol=0
            )
            artifact_info[kind] = {
                "file": path.name,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
                "head_state_file": head_path.name,
                "head_state_sha256": file_digest(head_path),
                "head_state_bytes": head_path.stat().st_size,
                "extra_head_cache_bytes": 0,
                "backbone_digest": _backbone_digest(restored[kind]),
            }
        models = restored
        report = {
            "schema_version": 1,
            "scope": "dense_head_only_cpu_dynamic_ptq",
            "source_run": str(source_run),
            "source_hashes": source_hashes,
            "implementation_digest": file_digest(Path(__file__)),
            "runtime": {
                "torch": str(torch.__version__),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "engine": actual_engine,
                "threads": threads,
            },
            "quantization": {
                "api": "torch.ao.quantization.quantize_dynamic",
                "api_status": "deprecated_version_dependent",
                "module_type": type(models["int8"].classifier.linear).__module__
                + ".Linear",
                "operator": "quantized::linear_dynamic",
                "weights": "qint8_per_channel",
                "activations": "dynamic_per_tensor_uint8_internal_float32_input_output",
                "weight_qscheme": str(
                    models["int8"].classifier.linear.weight().qscheme()
                ),
                "quantized_modules": ["classifier.linear"],
                "fp32_exclusions": [
                    "backbone",
                    "embedding_projection",
                    "feature_normalization",
                    "score_clamp",
                    "logit_scale",
                ],
                "activation_calibration_data": None,
                "training_margin": "omitted_target_free",
                "prototype_policy": (
                    "normalize_FP32_once_then_quantize; no_post_quant_renormalization"
                ),
            },
            "artifacts": artifact_info,
            "original_fp32_export_fidelity": fidelity,
            "synthetic_head_score_drift": _batch_one_difference(
                models["fp32"].classifier,
                models["int8"].classifier,
                embeddings,
            ),
            "synthetic_image_score_drift": _batch_one_difference(
                models["fp32"], models["int8"], images
            ),
            "timing_protocol": {
                "batch_size": 1,
                "warmup_per_block": warmup,
                "iterations_per_block": iterations,
                "blocks": blocks,
                "order": "alternating_fp32_int8",
                "seed": seed,
                "inputs": "synthetic_random_images_and_embeddings",
                "preprocessing_included": False,
                "isolation_verified": False,
                "autograd": "no_grad",
            },
            "head_only_latency": _timings(
                {kind: model.classifier for kind, model in models.items()},
                list(embeddings.split(1)),
                warmup,
                iterations,
                blocks,
            ),
            "end_to_end_latency": _timings(
                models, list(images.split(1)), warmup, iterations, blocks
            ),
            "accuracy": None,
            "limitations": [
                "Synthetic agreement is not dataset accuracy or a speedup claim",
                "Not whole-model PTQ, QAT, ONNX, or factorized-head quantization",
                "Packed artifacts need a compatible PyTorch CPU quantization runtime",
            ],
        }
        if validation_manifest is not None:
            report["accuracy"] = _validation(
                Path(validation_manifest),
                source_run,
                metadata,
                models,
                validation_samples,
            )
        for name, digest in source_hashes.items():
            if file_digest(source_run / name) != digest:
                raise ValueError("Source checkpoint changed during benchmarking")
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    return report
