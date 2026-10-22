"""Experiment configuration and hardware requirements."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from hccr.config.loader import load_yaml


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    manifest: Path
    output_dir: Path
    seeds: tuple[int, ...]
    base_args: dict[str, Any]
    variants: tuple[ExperimentVariant, ...]
    profile_devices: tuple[str, ...] = ("cpu",)
    full_class_num_classes: int | None = 7186
    torchrun_nproc_per_node: int | None = None
    required_cuda_device_count: int | None = None
    required_cuda_device_name: str | None = None


@dataclass(frozen=True)
class ExperimentJob:
    key: str
    variant: str
    seed: int
    train_args: dict[str, Any]
    command: tuple[str, ...]


def load_experiment_spec(
    arguments: argparse.Namespace, project_root: Path
) -> ExperimentSpec:
    raw = load_yaml(arguments.config) if arguments.config else {}
    experiment_id = arguments.experiment_id or raw.get("experiment_id")
    if not experiment_id or any(character in experiment_id for character in "/\\"):
        raise ValueError("experiment_id must be a non-path name")
    manifest_value = arguments.manifest or raw.get(
        "manifest", "data/processed/casia_hwdb/manifest.csv"
    )
    output_value = arguments.output_dir or raw.get("output_dir", "experiments")
    seeds = tuple(arguments.seeds or raw.get("seeds", (7, 17, 29)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    base_args = dict(raw.get("base_args", {}))
    base_args.update(_parse_assignments(arguments.base_overrides))
    base_args.setdefault("evaluation_policy", "validation_only")
    raw_variants = list(raw.get("variants", []))
    raw_variants.extend(_parse_variant(value) for value in arguments.variant_overrides)
    if not raw_variants:
        raw_variants = [{"name": "default", "args": {}}]
    variants = tuple(
        ExperimentVariant(str(item["name"]), dict(item.get("args", {})))
        for item in raw_variants
    )
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("variant names must be unique")
    forbidden = {"manifest", "output_dir", "seed"}
    for source_name, values in [
        ("base_args", base_args),
        *((f"variant {variant.name}", variant.args) for variant in variants),
    ]:
        overlap = forbidden.intersection(values)
        if overlap:
            raise ValueError(
                f"{source_name} cannot override reserved arguments: {sorted(overlap)}"
            )
    profile_devices = tuple(
        arguments.profile_devices or raw.get("profile_devices", ("cpu",))
    )
    full_class_num_classes = (
        arguments.full_class_num_classes
        if arguments.full_class_num_classes is not None
        else raw.get("full_class_num_classes", 7186)
    )
    if full_class_num_classes is not None and full_class_num_classes < 2:
        raise ValueError("full_class_num_classes must be at least 2 or null")
    torchrun_nproc_per_node = (
        arguments.torchrun_nproc_per_node
        if arguments.torchrun_nproc_per_node is not None
        else raw.get("torchrun_nproc_per_node")
    )
    if torchrun_nproc_per_node is not None and torchrun_nproc_per_node < 2:
        raise ValueError("torchrun_nproc_per_node must be at least 2")
    if torchrun_nproc_per_node is not None and not base_args.get("distributed"):
        raise ValueError("torchrun_nproc_per_node requires base_args.distributed=true")
    disabled_variants = [
        variant.name
        for variant in variants
        if torchrun_nproc_per_node is not None
        and variant.args.get("distributed") is False
    ]
    if disabled_variants:
        raise ValueError(
            "torchrun_nproc_per_node cannot disable distributed training in "
            f"variants: {disabled_variants}"
        )
    required_cuda_device_count = raw.get("required_cuda_device_count")
    if required_cuda_device_count is not None:
        required_cuda_device_count = int(required_cuda_device_count)
        if required_cuda_device_count < 1:
            raise ValueError("required_cuda_device_count must be positive")
    required_cuda_device_name = raw.get("required_cuda_device_name")
    if required_cuda_device_name is not None:
        required_cuda_device_name = str(required_cuda_device_name).strip()
        if not required_cuda_device_name:
            raise ValueError("required_cuda_device_name must be non-empty")
        if required_cuda_device_count is None:
            raise ValueError(
                "required_cuda_device_name requires required_cuda_device_count"
            )
    if (
        torchrun_nproc_per_node is not None
        and required_cuda_device_count is not None
        and torchrun_nproc_per_node != required_cuda_device_count
    ):
        raise ValueError(
            "torchrun_nproc_per_node must match required_cuda_device_count"
        )
    return ExperimentSpec(
        experiment_id=experiment_id,
        manifest=_resolve_from_root(project_root, Path(manifest_value)),
        output_dir=_resolve_from_root(project_root, Path(output_value)),
        seeds=seeds,
        base_args=base_args,
        variants=variants,
        profile_devices=profile_devices,
        full_class_num_classes=full_class_num_classes,
        torchrun_nproc_per_node=torchrun_nproc_per_node,
        required_cuda_device_count=required_cuda_device_count,
        required_cuda_device_name=required_cuda_device_name,
    )


def _parse_assignments(values: list[str]) -> dict[str, Any]:
    import yaml

    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE: {value}")
        key, encoded = value.split("=", 1)
        result[key.replace("-", "_")] = yaml.safe_load(encoded)
    return result


def _parse_variant(value: str) -> dict[str, Any]:
    import yaml

    parsed = yaml.safe_load(value)
    if not isinstance(parsed, dict) or "name" not in parsed:
        raise ValueError("--variant must be a mapping with name and optional args")
    return parsed


def _resolve_from_root(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _jsonable_spec(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "manifest": str(spec.manifest),
        "output_dir": str(spec.output_dir),
        "seeds": list(spec.seeds),
        "base_args": spec.base_args,
        "variants": [asdict(variant) for variant in spec.variants],
        "profile_devices": list(spec.profile_devices),
        "full_class_num_classes": spec.full_class_num_classes,
        "torchrun_nproc_per_node": spec.torchrun_nproc_per_node,
        "required_cuda_device_count": spec.required_cuda_device_count,
        "required_cuda_device_name": spec.required_cuda_device_name,
    }


def _validate_hardware_requirements(spec: ExperimentSpec) -> None:
    required_count = spec.required_cuda_device_count
    if required_count is None:
        return
    available_count = torch.cuda.device_count()
    if available_count < required_count:
        raise RuntimeError(
            f"experiment requires {required_count} CUDA devices, "
            f"found {available_count}"
        )
    required_name = spec.required_cuda_device_name
    if required_name is None:
        return
    mismatches = [
        f"cuda:{index}={torch.cuda.get_device_name(index)}"
        for index in range(required_count)
        if required_name.casefold() not in torch.cuda.get_device_name(index).casefold()
    ]
    if mismatches:
        raise RuntimeError(
            f"experiment requires CUDA device name containing {required_name!r}; "
            f"mismatched devices: {', '.join(mismatches)}"
        )
