"""Benchmark eager or deployment-optimized model inference."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from hccr.benchmarking import (
    BENCHMARK_SCHEMA_VERSION,
    build_benchmark_protocol,
    summarize_timings,
)
from hccr.models import MODEL_NAMES, build_model, optimize_model_for_inference

NAME = "benchmark"
HELP = "Benchmark eager or optimized inference across supported precisions."


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--mode",
        choices=("eager", "optimized"),
        default="eager",
        help="eager for fair main-table results; optimized for deployment studies",
    )
    parser.add_argument("--num-classes", type=int, default=7186)
    parser.add_argument("--model", choices=MODEL_NAMES, default="efficient_hccr")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--stage-depths", type=int, nargs=3, default=(1, 2, 2))
    parser.add_argument("--stem-stride", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--reparameterize-depthwise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--classification-head",
        choices=("softmax", "cosface", "arcface"),
        default="cosface",
    )
    parser.add_argument("--logit-scale", type=float, default=32.0)
    parser.add_argument("--angular-margin", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--precisions", nargs="+", choices=("float32", "float16"), default=("float32",)
    )
    parser.add_argument("--warmup-iterations", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser retained for the compatibility script."""
    parser = argparse.ArgumentParser(prog="benchmark_inference.py")
    configure_parser(parser)
    return parser


def benchmark(
    model: torch.nn.Module,
    image_size: int,
    device: str,
    dtype: torch.dtype,
    warmup_iterations: int,
    iterations: int,
) -> dict[str, float]:
    sample = torch.zeros(1, 1, image_size, image_size, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(sample)
        _synchronize(device)
        timings = []
        for _ in range(iterations):
            _synchronize(device)
            started_at = time.perf_counter()
            model(sample)
            _synchronize(device)
            timings.append((time.perf_counter() - started_at) * 1000)
    return summarize_timings(timings)


def run(arguments: argparse.Namespace) -> int:
    """Run a benchmark from already parsed top-level CLI arguments."""
    if min(arguments.warmup_iterations, arguments.iterations) < 1:
        raise ValueError("warmup_iterations and iterations must be positive")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if arguments.device == "cpu" and "float16" in arguments.precisions:
        raise ValueError("float16 benchmarking requires CUDA")
    if arguments.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if arguments.mode == "eager" and arguments.cuda_graph:
        raise ValueError("CUDA graphs require --mode optimized")
    if arguments.cuda_graph and arguments.device != "cuda":
        raise ValueError("CUDA graph benchmarking requires CUDA")
    if arguments.device == "cpu":
        torch.set_num_threads(arguments.cpu_threads)
    model = _build_benchmark_model(arguments)
    if arguments.checkpoint is not None:
        model.load_state_dict(
            torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
        )
    model = model.to(arguments.device).eval()
    optimization_transforms = (
        _deploy_transforms(model) if arguments.mode == "optimized" else []
    )
    results = {}
    for precision in arguments.precisions:
        dtype = torch.float16 if precision == "float16" else torch.float32
        candidate = _prepare_benchmark_model(model, arguments.mode, dtype)
        results[precision] = benchmark(
            candidate,
            arguments.image_size,
            arguments.device,
            dtype,
            arguments.warmup_iterations,
            arguments.iterations,
        )
        if arguments.cuda_graph:
            graph_runner = _capture_cuda_graph(candidate, arguments.image_size, dtype)
            results[f"{precision}_cuda_graph"] = benchmark(
                graph_runner,
                arguments.image_size,
                arguments.device,
                dtype,
                arguments.warmup_iterations,
                arguments.iterations,
            )
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_protocol": build_benchmark_protocol(
            device=arguments.device,
            batch_size=1,
            warmup_iterations=arguments.warmup_iterations,
            timed_iterations=arguments.iterations,
            repetitions=1,
            scope="model_forward",
            metadata={
                "mode": arguments.mode,
                "precisions": list(arguments.precisions),
                "cuda_graph": arguments.cuda_graph,
                "optimization_transforms": optimization_transforms,
                "intra_op_threads": torch.get_num_threads(),
                "inter_op_threads": torch.get_num_interop_threads(),
            },
        ),
        "device": arguments.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(arguments.device))
            if arguments.device == "cuda"
            else None
        ),
        "cpu_threads": torch.get_num_threads(),
        "batch_size": 1,
        "mode": arguments.mode,
        "image_size": arguments.image_size,
        "num_classes": arguments.num_classes,
        "model": arguments.model,
        "width": arguments.width,
        "embedding_dim": arguments.embedding_dim,
        "stage_depths": list(arguments.stage_depths),
        "stem_stride": arguments.stem_stride,
        "reparameterize_depthwise": arguments.reparameterize_depthwise,
        "classification_head": arguments.classification_head,
        "logit_scale": arguments.logit_scale,
        "angular_margin": arguments.angular_margin,
        "warmup_iterations": arguments.warmup_iterations,
        "iterations": arguments.iterations,
        "cuda_graph": arguments.cuda_graph,
        "shared_inference_behavior": _shared_inference_behavior(model),
        "optimization_transforms": optimization_transforms,
        "deploy_transforms": optimization_transforms,
        "results": results,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compatibility standalone benchmark entrypoint."""
    return run(build_parser().parse_args(argv))


def _build_benchmark_model(arguments: argparse.Namespace) -> torch.nn.Module:
    classifier_options = {
        "classification_head": arguments.classification_head,
        "embedding_dim": arguments.embedding_dim,
        "logit_scale": arguments.logit_scale,
        "angular_margin": arguments.angular_margin,
    }
    if arguments.model != "efficient_hccr":
        return build_model(
            arguments.model,
            num_classes=arguments.num_classes,
            **classifier_options,
        )
    return build_model(
        arguments.model,
        num_classes=arguments.num_classes,
        width=arguments.width,
        stage_depths=tuple(arguments.stage_depths),
        stem_stride=arguments.stem_stride,
        reparameterize_depthwise=arguments.reparameterize_depthwise,
        **classifier_options,
    )


def _prepare_benchmark_model(
    model: torch.nn.Module, mode: str, dtype: torch.dtype
) -> torch.nn.Module:
    """Prepare exactly the model form requested by the benchmark protocol."""
    if mode == "eager":
        # Calling eval after dtype conversion refreshes the inference-only
        # normalized-weight cache used by the shared angular-margin head.
        return model.to(dtype=dtype).eval()
    if mode == "optimized":
        return optimize_model_for_inference(model).to(dtype=dtype).eval()
    raise ValueError(f"unsupported benchmark mode: {mode}")


def _shared_inference_behavior(model: torch.nn.Module) -> list[str]:
    behavior = ["model_eval", "torch_inference_mode"]
    if getattr(model, "classification_head", None) in {"cosface", "arcface"}:
        behavior.append("cache_normalized_classifier_weight")
    return behavior


def _deploy_transforms(model: torch.nn.Module) -> list[str]:
    if getattr(model, "name", None) != "efficient_hccr":
        return ["freeze_eval_copy"]
    transforms = [
        "fold_conv_batch_norm",
        "fuse_depthwise_training_branches",
        "remove_eval_dropout_hop",
        "sequential_feature_fast_path",
    ]
    return transforms


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _capture_cuda_graph(model: torch.nn.Module, image_size: int, dtype: torch.dtype):
    static_input = torch.zeros(1, 1, image_size, image_size, device="cuda", dtype=dtype)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(3):
            model(static_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        static_output = model(static_input)

    def replay(inputs: torch.Tensor) -> torch.Tensor:
        static_input.copy_(inputs)
        graph.replay()
        return static_output

    return replay
