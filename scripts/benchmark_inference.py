"""Benchmark the exact deploy form across supported inference precisions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from hccr.models import EfficientHCCRNet, optimize_model_for_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-classes", type=int, default=7186)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--stage-depths", type=int, nargs=3, default=(1, 2, 2))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--precisions", nargs="+", choices=("float32", "float16"), default=("float32",)
    )
    parser.add_argument("--warmup-iterations", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output", type=Path)
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
    return {
        "latency_mean_ms": statistics.mean(timings),
        "latency_p50_ms": _percentile(timings, 50),
        "latency_p95_ms": _percentile(timings, 95),
        "latency_p99_ms": _percentile(timings, 99),
        "samples_per_second": 1000 / statistics.mean(timings),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if min(arguments.warmup_iterations, arguments.iterations) < 1:
        raise ValueError("warmup_iterations and iterations must be positive")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if arguments.device == "cpu" and "float16" in arguments.precisions:
        raise ValueError("float16 benchmarking requires CUDA")
    if arguments.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if arguments.device == "cpu":
        torch.set_num_threads(arguments.cpu_threads)
    model = EfficientHCCRNet(
        num_classes=arguments.num_classes,
        width=arguments.width,
        stage_depths=tuple(arguments.stage_depths),
    )
    if arguments.checkpoint is not None:
        model.load_state_dict(
            torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
        )
    model = model.to(arguments.device).eval()
    results = {}
    for precision in arguments.precisions:
        dtype = torch.float16 if precision == "float16" else torch.float32
        candidate = optimize_model_for_inference(model).to(dtype=dtype)
        results[precision] = benchmark(
            candidate,
            arguments.image_size,
            arguments.device,
            dtype,
            arguments.warmup_iterations,
            arguments.iterations,
        )
        if arguments.cuda_graph:
            if arguments.device != "cuda":
                raise ValueError("CUDA graph benchmarking requires CUDA")
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
        "device": arguments.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(arguments.device))
            if arguments.device == "cuda"
            else None
        ),
        "cpu_threads": torch.get_num_threads(),
        "batch_size": 1,
        "image_size": arguments.image_size,
        "num_classes": arguments.num_classes,
        "width": arguments.width,
        "stage_depths": list(arguments.stage_depths),
        "warmup_iterations": arguments.warmup_iterations,
        "iterations": arguments.iterations,
        "cuda_graph": arguments.cuda_graph,
        "deploy_transforms": [
            "cache_normalized_classifier_weight",
            "fold_conv_batch_norm",
            "remove_eval_dropout_hop",
            "sequential_feature_fast_path",
        ],
        "results": results,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


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


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
