"""Frozen development-only arithmetic diagnostics. Never writes model state.

All arithmetic interventions operate on a separately loaded diagnostic instance
or an in-memory ONNX graph. The source checkpoints and deployment graphs remain
immutable. No calibration, optimization of parameters, or BN updates occur.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_local_post_training import reconcile_panel, read_json, digest
from hccr.models.artifacts import load_model_from_run


def write_json(path, payload):
    """Atomic receipt replacement, tolerating transient Windows reader locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            # This bounded local I/O retry runs only inside the background worker.
            time.sleep(0.025 * (attempt+1))


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        data = tensor.detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str(data.dtype).encode())
        value.update(str(tuple(data.shape)).encode())
        value.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def round_bf16(value):
    return value.to(torch.bfloat16).float()


@contextmanager
def arithmetic(model, policy):
    """Temporarily change arithmetic only, restoring original bound forwards."""
    changed = []
    for name, module in model.named_modules():
        original = module.forward
        replacement = None
        if isinstance(module, nn.Conv2d) and policy.startswith("conv_"):
            def replacement(x, m=module, old=original):
                with torch.autocast(x.device.type, enabled=False):
                    if policy == "conv_bf16":
                        with torch.autocast(x.device.type, dtype=torch.bfloat16):
                            return old(x).float()
                    if policy == "conv_weights_round_bf16":
                        weight, x = round_bf16(m.weight), x.float()
                    elif policy == "conv_inputs_round_bf16":
                        weight, x = m.weight, round_bf16(x)
                    else:
                        return round_bf16(old(x.float()))
                    return F.conv2d(x, weight, m.bias, m.stride, m.padding, m.dilation, m.groups)
        elif isinstance(module, nn.BatchNorm2d) and policy == "bn_float_in_bf16":
            def replacement(x, old=original):
                with torch.autocast(x.device.type, enabled=False):
                    return old(x.float())
        elif isinstance(module, nn.BatchNorm2d) and policy == "bn_output_round_bf16":
            def replacement(x, old=original):
                return round_bf16(old(x))
        if replacement is not None:
            changed.append((module, original))
            module.forward = replacement
    try:
        yield
    finally:
        for module, original in changed:
            module.forward = original


@contextmanager
def stage_arithmetic(model, bf16_stages):
    stages = [model.stem, *model.features]
    changed = []
    for index, module in enumerate(stages):
        original = module.forward
        def replacement(x, old=original, bf=index in bf16_stages):
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=bf):
                return old(x if bf else x.float())
        module.forward = replacement
        changed.append((module, original))
    try:
        yield
    finally:
        for module, original in changed:
            module.forward = original


def forward(model, images, device, backbone_bf16=False, head_bf16=False, batch=32):
    scores, embeddings = [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            x = images[start:start + batch].to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=backbone_bf16):
                z = model._forward_embedding(x)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=head_bf16):
                y = model.classifier(z if head_bf16 else z.float())
            if not torch.isfinite(y).all() or not torch.isfinite(z).all():
                raise ValueError("Nonfinite diagnostic activation or score")
            scores.append(y.float().cpu())
            embeddings.append(z.float().cpu())
    return torch.cat(scores), torch.cat(embeddings)


def statistics(scores, targets, reference=None, embedding=None):
    scores = torch.as_tensor(scores).float()
    pred = scores.argmax(1)
    top2 = scores.topk(2, dim=1).values
    result = {"n": len(targets), "top1": float(pred.eq(targets).double().mean()),
              "finite": bool(torch.isfinite(scores).all()),
              "score_abs_max": float(scores.abs().max()),
              "top2_margin_median": float((top2[:, 0] - top2[:, 1]).median())}
    if reference is not None:
        reference = torch.as_tensor(reference).float()
        result.update(prediction_agreement=float(pred.eq(reference.argmax(1)).double().mean()),
                      score_mean_abs_difference=float((scores-reference).abs().mean()),
                      score_max_abs_difference=float((scores-reference).abs().max()),
                      score_relative_l2_difference=float((scores-reference).norm()/reference.norm()))
    if embedding is not None:
        norms = embedding.norm(dim=1)
        result.update(embedding_norm_min=float(norms.min()), embedding_norm_median=float(norms.median()))
    return result


def bn_trace(model, images, device):
    results, handles, active = {}, [], ["fp32"]
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.BatchNorm2d):
            continue
        def hook(module, inputs, output, key=name):
            x, y = inputs[0].float(), output.float()
            mean, var = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False)
            std = (module.running_var + module.eps).sqrt()
            gain = module.weight.abs()/std
            entry = results.setdefault(key, {"running_var_min": float(module.running_var.min()),
                "running_var_median": float(module.running_var.median()), "bn_eps": module.eps,
                "effective_gain_max": float(gain.max()), "effective_gain_median": float(gain.median())})
            entry[active[0]] = {"input_dtype": str(inputs[0].dtype), "output_dtype": str(output.dtype),
                "input_mean_standardized_error_median": float(((mean-module.running_mean).abs()/std).median()),
                "input_mean_standardized_error_max": float(((mean-module.running_mean).abs()/std).max()),
                "input_var_over_running_var_median": float((var/(module.running_var+module.eps)).median()),
                "output_abs_max": float(y.abs().max()), "finite": bool(torch.isfinite(y).all())}
            if active[0] == "fp32":
                entry["reference_mean"] = mean.detach().cpu().tolist()
            else:
                ref = torch.tensor(entry.pop("reference_mean"), device=x.device)
                entry["mean_shift_in_running_std_max"] = float(((mean-ref).abs()/std).max())
        handles.append(layer.register_forward_hook(hook))
    try:
        for mode in ("fp32", "bf16"):
            active[0] = mode
            forward(model, images, device, mode == "bf16", mode == "bf16", len(images))
    finally:
        for handle in handles:
            handle.remove()
    return results


def bypass_qdq(model, fp32, activations=False, weights=False):
    """Bypass QDQ in a diagnostic graph without estimating any new scales."""
    model = copy.deepcopy(model)
    initializers = {x.name: x for x in model.graph.initializer}
    originals = {x.name: x for x in fp32.graph.initializer}
    producers = {out: node for node in model.graph.node for out in node.output}
    aliases, records = {}, []
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        producer = producers.get(node.input[0])
        if activations and producer is not None and producer.op_type == "QuantizeLinear":
            aliases[node.output[0]] = producer.input[0]
            records.append({"kind": "activation", "node": node.name})
        if weights and node.input[0] in initializers:
            name = node.name.removesuffix("_DequantizeLinear")
            if name not in originals:
                raise ValueError(f"No exact original initializer for {name}")
            if name not in initializers:
                model.graph.initializer.append(copy.deepcopy(originals[name]))
                initializers[name] = originals[name]
            aliases[node.output[0]] = name
            records.append({"kind": "weight", "node": node.name, "original_initializer": name})
    # Resolve cascaded aliases and leave unused nodes for the optimizer to prune.
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            while value in aliases:
                value = aliases[value]
            node.input[index] = value
    return model, records


def ort_scores(graph, images, optimized):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
    output = []
    for start in range(0, len(images), 32):
        output.append(session.run(None, {session.get_inputs()[0].name: images[start:start+32].numpy()})[0])
    return torch.from_numpy(np.concatenate(output))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--narrow", type=int, default=512)
    p.add_argument("--cpu-samples", type=int, default=64)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "pid": os.getpid(), "started": time.time(), "units": {}}
    write_json(args.output / "status.json", state)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for archived BF16 arithmetic comparison")
    torch.cuda.set_per_process_memory_fraction(0.35)
    downloads = ROOT / "artifacts/downloads/vastai-20261026"
    previous = ROOT / "artifacts/local-post-training/20261026-retry-af8b8078/results"
    onnx_dir = ROOT / "artifacts/local-post-training/20261026-onnx-repair-verified/results/onnx"
    rows = read_json(previous / "input-inventory.json")["runs"]
    rows = [r for r in rows if r["group"] == "final_core"]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "src/hccr").rglob("*.py"))}
    checkpoints = {r["run"]: digest(Path(r["run"])/"checkpoint.pt") for r in rows}
    protocol = {"script_sha256": digest(__file__), "source_hashes": sources, "checkpoint_sha256": checkpoints,
        "reconciliation_script_sha256": digest(ROOT/"scripts/run_local_post_training.py"),
        "python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
        "threads": 2, "gpu_memory_fraction": 0.35, "batch": 32, "narrow": args.narrow,
        "cpu_samples": args.cpu_samples, "deterministic": True, "tf32": False,
        "optimizer_updates": 0, "bn_updates": 0, "test_images_read": 0, "icdar_images_read": 0,
        "interventions": "diagnostic model copies only; no deployment artifact replacement",
        "head_dtype_note": "head_bf16 requests outer autocast; the published classifier explicitly disables autocast and computes FP32. Backbone/head exchanges therefore test the original FP32 head on changed embeddings.",
        "timing_claims": False}
    write_json(args.output/"protocol.json", protocol)
    def unit(name, action):
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        target = args.output / f"{name}.json"
        if target.exists() and read_json(target).get("protocol_identity") == identity:
            print("resume", name, flush=True)
            return read_json(target)["result"]
        print("start", name, flush=True)
        state["active_unit"] = name
        write_json(args.output/"status.json", state)
        try:
            result = action()
        except Exception as exc:
            state["units"][name] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            write_json(args.output/"status.json", state)
            raise
        write_json(target, {"protocol_identity": identity, "result": result})
        state["units"][name] = "completed"
        write_json(args.output/"status.json", state)
        print("completed", name, flush=True)
        return result
    panel_args = argparse.Namespace(downloads=downloads, output=args.output,
        manifest=ROOT/"data/processed/hwdb1_0-1_1_official_five_splits/manifest.csv",
        panel=4096, calibration_panel=32)
    panels = reconcile_panel(panel_args, rows)
    panel = panels["validation"]
    images, targets = panel["images"], panel["targets"]
    del panels
    narrow_indices = torch.linspace(0, len(images)-1, args.narrow).long()
    x, y = images[narrow_indices], targets[narrow_indices]
    write_json(args.output/"panel.json", {"samples": len(images), "writers": sorted(set(panel["writers"])),
        "sample_ids": panel["sample_ids"], "target_sha256": hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
        "image_tensor_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
        "narrow_indices": narrow_indices.tolist(), "narrow_selection": "evenly spaced within fixed 4096 prefix",
        "scope": "two-writer development diagnostic panel, not full validation"})
    for row in rows:
        name = f"seed-{row['seed']}-{row['variant']}"
        def core_action(row=row):
            model = load_model_from_run(Path(row["run"]), "cuda").float().eval()
            before = state_digest(model)
            reference, emb = forward(model, x, "cuda")
            result = {"fp32": statistics(reference, y, embedding=emb)}
            for bb, head, label in ((True, True, "bf16"), (True, False, "bf16_backbone_fp32_head"), (False, True, "fp32_backbone_bf16_head")):
                scores, z = forward(model, x, "cuda", bb, head)
                result[label] = statistics(scores, y, reference, z)
            result["state_unchanged"] = state_digest(model) == before
            if not result["state_unchanged"]:
                raise ValueError("Diagnostic mutated model state")
            return result
        unit("core-"+name, core_action)
    for row in [r for r in rows if r["variant"] == "dense_reference"]:
        name = f"seed-{row['seed']}-dense_reference"
        def dense_action(row=row):
            model = load_model_from_run(Path(row["run"]), "cuda").float().eval()
            before = state_digest(model)
            reference, _ = forward(model, images, "cuda")
            result = {"full_panel_fp32": statistics(reference, targets), "arithmetic": {}, "stage_sweeps": []}
            for label, bb, hd in (("full_panel_bf16", True, True), ("full_panel_bf16_backbone_fp32_head", True, False), ("full_panel_fp32_backbone_bf16_head", False, True)):
                scores, _ = forward(model, images, "cuda", bb, hd)
                result[label] = statistics(scores, targets, reference)
            ref = reference[narrow_indices]
            for policy in ("conv_bf16", "conv_weights_round_bf16", "conv_inputs_round_bf16", "conv_outputs_round_bf16", "bn_output_round_bf16", "bn_float_in_bf16"):
                with arithmetic(model, policy):
                    scores, z = forward(model, x, "cuda", policy == "bn_float_in_bf16")
                result["arithmetic"][policy] = statistics(scores, y, ref, z)
            # Both cumulative BF16 prefixes and one-stage substitutions retain
            # every checkpoint value and every BN running statistic.
            count = 1 + len(model.features)
            for kind in ("prefix", "single_stage"):
                for boundary in range(count):
                    selected = set(range(boundary+1)) if kind == "prefix" else {boundary}
                    with stage_arithmetic(model, selected):
                        scores, _ = forward(model, x, "cuda")
                    result["stage_sweeps"].append({"kind": kind, "bf16_stages": sorted(selected), **statistics(scores, y, ref)})
            result["bn_trace_32"] = bn_trace(model, images[:32], "cuda")
            cpu_x, cpu_y = images[:args.cpu_samples], targets[:args.cpu_samples]
            cuda_fp, _ = forward(model, cpu_x, "cuda")
            cuda_bf, _ = forward(model, cpu_x, "cuda", True, True)
            model.cpu()
            result["cross_device"] = {}
            for mode in (False, True):
                scores, _ = forward(model, cpu_x, "cpu", mode, mode)
                result["cross_device"]["bf16" if mode else "fp32"] = statistics(scores, cpu_y, cuda_bf if mode else cuda_fp)
            result["state_unchanged"] = state_digest(model) == before
            if not result["state_unchanged"]:
                raise ValueError("Diagnostic mutated model state")
            return result
        unit("dense-"+name, dense_action)
        def onnx_action(name=name):
            import onnx
            n = min(args.cpu_samples, len(x))
            ox, oy = x[:n], y[:n]
            fp_path, q_path = onnx_dir/(name+"-fp32.onnx"), onnx_dir/(name+"-mixed-int8.onnx")
            fp, q = onnx.load(fp_path), onnx.load(q_path)
            reference = ort_scores(fp, ox, False)
            result = {"n": n, "input_narrow_indices": narrow_indices[:n].tolist(),
                "artifact_sha256": {str(fp_path): digest(fp_path), str(q_path): digest(q_path)}, "observations": {}}
            for label, graph in (("fp32", fp), ("mixed_int8", q)):
                for optimized in (False, True):
                    scores = ort_scores(graph, ox, optimized)
                    result["observations"][f"{label}_optimization_{optimized}"] = statistics(scores, oy, reference)
            for activations, weights, label in ((True, False, "bypass_activation_qdq"), (False, True, "restore_exact_fp32_weights"), (True, True, "bypass_both")):
                graph, changes = bypass_qdq(q, fp, activations, weights)
                result[label+"_nodes"] = changes
                for optimized in (False, True):
                    scores = ort_scores(graph, ox, optimized)
                    result["observations"][f"{label}_optimization_{optimized}"] = statistics(scores, oy, reference)
            return result
        unit("onnx-"+name, onnx_action)
    changed = [str(path) for path, value in checkpoints.items() if digest(Path(path)/"checkpoint.pt") != value]
    changed += [name for name, value in sources.items() if digest(ROOT/name) != value]
    if changed:
        raise ValueError(f"Frozen inputs changed: {changed}")
    state.update(status="completed", finished=time.time(), active_unit=None,
        checkpoint_and_source_integrity="verified", optimizer_updates=0, bn_updates=0)
    write_json(args.output/"status.json", state)
    print("completed all diagnostics", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
