"""Confirm a stem rounding mechanism on frozen development inputs only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_precision_forensics import (
    ROOT, torch, arithmetic, stage_arithmetic, state_digest, forward, statistics,
    load_model_from_run, reconcile_panel, read_json, write_json, digest,
    ort_scores, copy,
)


def separate_onnx_head_backbone(fp, quantized, head=False, backbone=False):
    """Bypass selected existing QDQ boundaries using exact original values."""
    graph = copy.deepcopy(quantized)
    producers = {v: node for node in graph.graph.node for v in node.output}
    original = {x.name: x for x in fp.graph.initializer}
    existing = {x.name for x in graph.graph.initializer}
    aliases, changes = {}, []
    for node in graph.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        producer = producers.get(node.input[0])
        if producer is not None and producer.op_type == "QuantizeLinear":
            source = producer.input[0]
            is_head = source.startswith("/classifier/")
            if (head and is_head) or (backbone and not is_head):
                aliases[node.output[0]] = source
                changes.append({"kind": "head_runtime" if is_head else "backbone_runtime", "source": source})
        elif backbone and node.input[0] in existing:
            name = node.name.removesuffix("_DequantizeLinear")
            assert name in original, name
            assert name.endswith("weight") and not name.startswith("classifier"), name
            if name not in existing:
                graph.graph.initializer.append(copy.deepcopy(original[name]))
                existing.add(name)
            aliases[node.output[0]] = name
            changes.append({"kind": "backbone_exact_weight", "source": name})
    for node in graph.graph.node:
        for index, source in enumerate(node.input):
            while source in aliases:
                source = aliases[source]
            node.input[index] = source
    return graph, changes


def stem_conditioning(model, images):
    from torch.nn import functional as F
    conv, bn = model.stem[0], model.stem[1]
    with torch.inference_mode():
        x = images[:32].cuda()
        fp = conv(x)
        rounded = fp.bfloat16().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf = conv(x).float()
        white = F.avg_pool2d((x == 1).float(), conv.kernel_size, conv.stride, conv.padding) == 1
        black = F.avg_pool2d((x == 0).float(), conv.kernel_size, conv.stride, conv.padding) == 1
        output = {"white_pixel_fraction": float((x == 1).float().mean()),
                  "black_pixel_fraction": float((x == 0).float().mean()), "channels": []}
        gain = bn.weight/(bn.running_var + bn.eps).sqrt()
        for index in range(fp.shape[1]):
            before, after = fp[:, index], rounded[:, index]
            delta = after-before
            record = {"channel": index, "bn_gain": float(gain[index]),
                "round_delta_mean": float(delta.mean()), "round_delta_abs_max": float(delta.abs().max()),
                "round_bn_output_shift_mean": float((delta*gain[index]).mean()),
                "round_bn_output_shift_abs_max": float((delta*gain[index]).abs().max()),
                "bf16_conv_vs_output_round_mean_abs": float((bf[:, index]-after).abs().mean())}
            for label, mask in (("white", white[:, 0]), ("black", black[:, 0])):
                if mask.any():
                    record[label+"_uniform_patch_fraction"] = float(mask.float().mean())
                    record[label+"_fp32_conv_mean"] = float(before[mask].mean())
                    record[label+"_rounded_conv_mean"] = float(after[mask].mean())
                    record[label+"_round_bn_shift_mean"] = float((delta[mask]*gain[index]).mean())
            output["channels"].append(record)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(0.35)
    base_protocol = read_json(args.base/"protocol.json")
    identity = hashlib.sha256(json.dumps(base_protocol, sort_keys=True).encode()).hexdigest()
    assert read_json(args.base/"audit.json")["status"] == "verified"
    protocol = {"script_sha256": digest(__file__), "helper_sha256": digest(ROOT/"scripts/run_precision_forensics.py"),
        "base_protocol_identity": identity, "threads": 2, "batch": 32, "optimizer_updates": 0,
        "bn_updates": 0, "test_images_read": 0, "icdar_images_read": 0,
        "scope": "diagnostic stem arithmetic only, not a selected deployment recipe"}
    write_json(args.output/"protocol.json", protocol)
    inventory = read_json(ROOT/"artifacts/local-post-training/20261026-retry-af8b8078/results/input-inventory.json")
    rows = [r for r in inventory["runs"] if r["group"] == "final_core"]
    state = {"status": "running", "started": time.time(), "completed": []}
    write_json(args.output/"status.json", state)
    panel_args = argparse.Namespace(downloads=ROOT/"artifacts/downloads/vastai-20261026", output=args.output,
        manifest=ROOT/"data/processed/hwdb1_0-1_1_official_five_splits/manifest.csv", panel=4096, calibration_panel=32)
    panels = reconcile_panel(panel_args, rows)
    images, targets = panels["validation"]["images"], panels["validation"]["targets"]
    indices = torch.tensor(read_json(args.base/"panel.json")["narrow_indices"])
    assert hashlib.sha256(images.numpy().tobytes()).hexdigest() == read_json(args.base/"panel.json")["image_tensor_sha256"]
    x, y = images[indices], targets[indices]
    for row in rows:
        name = f"seed-{row['seed']}-{row['variant']}"
        print("stem", name, flush=True)
        path = args.output/(name+".json")
        if path.exists():
            saved = read_json(path)
            if saved["protocol"] == protocol:
                state["completed"].append(name)
                continue
        model = load_model_from_run(Path(row["run"]), "cuda").float().eval()
        before = state_digest(model)
        reference, _ = forward(model, x, "cuda")
        bf16, _ = forward(model, x, "cuda", True, True)
        result = {"fp32": statistics(reference, y), "bf16": statistics(bf16, y, reference)}
        with arithmetic(model.stem[0], "conv_outputs_round_bf16"):
            scores, _ = forward(model, x, "cuda")
        result["only_stem_conv_output_round"] = statistics(scores, y, reference)
        with stage_arithmetic(model, {0}):
            scores, _ = forward(model, x, "cuda")
        result["only_stem_bf16"] = statistics(scores, y, reference)
        with arithmetic(model.stem[0], "conv_bf16"):
            scores, _ = forward(model, x, "cuda")
        result["only_stem_conv_bf16_bn_fp32"] = statistics(scores, y, reference)
        with stage_arithmetic(model, set(range(1, 1+len(model.features)))):
            scores, _ = forward(model, x, "cuda")
        result["bf16_backbone_except_fp32_stem"] = statistics(scores, y, reference)
        if row["variant"] == "dense_reference":
            for label, module, policy in (
                ("only_stem_weights_round", model.stem[0], "conv_weights_round_bf16"),
                ("only_stem_inputs_round", model.stem[0], "conv_inputs_round_bf16"),
                ("only_stem_bn_output_round", model.stem[1], "bn_output_round_bf16")):
                with arithmetic(module, policy):
                    scores, _ = forward(model, x, "cuda")
                result[label] = statistics(scores, y, reference)
            result["stem_conditioning_32"] = stem_conditioning(model, images)
            full_reference, _ = forward(model, images, "cuda")
            result["full_panel_fp32"] = statistics(full_reference, targets)
            with arithmetic(model.stem[0], "conv_outputs_round_bf16"):
                scores, _ = forward(model, images, "cuda")
            result["full_panel_only_stem_conv_output_round"] = statistics(scores, targets, full_reference)
            with stage_arithmetic(model, {0}):
                scores, _ = forward(model, images, "cuda")
            result["full_panel_only_stem_bf16"] = statistics(scores, targets, full_reference)
            model.cpu()
            with arithmetic(model.stem[0], "conv_outputs_round_bf16"):
                scores, _ = forward(model, x[:64], "cpu")
            result["cpu_64_only_stem_conv_output_round"] = statistics(scores, y[:64], reference[:64])
            import onnx
            onnx_dir = ROOT/"artifacts/local-post-training/20261026-onnx-repair-verified/results/onnx"
            fp_path, q_path = onnx_dir/(name+"-fp32.onnx"), onnx_dir/(name+"-mixed-int8.onnx")
            fp, quantized = onnx.load(fp_path), onnx.load(q_path)
            ox, oy = x[:64], y[:64]
            ort_reference = ort_scores(fp, ox, False)
            result["onnx_artifact_sha256"] = {str(fp_path): digest(fp_path), str(q_path): digest(q_path)}
            result["onnx_head_backbone"] = {}
            for label, head, backbone in (("unchanged_mixed_int8", False, False), ("backbone_only_int8", True, False), ("head_only_int8", False, True), ("all_qdq_bypassed", True, True)):
                graph, changes = separate_onnx_head_backbone(fp, quantized, head, backbone)
                for optimized in (False, True):
                    scores = ort_scores(graph, ox, optimized)
                    result["onnx_head_backbone"][label+"_optimization_"+str(optimized)] = {
                        "changes": changes, **statistics(scores, oy, ort_reference)}
        assert state_digest(model) == before
        result["state_unchanged"] = True
        assert digest(Path(row["run"])/"checkpoint.pt") == base_protocol["checkpoint_sha256"][row["run"]]
        write_json(path, {"protocol": protocol, "result": result})
        state["completed"].append(name)
        write_json(args.output/"status.json", state)
    state.update(status="completed", finished=time.time())
    write_json(args.output/"status.json", state)
    print("completed stem localization", flush=True)


if __name__ == "__main__":
    main()
