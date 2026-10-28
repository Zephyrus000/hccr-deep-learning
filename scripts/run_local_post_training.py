"""Run the frozen HCCR post-training analysis locally without optimizer updates.

Outputs are separate from downloaded evidence. Every stage has a durable status;
image analyses require reconciliation against the recorded validation panel.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from hccr.evaluation.post_training_analysis import (
    overlap_statistics, paired_writer_interval, prediction_statistics,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_identity(row):
    return f"seed-{row['seed']}-{row['variant']}"


def inventory(args):
    rows = csv_rows(args.downloads / "consolidated/heldout-results.csv")
    if len(rows) != 78 or len({r["fingerprint"] for r in rows}) != 78:
        raise ValueError("Expected 78 unique completed baseline/final checkpoints")
    audit = read_json(args.downloads / "consolidated/heldout-coverage-audit.json")
    if audit["status"] != "verified":
        raise ValueError("Local download coverage has not been verified")
    for row in rows:
        run = Path(row["training_metrics"]).parent
        if digest(run / "checkpoint.pt") != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {run}")
        row["run"] = str(run)
        row["metadata"] = read_json(run / "checkpoint_metadata.json")
    code_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(code_root)): digest(p)
                     for p in sorted((code_root / "src/hccr").rglob("*.py"))}
    write_json(args.output / "input-inventory.json", {
        "runs": rows, "script_sha256": digest(__file__),
        "research_source_hashes": source_hashes,
        "analysis_sha256": digest(Path(__file__).resolve().parents[1] / "src/hccr/evaluation/post_training_analysis.py"),
        "training": False, "optimizer_updates": 0, "mask_reselection": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return rows


def analyze_predictions(args, rows):
    reports, comparisons = [], []
    primary = args.downloads / "57418784/primary/repo/experiments/classifier-study"
    h1 = {s: read_json(primary / f"seed-{s}/h1-validation.json") for s in (7, 17, 29)}
    write_json(args.output / "h1-existing-and-mask-stability.json", {
        "h1_reports": {str(s): str(primary / f"seed-{s}/h1-validation.json") for s in h1},
        "mask_overlap": overlap_statistics({s: d["selected_class_ids"] for s, d in h1.items()}),
        "no_new_H_selection": True,
        "h1_validation_runtime_note": "Archived H1 runtime precision is retained as recorded; no silent BF16 substitution.",
    })
    for split in ("test", "icdar"):
        # The same ordered endpoint is shared across all frozen panels.
        index_path = Path(rows[0][split + "_metrics"]).parents[2] / f"{split}-index-gpu-0.csv"
        index = csv_rows(index_path)
        ids = [r["sample_id"] for r in index]
        identity = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        labels = np.array([int(r["class_id"]) for r in index])
        writers = np.array([r["writer_id"] for r in index])
        versions = np.array([r["source_version"] for r in index])
        if len(set(ids)) != len(ids):
            raise ValueError("Endpoint index has duplicate sample identities")
        correct = {}
        for number, row in enumerate(rows):
            path = Path(row[split + "_metrics"])
            saved = read_json(path)
            predictions_path = path.with_name("predictions.npz")
            if (saved["sample_identity_sha256"] != identity
                    or digest(predictions_path) != saved["predictions_sha256"]
                    or saved["checkpoint_sha256"] != row["checkpoint_sha256"]):
                raise ValueError("Prediction provenance mismatch")
            with np.load(predictions_path, allow_pickle=False) as data:
                if not np.array_equal(data["targets"], labels):
                    raise ValueError("Predictions do not align with endpoint index")
                selected = row["metadata"]["model"].get("residual_class_indices", [])
                metrics = prediction_statistics(labels, data["top5"], selected)
                correct[(row["variant"], int(row["seed"]))] = data["top5"][:, 0] == labels
            for key in ("top1", "top5"):
                if abs(metrics[key] - saved[key]) > 1e-12:
                    raise ValueError("Recomputed metric differs from archived inference")
            record = {"seed": int(row["seed"]), "variant": row["variant"],
                      "group": row["group"], "split": split,
                      "checkpoint_sha256": row["checkpoint_sha256"], **metrics}
            write_json(args.output / "prediction-analysis" / split / (run_identity(row) + ".json"), record)
            reports.append({k: v for k, v in record.items() if not k.startswith("per_class") and k != "top_confusions_exploratory"})
            if number % 10 == 0:
                print(f"prediction-analysis {split} {number + 1}/{len(rows)}", flush=True)
        pairs = [
            ("static_residual", "uniform_low_rank", "primary"),
            ("dynamic_residual", "static_residual", "primary"),
            ("static_residual", "uniform_near_budget", "primary"),
            ("dynamic_residual", "dense_reference", "primary"),
        ]
        for v in sorted({r["variant"] for r in rows if r["group"] == "final_ablation"}):
            if v.startswith("static_residual__"):
                pairs.append(("static_residual", v, "exploratory_allocation_control"))
            else:
                pairs.append(("dynamic_residual", v, "exploratory_gate_control"))
        for a, b, role in pairs:
            seed_differences = []
            for seed in (7, 17, 29):
                delta = correct[(a, seed)].astype(float) - correct[(b, seed)]
                seed_differences.append(delta)
                result = {"split": split, "candidate": a, "reference": b,
                          "seed": seed, "comparison_role": role}
                if split == "test":
                    result.update(paired_writer_interval(delta, writers, versions,
                                  repetitions=args.bootstrap, seed=20261026 + seed))
                else:
                    result.update(difference_pp=float(delta.mean() * 100), ci95_pp=None,
                                  interval_status="not_computed_writer_identity_reconciliation_unverified")
                comparisons.append(result)
            # Average paired outcomes over the three seeds, then resample writers
            # once. This is conditional on these seeds, not a joint seed/writer CI.
            average = np.mean(seed_differences, axis=0)
            combined = {"split": split, "candidate": a, "reference": b, "seed": "mean_of_3",
                        "comparison_role": role, "seed_uncertainty_in_interval": False}
            if split == "test":
                combined.update(paired_writer_interval(average, writers, versions, repetitions=args.bootstrap))
            else:
                combined.update(difference_pp=float(average.mean() * 100), ci95_pp=None,
                                interval_status="not_computed_writer_identity_reconciliation_unverified")
            comparisons.append(combined)
        del correct, index
    write_json(args.output / "paired-comparisons.json", {
        "comparisons": comparisons, "no_multiplicity_correction": True,
        "primary_means_planned_comparison_family_not_external_preregistration": True,
        "selection_from_test": False, "new_test_image_inference": False,
    })
    write_json(args.output / "prediction-summary.json", reports)
    write_csv(args.output / "paired-comparisons.csv", comparisons)


def analyze_telemetry(args, rows):
    scalars = []
    for row in rows:
        if not row["group"].startswith("final"):
            continue
        curves = read_json(Path(row["run"]) / "curves.json")
        for epoch in curves["epochs"]:
            for key, value in epoch.items():
                if key in {"gate_validation", "stages"} and isinstance(value, dict):
                    stack = [(key, value)]
                    while stack:
                        prefix, observation = stack.pop()
                        for name, item in observation.items():
                            if isinstance(item, dict):
                                stack.append((prefix + "." + name, item))
                            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                                scalars.append({"seed": row["seed"], "variant": row["variant"],
                                                "epoch": epoch["epoch"], "metric": prefix + "." + name,
                                                "value": item})
    write_csv(args.output / "gate-training-trajectories.csv", scalars)
    write_json(args.output / "telemetry-scope.json", {
        "scalar_observations": len(scalars),
        "source": "archived curves; total gradients retain their original attribution",
        "classification_only_historical_gradients": "cannot_reconstruct_without_historical_checkpoints",
        "missing_initialization_telemetry": "not_recreated_as_historical_observations",
    })
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = ["gate_validation.target.within_class_variance",
               "gate_validation.target.fraction_below_005",
               "gate_validation.target.fraction_above_095",
               "gate_validation.target.normalized_sigmoid_slope"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, targets, strict=True):
        for seed in ("7", "17", "29"):
            selected = [r for r in scalars if r["variant"] == "dynamic_residual"
                        and str(r["seed"]) == seed and r["metric"] == metric]
            selected.sort(key=lambda r: r["epoch"])
            ax.plot([r["epoch"] for r in selected], [r["value"] for r in selected], label=f"Seed {seed}")
        ax.set_title(metric.rsplit(".", 1)[-1]); ax.set_xlabel("Epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(args.output / "gate-trajectories.png", dpi=180); plt.close(fig)


def reconcile_panel(args, rows):
    import torch
    from hccr.data.lmdb_store import LMDBImageStore
    from hccr.preprocessing import EvalPreprocessor
    from torchvision.transforms.v2 import functional as tvf
    primary = args.downloads / "57418784/primary/repo/experiments/classifier-study"
    h1 = read_json(primary / "seed-7/h1-validation.json")
    local_labels = read_json(args.manifest.with_name("labels.json"))
    if local_labels != read_json(Path(rows[0]["run"]) / "labels.json"):
        raise ValueError("Local label mapping differs from training")
    # Prediction JSONL is sorted by sample ID, not original manifest order.
    exported = []
    with (primary / "seed-7/dense.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if "sample_id" in value:
                exported.append(value)
            if len(exported) == args.calibration_panel:
                break
    wanted_calibration = {r["sample_id"] for r in exported}
    calibration_by_id = {}
    validation, calibration, fit = [], [], []
    def cloud_row(local):
        value = dict(local)
        # The original uploaded ZIP has merged train/test directories. The
        # metadata-derived transformation must match the complete frozen H1 hash.
        relative = Path(local["source_file"])
        if local["source_version"] in {"HWDB1.0", "HWDB1.1"}:
            partition = "Train" if local["official_partition"] == "train" else "Test"
            value["source_file"] = f"HWDB1.0-1.1_{partition}/{relative.name}"
            value["sample_id"] = hashlib.sha256(
                (value["source_file"] + "\0" + local["record_offset"]).encode()).hexdigest()
        return value
    writer_sets = {k: set() for k in ("train", "validation", "calibration")}
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in writer_sets:
                continue
            writer_sets[split].add(row["writer_id"])
            if row["unicode_label"] != local_labels["labels"][row["class_id"]]:
                raise ValueError("Manifest label differs from frozen class mapping")
            if split == "validation":
                validation.append(row)
            elif split == "calibration":
                mapped_id = cloud_row(row)["sample_id"]
                if mapped_id in wanted_calibration:
                    calibration_by_id[mapped_id] = row
            elif split == "train" and len(fit) < 64:
                fit.append(row)
    if (writer_sets["validation"] != set(h1["validation_writer_ids"])
            or writer_sets["calibration"] != set(h1["calibration_writer_ids"])
            or writer_sets["train"] & writer_sets["validation"]
            or writer_sets["train"] & writer_sets["calibration"]
            or writer_sets["validation"] & writer_sets["calibration"]):
        raise ValueError("Local development writer allocation differs from frozen study")
    if len(validation) != 209395:
        raise ValueError("Unexpected validation support")
    if args.panel != 4096:
        raise ValueError("The recorded validation gate panel has exactly 4096 rows")
    candidates = [(validation, validation), (validation, [cloud_row(r) for r in validation])]
    paired = sorted(zip(validation, candidates[-1][1], strict=True),
                    key=lambda pair: (pair[1]["source_file"], int(pair[1]["record_offset"])))
    candidates.append(([p[0] for p in paired], [p[1] for p in paired]))
    chosen = None
    for local_view, frozen_view in candidates:
        endpoint_hash = hashlib.sha256(json.dumps(frozen_view, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if endpoint_hash == h1["validation_rows_digest"]:
            chosen = (local_view, frozen_view, endpoint_hash)
            break
    if chosen is None:
        raise ValueError("Cannot reconstruct the full frozen validation-row digest from local source metadata")
    validation, frozen_validation, endpoint_hash = chosen
    validation = validation[:args.panel]
    frozen_ids = [r["sample_id"] for r in frozen_validation[:args.panel]]
    identity = hashlib.sha256(json.dumps(frozen_ids).encode()).hexdigest()
    recorded = set()
    for row in rows:
        if row["group"] == "final_core" and row["variant"] == "dynamic_residual":
            selected = read_json(row["training_metrics"])["validation"]["selected"]
            recorded.add(selected["gate_validation"]["panel_identity_sha256"])
    positional_hash = hashlib.sha256(json.dumps([str(i) for i in range(args.panel)]).encode()).hexdigest()
    if recorded not in ({identity}, {positional_hash}):
        raise ValueError("Recorded telemetry panel is neither the reconciled image identities nor loader-prefix indices")
    if set(calibration_by_id) != wanted_calibration:
        raise ValueError("Local calibration samples differ from archived calibration predictions")
    calibration = [calibration_by_id[r["sample_id"]] for r in exported]
    if any(int(a["class_id"]) != int(b["class_id"]) or a["writer_id"] != b["writer_id"]
           for a, b in zip(calibration, exported, strict=True)):
        raise ValueError("Calibration target/writer identity mismatch")
    local_hash = digest(args.manifest)
    store = LMDBImageStore(args.manifest.with_name("images.lmdb"), local_hash)
    transform = EvalPreprocessor(96, 4)
    panels = {}
    try:
        _ = store.metadata
        for name, selected in (("validation", validation), ("calibration", calibration), ("fit", fit)):
            images = []
            for index, row in enumerate(selected):
                image = store.read_image(row["sample_id"])
                # The local native-image hash uses the same dimensions+pixels
                # protocol as the manifest builder, verified below by its helper.
                native_hash = hashlib.sha256(struct.pack("<HH", image.width, image.height))
                native_hash.update(image.tobytes())
                if native_hash.hexdigest() != row["pixel_sha256"]:
                    raise ValueError("LMDB pixels differ from the local native source manifest")
                image = tvf.to_dtype(tvf.to_image(transform(image)), torch.float32, scale=True)
                images.append(image)
                if index and index % 512 == 0:
                    print(f"prepare {name} {index}/{len(selected)}", flush=True)
            panels[name] = {"images": torch.stack(images),
                            "targets": torch.tensor([int(r["class_id"]) for r in selected]),
                            "writers": [r["writer_id"] for r in selected],
                            "versions": [r["source_version"] for r in selected],
                            "sample_ids": [r["sample_id"] for r in selected]}
    finally:
        store.close()
    write_json(args.output / "local-data-reconciliation.json", {
        "status": "verified_analysis_panels", "local_manifest_sha256": local_hash,
        "training_manifest_sha256": rows[0]["metadata"]["manifest_digest"],
        "manifest_files_identical": local_hash == rows[0]["metadata"]["manifest_digest"],
        "validation_panel_identity_sha256": identity,
        "frozen_validation_rows_digest_verified": endpoint_hash,
        "historical_gate_panel_hash_scope": "loader_indices_only" if recorded == {positional_hash} else "sample_ids",
        "basis": ["semantic class mapping", "validation/calibration writer sets and disjoint fit writers",
                  "complete frozen H1 validation rows digest", "archived calibration prediction prefix",
                  "local manifest-bound LMDB and per-image native pixel hashes"],
        "scope": "analysis panels only, not a byte-identical full cloud dataset claim",
        "test_images_read": 0, "icdar_images_read": 0,
    })
    return panels


def forward_panel(model, images, device, precision, batch):
    import torch
    result, embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=precision == "bfloat16"):
                x = model._forward_embedding(images[start:start + batch].to(device))
                scores = model.classifier(x)
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite logits in frozen checkpoint")
            result.append(scores.cpu()); embeddings.append(x.float().cpu())
    return torch.cat(result), torch.cat(embeddings)


def explicit_head(head, embeddings, alpha=None):
    """Direct prototype reference; temporary tensors are not deployment caches."""
    import torch
    from torch.nn import functional as F
    z = F.normalize(embeddings.float(), dim=1, eps=head.normalization_eps)
    q = F.normalize(head.coefficients.float() @ head.basis.float(), dim=1,
                    eps=head.normalization_eps)
    scores = z @ q.T
    if head.residual_indices.numel():
        norms = (head.coefficients.float() @ head.basis.float()).norm(dim=1).clamp_min(head.normalization_eps)
        gates = head.gate_values(z, scores, norms) if alpha is None else alpha
        vectors = q[head.residual_indices][None] + gates[..., None] * head.residuals[None].float()
        selected = (z[:, None] * F.normalize(vectors, dim=2, eps=head.normalization_eps)).sum(2)
        scores = scores.index_copy(1, head.residual_indices, selected)
    return scores.clamp(-1 + 1e-7, 1 - 1e-7) * head.scale


def batch_norm_precision_trace(model, images, device):
    import torch
    reference, comparisons = {}, []
    mode = "float32"
    def observe(name, layer):
        def hook(module, inputs, output):
            value = output.detach().float().flatten()
            stride = max(1, value.numel() // 4096)
            sampled = value[::stride][:4096].cpu()
            if mode == "float32":
                reference[name] = sampled
            else:
                previous = reference[name]
                comparisons.append({"layer": name,
                    "relative_sample_l2_difference": float((sampled - previous).norm() / previous.norm().clamp_min(1e-8)),
                    "sample_max_absolute_difference": float((sampled - previous).abs().max()),
                    "running_variance_min": float(layer.running_var.min()),
                    "running_variance_median": float(layer.running_var.median()),
                    "running_mean_max_abs": float(layer.running_mean.abs().max())})
        return hook
    handles = [layer.register_forward_hook(observe(name, layer))
               for name, layer in model.named_modules() if isinstance(layer, torch.nn.BatchNorm2d)]
    try:
        for mode in ("float32", "bfloat16"):
            forward_panel(model, images[:8], device, mode, 8)
    finally:
        for handle in handles:
            handle.remove()
    return {"samples": 8, "deterministic_strided_activation_samples_per_layer": 4096,
            "layers": comparisons,
            "interpretation": "localization only; activation divergence does not prove a root cause"}


def analyze_models(args, rows, panels):
    import torch
    from torch.nn import functional as F
    from hccr.models.artifacts import load_model_from_run
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    panel = panels["validation"]
    selected = [r for r in rows if r["group"] == "final_core"]
    for row in selected:
        print("model-analysis", run_identity(row), flush=True)
        model = load_model_from_run(Path(row["run"]), device).float().eval()
        before = digest(Path(row["run"]) / "checkpoint.pt")
        fp32, _ = forward_panel(model, panel["images"], device, "float32", args.batch)
        bf16, emb = forward_panel(model, panel["images"], device, "bfloat16", args.batch)
        y = panel["targets"]
        report = {"variant": row["variant"], "seed": int(row["seed"]), "device": device,
                  "samples": len(y), "panel": "recorded_validation_4096",
                  "fp32_top1": float(fp32.argmax(1).eq(y).float().mean()),
                  "bf16_top1": float(bf16.argmax(1).eq(y).float().mean()),
                  "prediction_agreement": float(fp32.argmax(1).eq(bf16.argmax(1)).float().mean()),
                  "mean_absolute_logit_difference": float((fp32 - bf16).abs().mean()),
                  "max_absolute_logit_difference": float((fp32 - bf16).abs().max()),
                  "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
                  "no_updates": True, "root_cause": "not_inferred_from_precision_difference_alone"}
        # Batch-one fidelity on a fixed prefix is a check, not a new test score.
        one, _ = forward_panel(model, panel["images"][:64], device, "float32", 1)
        report["fp32_batch1_vs_batchN_top1_agreement"] = float(one.argmax(1).eq(fp32[:64].argmax(1)).float().mean())
        report["batch_norm_precision_trace"] = batch_norm_precision_trace(model, panel["images"], device)
        head = model.classifier
        if hasattr(head, "basis"):
            max_error, disagreements = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(emb), args.batch):
                    z = emb[start:start + args.batch].to(device)
                    fast, direct = head(z), explicit_head(head, z)
                    max_error = max(max_error, float((fast - direct).abs().max()))
                    disagreements += int(fast.argmax(1).ne(direct.argmax(1)).sum())
            report["factorized_equivalence"] = {"max_absolute_score_error": max_error,
                "top1_disagreements": disagreements, "samples": len(emb),
                "passed": max_error < 1e-3 and disagreements == 0,
                "temporary_direct_reference_not_serialized": True}
        write_json(args.output / "precision-and-equivalence" / (run_identity(row) + ".json"), report)
        if row["variant"] == "dynamic_residual":
            if not report["factorized_equivalence"]["passed"]:
                raise ValueError("Counterfactual reference failed factorized equivalence")
            _, calibration_embeddings = forward_panel(model, panels["calibration"]["images"], device, "bfloat16", args.batch)
            def get_alpha(z):
                z = F.normalize(z.to(device), dim=1, eps=head.normalization_eps)
                norms = head._cached_norm_terms()[0]
                base = ((z @ head.basis.float().T) @ head.coefficients.float().T) / norms
                return head.gate_values(z, base, norms)
            with torch.no_grad():
                alpha = torch.cat([get_alpha(z).cpu() for z in emb.split(args.batch)])
                means = torch.cat([get_alpha(z).cpu() for z in calibration_embeddings.split(args.batch)]).mean(0)
            generator = torch.Generator().manual_seed(20261026)
            permutation = torch.randperm(len(alpha), generator=generator)
            interventions = {"calibration_mean_gate": means.expand_as(alpha),
                             "shuffled_gate": alpha[permutation],
                             "constant_half_gate": torch.full_like(alpha, 0.5),
                             "zero_residual_gate": torch.zeros_like(alpha)}
            counterfactual = []
            natural_loss = F.cross_entropy(bf16, y, reduction="none")
            with torch.no_grad():
                for name, gates in interventions.items():
                    scores = torch.cat([explicit_head(head, emb[start:start + args.batch].to(device),
                                                       gates[start:start + args.batch].to(device)).cpu()
                                        for start in range(0, len(emb), args.batch)])
                    delta = bf16.argmax(1).eq(y).float().numpy() - scores.argmax(1).eq(y).float().numpy()
                    counterfactual.append({"intervention": name,
                        "top1": float(scores.argmax(1).eq(y).float().mean()),
                        "prediction_change_fraction": float(scores.argmax(1).ne(bf16.argmax(1)).float().mean()),
                        "mean_counterfactual_minus_natural_ce": float((F.cross_entropy(scores, y, reduction="none") - natural_loss).mean()),
                        "natural_minus_intervention": paired_writer_interval(delta, panel["writers"], panel["versions"], repetitions=args.bootstrap)})
            gradients = []
            gate_parameters = [(n, p) for n, p in head.named_parameters() if n.startswith("gate_")]
            for objective in ("target_free_cross_entropy", "cosface_training_loss"):
                for start in range(0, len(emb), args.batch):
                    features = emb[start:start + args.batch].to(device).detach()
                    targets = y[start:start + args.batch].to(device)
                    scores = head(features, targets=targets if objective == "cosface_training_loss" else None)
                    loss = F.cross_entropy(scores, targets)
                    values = torch.autograd.grad(loss, [p for _, p in gate_parameters], allow_unused=True)
                    gradients.append({"objective": objective, "batch_start": start, "samples": len(targets),
                        "loss": float(loss.detach()), "classification_only": True,
                        "parameters": {n: None if g is None else {"norm": float(g.norm()), "max_abs": float(g.abs().max())}
                                       for (n, _), g in zip(gate_parameters, values, strict=True)}})
            q = F.normalize(head.coefficients.detach().float() @ head.basis.detach().float(), dim=1)
            q = q[head.residual_indices]; residual = head.residuals.detach().float()
            parallel = (residual * q).sum(1)
            perpendicular = (residual - parallel[:, None] * q).norm(dim=1)
            denominators, directional_derivatives = [], []
            with torch.no_grad():
                for start in range(0, len(emb), args.batch):
                    z = F.normalize(emb[start:start + args.batch].to(device), dim=1, eps=head.normalization_eps)
                    gates = alpha[start:start + args.batch].to(device)
                    vectors = q[None] + gates[..., None] * residual[None]
                    norm = vectors.norm(dim=2).clamp_min(head.normalization_eps)
                    zr = z @ residual.T
                    zv = (z[:, None] * vectors).sum(2)
                    vr = (vectors * residual[None]).sum(2)
                    derivative = zr / norm - zv * vr / norm.pow(3)
                    denominators.append(norm.cpu().flatten())
                    directional_derivatives.append(derivative.cpu().flatten())
            denominator = torch.cat(denominators)
            derivative = torch.cat(directional_derivatives)
            write_json(args.output / "gate-counterfactual" / (run_identity(row) + ".json"), {
                "split": "validation", "samples": len(y), "counterfactuals": counterfactual,
                "mean_gate_source": "fixed_calibration_prefix", "shuffling_seed": 20261026,
                "shuffling_policy": "permute_entire_gate_vectors_across_validation_inputs",
                "frozen_checkpoint_classification_gradients": gradients,
                "gradient_scope": "at selected checkpoint; frozen BF16-backbone embeddings, no optimizer or regularizer",
                "perpendicular_residual_norm_mean": float(perpendicular.mean()),
                "parallel_residual_norm_mean": float(parallel.abs().mean()),
                "residual_normalization_denominator_quantiles": {
                    str(p): float(torch.quantile(denominator, p)) for p in (0.0, 0.05, 0.5, 0.95, 1.0)},
                "d_target_free_cosine_d_alpha_absolute_quantiles": {
                    str(p): float(torch.quantile(derivative.abs(), p)) for p in (0.0, 0.05, 0.5, 0.95, 1.0)},
                "derivative_scope": "all selected-class scores on panel; geometry derivative, not total CE or optimizer gradient",
                "gate_parameter_norms": {n: float(p.detach().norm()) for n, p in gate_parameters},
                "saturation_below_005": float((alpha < 0.05).float().mean()),
                "saturation_above_095": float((alpha > 0.95).float().mean()),
                "pooled_saturation_is_not_class_balanced": True,
                "test_images_read": 0, "optimizer_updates": 0})
        if digest(Path(row["run"]) / "checkpoint.pt") != before:
            raise ValueError("Read-only analysis modified a checkpoint")
        del model, fp32, bf16, emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def deployment_controls(args, rows, panels):
    import torch
    from hccr.models.artifacts import load_model_from_run
    from hccr.benchmarking.quantization import build_dense_head_pair, load_classifier_deployment
    original_models, exported = {}, []
    images, targets = panels["validation"]["images"], panels["validation"]["targets"]
    torch.set_num_threads(args.threads)
    for row in rows:
        if row["group"] != "final_core":
            continue
        name = run_identity(row)
        model = load_model_from_run(Path(row["run"]), "cpu").float().eval()
        original_models[name] = model
        path = args.output / "native-exports" / (name + ".pt")
        path.parent.mkdir(exist_ok=True)
        torch.save({"metadata": row["metadata"], "state_dict": model.state_dict(),
                    "source_checkpoint_sha256": row["checkpoint_sha256"]}, path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        from hccr.models.artifacts import build_model_from_metadata
        restored = build_model_from_metadata(payload["metadata"]).float().eval()
        restored.load_state_dict(payload["state_dict"])
        with torch.no_grad():
            torch.testing.assert_close(model(images[:4]), restored(images[:4]), rtol=0, atol=0)
        exported.append({"model": name, "kind": "native_fp32", "file": str(path),
                         "bytes": path.stat().st_size, "sha256": digest(path), "reload_equivalent": True})
        del restored
        if row["variant"] != "dense_reference":
            continue
        fp32, int8 = build_dense_head_pair(model)
        candidates = {"fp32": fp32, "int8": int8}
        for kind, candidate in candidates.items():
            path = args.output / "native-exports" / (name + f"-head-{kind}.pt")
            torch.save({"schema_version": 1, "kind": kind,
                        "engine": torch.backends.quantized.engine,
                        "metadata": row["metadata"], "state_dict": candidate.state_dict()}, path)
            loaded = load_classifier_deployment(path)
            with torch.no_grad():
                torch.testing.assert_close(candidate(images[:4]), loaded(images[:4]), rtol=0, atol=0)
            scores = []
            with torch.no_grad():
                for start in range(0, len(images), args.batch):
                    scores.append(loaded(images[start:start + args.batch]))
            logits = torch.cat(scores)
            exported.append({"model": name, "kind": f"dense_head_{kind}", "file": str(path),
                "bytes": path.stat().st_size, "sha256": digest(path), "reload_equivalent": True,
                "validation_panel_samples": len(targets),
                "validation_panel_top1": float(logits.argmax(1).eq(targets).float().mean()),
                "backbone_precision": "float32", "scope": "head_only_dynamic_quantization"})
            del loaded
        with torch.no_grad():
            torch.testing.assert_close(model(images[:16]), fp32(images[:16]), rtol=1e-5, atol=1e-5)
        original_models[name + "-head-int8"] = int8
        write_json(args.output / "deployment-artifacts.json", exported)
    write_json(args.output / "deployment-artifacts.json", exported)
    # Mixed whole-model ONNX PTQ is a separate comparator, never called integer-only.
    for row in rows:
        if row["group"] != "final_core" or row["variant"] != "dense_reference":
            continue
        name = run_identity(row)
        try:
            onnx_dense_control(args, row, original_models[name], panels)
        except Exception as exc:
            write_json(args.output / "onnx" / (name + "-unsupported.json"), {
                "status": "failed_or_unsupported", "error": str(exc), "traceback": traceback.format_exc(),
                "no_scientific_recipe_change": True})
    return original_models


def onnx_session_options(threads, *, quantized=False):
    """Keep FP32 batch normalization separate; INT8 explicitly permits QDQ fusion."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if quantized
                                        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return options


def onnx_score_equivalence(reference, actual):
    """Report the original elementwise tolerance without concealing bad scores."""
    if reference.shape != actual.shape:
        raise ValueError(f"ONNX output shape {actual.shape} differs from native {reference.shape}")
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    difference = np.abs(actual - reference)
    close = np.isclose(actual, reference, rtol=1e-4, atol=1e-4)
    return {"samples": len(reference), "scores": reference.size, "finite": finite,
            "allclose": bool(finite and close.all()), "rtol": 1e-4, "atol": 1e-4,
            "violating_scores": int(np.count_nonzero(~close)),
            "max_abs_error": float(difference.max()) if finite else None,
            "mean_abs_error": float(difference.mean()) if finite else None,
            "top1_disagreements": int(np.count_nonzero(actual.argmax(1) != reference.argmax(1)))}


def onnx_dense_control(args, row, model, panels):
    import torch
    import onnx
    import onnxruntime as ort
    from collections import Counter
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    folder = args.output / "onnx"; folder.mkdir(exist_ok=True)
    name = run_identity(row); fp32_path = folder / (name + "-fp32.onnx")
    int8_path = folder / (name + "-mixed-int8.onnx")
    with torch.no_grad():
        torch.onnx.export(model, panels["fit"]["images"][:1], str(fp32_path),
                          input_names=["images"], output_names=["scores"],
                          dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
                          opset_version=17, dynamo=False, do_constant_folding=False)
    graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(graph)
    policy = {"exporter": "torchscript", "opset": 17, "do_constant_folding": False,
              "fp32_graph_optimization_level": "ORT_DISABLE_ALL",
              "mixed_int8_graph_optimization_level": "ORT_ENABLE_ALL",
              "reason": "Conv/BatchNorm fusion changes FP32 rounding in precision-sensitive frozen checkpoints",
              "input_shape": ["batch", *panels["fit"]["images"].shape[1:]],
              "graph_operators": dict(Counter(node.op_type for node in graph.graph.node))}
    options = onnx_session_options(args.threads)
    session = ort.InferenceSession(str(fp32_path), options, providers=["CPUExecutionProvider"])
    images = panels["validation"]["images"]
    equivalence = {"reference": "unmodified native PyTorch FP32 CPU", "policy": policy,
                   "probes": [], "panel_batches": []}
    equivalence_path = folder / (name + "-equivalence.json")
    for batch in (1, 32):
        with torch.no_grad():
            reference = model(images[:batch]).numpy()
        actual = session.run(None, {"images": images[:batch].numpy()})[0]
        observation = {"batch_size": batch, **onnx_score_equivalence(reference, actual)}
        equivalence["probes"].append(observation)
        write_json(equivalence_path, equivalence)
        if not observation["allclose"]:
            raise ValueError(f"ONNX FP32 batch-{batch} equivalence failed: {observation}")
    fp32_predictions = []
    for start in range(0, len(images), args.batch):
        batch_images = images[start:start + args.batch]
        with torch.no_grad():
            reference = model(batch_images).numpy()
        actual = session.run(None, {"images": batch_images.numpy()})[0]
        observation = {"start": start, **onnx_score_equivalence(reference, actual)}
        equivalence["panel_batches"].append(observation)
        if not observation["allclose"]:
            write_json(equivalence_path, equivalence)
            raise ValueError(f"ONNX FP32 validation-panel equivalence failed: {observation}")
        fp32_predictions.extend(actual.argmax(1).tolist())
    equivalence["status"] = "passed"
    equivalence["panel_samples"] = len(images)
    equivalence["panel_max_abs_error"] = max(b["max_abs_error"] for b in equivalence["panel_batches"])
    equivalence["panel_top1_disagreements"] = sum(b["top1_disagreements"] for b in equivalence["panel_batches"])
    write_json(equivalence_path, equivalence)

    class FitCalibration(CalibrationDataReader):
        def __init__(self):
            self.samples = iter(panels["fit"]["images"].split(1))

        def get_next(self):
            sample = next(self.samples, None)
            return None if sample is None else {"images": sample.numpy()}

    quantize_static(str(fp32_path), str(int8_path), FitCalibration(),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    results = []
    for kind, path in (("onnx_fp32", fp32_path), ("onnx_mixed_int8", int8_path)):
        predictions = fp32_predictions if kind == "onnx_fp32" else []
        if kind == "onnx_mixed_int8":
            session = ort.InferenceSession(str(path), onnx_session_options(args.threads, quantized=True),
                                           providers=["CPUExecutionProvider"])
            for start in range(0, len(images), args.batch):
                values = session.run(None, {"images": images[start:start + args.batch].numpy()})[0]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite ONNX outputs")
                predictions.extend(values.argmax(1).tolist())
        results.append({"kind": kind, "bytes": path.stat().st_size, "sha256": digest(path),
                        "graph_optimization_level": policy["fp32_graph_optimization_level" if kind == "onnx_fp32"
                                                          else "mixed_int8_graph_optimization_level"],
                        "validation_panel_samples": len(images),
                        "validation_panel_top1": float(np.mean(predictions == panels["validation"]["targets"].numpy()))})
    write_json(folder / (name + "-report.json"), {
        "status": "completed", "models": results,
        "fp32_max_abs_equivalence_error": equivalence["panel_max_abs_error"],
        "fp32_equivalence": str(equivalence_path), "export_policy": policy,
        "calibration": "64 fixed fit-split images; allocation-calibration and holdouts excluded",
        "quantization": "mixed QDQ W8A8 supported Conv/MatMul/Gemm; other operators remain floating point",
        "provider": "CPUExecutionProvider", "full_test_accuracy": "not_run_panel_only",
        "runtime": ort.__version__})


def cpu_benchmarks(args, rows, panels, candidates):
    import torch
    import psutil
    from hccr.models.artifacts import load_model_from_run
    from hccr.benchmarking.schema import summarize_timings
    class ONNXModel:
        def __init__(self, path):
            self.path = path

        def configure_threads(self, count):
            import onnxruntime as ort
            options = onnx_session_options(count, quantized=self.path.name.endswith("-mixed-int8.onnx"))
            self.session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])

        def __call__(self, images):
            return torch.from_numpy(self.session.run(None, {"images": images.numpy()})[0])

    process = psutil.Process()
    affinity = process.cpu_affinity()[:max(4, args.threads)]
    process.cpu_affinity(affinity)
    torch.set_num_interop_threads(1)
    for row in rows:
        name = run_identity(row)
        if name not in candidates:
            candidates[name] = load_model_from_run(Path(row["run"]), "cpu").float().eval()
    for path in (args.output / "onnx").glob("*-report.json"):
        report = read_json(path)
        if report["status"] != "completed":
            continue
        name = path.name.removesuffix("-report.json")
        for suffix, kind in (("fp32", "onnx_fp32"), ("mixed-int8", "onnx_mixed_int8")):
            artifact = path.with_name(name + f"-{suffix}.onnx")
            expected = next(x["sha256"] for x in report["models"] if x["kind"] == kind)
            if digest(artifact) != expected:
                raise ValueError("ONNX export changed before benchmark")
            candidates[name + "-" + kind] = ONNXModel(artifact)
    inputs = panels["validation"]["images"][:64]
    protocol = {"device": "cpu", "processor": platform.processor(), "platform": platform.platform(),
                "torch": str(torch.__version__), "batch_size": 1, "dtype": "float32",
                "input": "64 recorded validation-panel images, cycle in fixed order", "shape": [1, 1, 96, 96],
                "mode": "eager", "scope": ["head_only", "model_forward_without_preprocessing"],
                "affinity": affinity, "thread_counts": [1, args.threads],
                "warmup": 20, "iterations_per_block": args.iterations, "blocks": args.blocks,
                "order": "seeded shuffled candidates per block", "raw_timings_saved": True,
                "isolation": "single suite process; interactive host isolation not certified",
                "no_speedup_verdict_without_repeatability_and_isolation_review": True,
                "onnx_scope": "model-forward only, including NumPy/PyTorch adapter; compare ORT FP32 and ORT INT8 within their engine",
                "onnx_graph_optimization": {"fp32": "ORT_DISABLE_ALL", "mixed_int8": "ORT_ENABLE_ALL"},
                "onnx_comparison_limit": "FP32 and INT8 use different graph policies; no isolated quantization speedup claim",
                "cpu_percent_at_start": psutil.cpu_percent(), "memory_at_start": psutil.virtual_memory()._asdict()}
    write_json(args.output / "cpu-benchmark-protocol.json", protocol)
    with torch.inference_mode():
        embeddings = {name: [m._forward_embedding(x[None]).contiguous() for x in inputs]
                      for name, m in candidates.items() if hasattr(m, "_forward_embedding")}
    summaries = []
    for threads in sorted({1, args.threads}):
        torch.set_num_threads(threads)
        for module in candidates.values():
            if isinstance(module, ONNXModel):
                module.configure_threads(threads)
        observations = {name: {scope: [] for scope in protocol["scope"]} for name in candidates}
        for block in range(args.blocks):
            order = list(candidates); random.Random(20261026 + block).shuffle(order)
            for name in order:
                model = candidates[name]
                for scope in protocol["scope"]:
                    if scope == "head_only" and name not in embeddings:
                        continue
                    samples = embeddings[name] if scope == "head_only" else [x[None] for x in inputs]
                    module = model.classifier if scope == "head_only" else model
                    timings = []
                    with torch.inference_mode():
                        for i in range(20):
                            module(samples[i % len(samples)])
                        for i in range(args.iterations):
                            begin = time.perf_counter_ns()
                            scores = module(samples[i % len(samples)])
                            timings.append((time.perf_counter_ns() - begin) / 1e6)
                            if not torch.isfinite(scores).all():
                                raise ValueError("Nonfinite benchmark output")
                    summary = summarize_timings(timings)
                    observations[name][scope].append(summary)
                    write_json(args.output / "cpu-raw" / f"t{threads}-b{block}-{name}-{scope}.json", {
                        "timings_ms": timings, "summary": summary, "failures": 0,
                        "cpu_percent_after_block": psutil.cpu_percent()})
                print(f"cpu threads={threads} block={block + 1}/{args.blocks} {name}", flush=True)
        for name, scopes in observations.items():
            for scope, blocks in scopes.items():
                if not blocks:
                    continue
                summaries.append({"model": name, "threads": threads, "scope": scope,
                    "runtime": "onnxruntime" if isinstance(candidates[name], ONNXModel) else "pytorch",
                    **{key: statistics.median(b[key] for b in blocks) for key in blocks[0]},
                    "block_mean_min_ms": min(b["latency_mean_ms"] for b in blocks),
                    "block_mean_max_ms": max(b["latency_mean_ms"] for b in blocks)})
        write_csv(args.output / "cpu-benchmarks.csv", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("artifacts/downloads/vastai-20261026"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hwdb1_0-1_1_official_five_splits/manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--panel", type=int, default=4096)
    parser.add_argument("--calibration-panel", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--statistics-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve(); args.downloads = args.downloads.resolve()
    args.manifest = args.manifest.resolve()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; completed evidence is never overwritten")
    args.output.mkdir(parents=True)
    state = {"pid": os.getpid(), "status": "running", "stages": {}, "training": False}
    import psutil
    import torch
    write_json(args.output / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
        "cpu_isa": torch.backends.cpu.get_cpu_capability(),
        "quantized_engines": torch.backends.quantized.supported_engines,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "ram": psutil.virtual_memory()._asdict(), "disk": psutil.disk_usage(str(args.output))._asdict(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "lmdb", "psutil", "timm", "onnx", "onnxruntime")
                     if importlib.util.find_spec(name) is not None},
    })
    write_json(args.output / "status.json", state)
    def stage(name, action):
        state["stages"][name] = {"status": "running", "started": time.time()}
        write_json(args.output / "status.json", state)
        try:
            result = action()
            state["stages"][name].update(status="completed", finished=time.time())
            return result
        except Exception as exc:
            state["stages"][name].update(status="blocked", error=str(exc), traceback=traceback.format_exc(), finished=time.time())
            traceback.print_exc()
            return None
        finally:
            write_json(args.output / "status.json", state)
    rows = stage("input_inventory", lambda: inventory(args))
    if rows is not None:
        stage("prediction_statistics", lambda: analyze_predictions(args, rows))
        stage("archived_gating_dynamics", lambda: analyze_telemetry(args, rows))
        if not args.statistics_only:
            panels = stage("image_panel_reconciliation", lambda: reconcile_panel(args, rows))
            if panels is not None:
                stage("precision_equivalence_and_gate_interventions", lambda: analyze_models(args, rows, panels))
                candidates = stage("native_and_quantized_exports", lambda: deployment_controls(args, rows, panels))
                stage("cpu_mean_percentiles", lambda: cpu_benchmarks(args, rows, panels, candidates or {}))
    blocked = [name for name, entry in state["stages"].items() if entry["status"] == "blocked"]
    if list((args.output / "onnx").glob("*-unsupported.json")):
        blocked.append("onnx_exports_see_individual_receipts")
    frozen_sources = read_json(args.output / "input-inventory.json").get("research_source_hashes", {}) if rows else {}
    source_root = Path(__file__).resolve().parents[1]
    changed = [name for name, value in frozen_sources.items() if digest(source_root / name) != value]
    if changed:
        blocked.append("research_source_changed_during_analysis")
        state["changed_source_files"] = changed
    state.update(status="completed_with_blockers" if blocked else "completed", blocked_stages=blocked,
                 finished=time.time(), optimizer_updates=0)
    write_json(args.output / "status.json", state)
    write_json(args.output / "scope-and-remaining.json", {
        "new_test_or_icdar_image_inference": False,
        "validation_panel_results_are_not_full_endpoint_accuracy": True,
        "conditional_training_not_started": ["QAT recovery", "pruning recovery", "new vocabulary", "periodic H reselection"],
        "factorized_hybrid_PTQ": "not_implemented; native factorized serialization/equivalence is separate",
        "quantized_full_test_accuracy": "pending; only validation panel assessed",
        "historical_classification_only_gradients": "unrecoverable at unsaved training steps",
    })
    print(json.dumps({"status": state["status"], "output": str(args.output), "blockers": blocked}), flush=True)
    return 0 if not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
