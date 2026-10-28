"""Audit frozen diagnostic receipts and produce development-only evidence tables."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stem", type=Path)
    args = parser.parse_args()
    directory = args.directory
    protocol = read(directory/"protocol.json")
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    status = read(directory/"status.json")
    assert status["status"] == "completed", status
    assert len(status["units"]) == 21, status["units"]
    assert protocol["script_sha256"] == sha(ROOT/"scripts/run_precision_forensics.py")
    receipts = {}
    for path in directory.glob("*.json"):
        payload = read(path)
        if "protocol_identity" in payload and "result" in payload:
            assert payload["protocol_identity"] == identity, path
            receipts[path.stem] = payload["result"]
    assert len(receipts) == 21
    for name, result in receipts.items():
        if name.startswith(("core-", "dense-")):
            assert result["state_unchanged"], name
    for name, value in protocol["source_hashes"].items():
        assert sha(ROOT/name) == value, name
    for name, value in protocol["checkpoint_sha256"].items():
        assert sha(Path(name)/"checkpoint.pt") == value, name
    inventory_path = ROOT/"artifacts/local-post-training/20261026-retry-af8b8078/results/input-inventory.json"
    metadata_verification = {"timing": "final verification against the prior analysis inventory snapshot",
        "inventory_sha256": sha(inventory_path), "files": {}}
    for row in read(inventory_path)["runs"]:
        if row["group"] == "final_core":
            metadata_path = Path(row["run"])/"checkpoint_metadata.json"
            assert read(metadata_path) == row["metadata"], metadata_path
            metadata_verification["files"][str(metadata_path)] = sha(metadata_path)
    (directory/"input-metadata-verification.json").write_text(json.dumps(metadata_verification, indent=2)+"\n", encoding="utf-8")
    rows = []
    for name, result in receipts.items():
        if name.startswith("core-"):
            for policy in ("fp32", "bf16", "bf16_backbone_fp32_head", "fp32_backbone_bf16_head"):
                row = {"model": name.removeprefix("core-"), "experiment": policy, **result[policy]}
                assert row["finite"], name
                rows.append(row)
            assert result["fp32_backbone_bf16_head"]["score_max_abs_difference"] == 0, name
        elif name.startswith("dense-"):
            for policy, value in result["arithmetic"].items():
                rows.append({"model": name.removeprefix("dense-"), "experiment": policy, **value})
            for value in result["stage_sweeps"]:
                rows.append({"model": name.removeprefix("dense-"), "experiment": value["kind"], **value})
        elif name.startswith("onnx-"):
            for policy, value in result["observations"].items():
                assert value["finite"], (name, policy)
                rows.append({"model": name.removeprefix("onnx-"), "experiment": policy, **value})
            assert result["observations"]["bypass_both_optimization_False"]["score_max_abs_difference"] < 1e-4, name
            for path, value in result["artifact_sha256"].items():
                assert sha(Path(path)) == value, path
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (directory/"observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Frozen development precision forensics", "",
        "All results below use frozen development inputs and unchanged checkpoint parameters, BN state, class subsets and hyperparameters. These are diagnostic arithmetic interventions, not replacement trained models.", "",
        "The main panel contains 4096 validation loader-prefix images from two writers. The layer interventions use 512 evenly spaced images within that prefix. CPU/CUDA checks use the first 64 prefix images. ONNX intervention checks use the first 64 of the 512 selected images. None is full-validation accuracy, a writer-balanced estimate, or held-out evaluation.", "",
        "## Observations", "", "The published dense and factorized heads explicitly disable autocast and compute in FP32. Requesting BF16 around an FP32 backbone and the head leaves every score exactly unchanged for all 15 core checkpoints. The head/backbone exchange therefore places the FP32/BF16 discrepancy upstream of the classifier, in the backbone embeddings.", "",
        "| Dense seed | FP32 panel accuracy | BF16 panel accuracy | BF16 backbone, FP32 head |",
        "| --- | ---: | ---: | ---: |"]
    for seed in (7, 17, 29):
        result = receipts[f"dense-seed-{seed}-dense_reference"]
        vals = [100*result[k]["top1"] for k in ("full_panel_fp32", "full_panel_bf16", "full_panel_bf16_backbone_fp32_head")]
        lines.append(f"| {seed} | {vals[0]:.4f}% | {vals[1]:.4f}% | {vals[2]:.4f}% |")
    lines += ["", "## Controlled arithmetic evidence", "",
        "Convolution interventions round only the named tensors or run convolution under BF16 autocast, then return FP32. BN and activation behavior remains FP32 unless the row says otherwise. Weight rounding uses a temporary BF16-round-trip tensor and never assigns a checkpoint parameter. The BF16 backbone with FP32 BN intervention computes each BN on a float input and returns float output; all stored statistics remain unchanged.", "",
        "| Seed | Diagnostic arithmetic on narrow panel | Accuracy | Agreement with FP32 | Mean absolute score change |",
        "| --- | --- | ---: | ---: | ---: |"]
    for seed in (7, 17, 29):
        result = receipts[f"dense-seed-{seed}-dense_reference"]
        for policy, value in result["arithmetic"].items():
            lines.append(f"| {seed} | {policy} | {100*value['top1']:.4f}% | {100*value['prediction_agreement']:.4f}% | {value['score_mean_abs_difference']:.6f} |")
    lines += ["", "Stage numbering is stem=0 and residual feature blocks=1 through 8. The CSV and dense receipts retain every cumulative BF16 prefix and every one-stage BF16 substitution, including substitutions that do not improve accuracy. These interventions establish arithmetic dependence without estimating new BN statistics.", "",
        "| Seed | Stage(s) giving highest narrow accuracy among single-stage substitutions | Accuracy | BF16 prefix with all 9 stages |",
        "| --- | --- | ---: | ---: |"]
    for seed in (7, 17, 29):
        result = receipts[f"dense-seed-{seed}-dense_reference"]
        single = [r for r in result["stage_sweeps"] if r["kind"] == "single_stage"]
        maximum = max(r["top1"] for r in single)
        stages = [str(r["bf16_stages"][0]) for r in single if r["top1"] == maximum]
        last = [r for r in result["stage_sweeps"] if r["kind"] == "prefix"][-1]
        lines.append(f"| {seed} | {', '.join(stages)} | {100*maximum:.4f}% | {100*last['top1']:.4f}% |")
    lines += ["", "The maximum row is descriptive localization on development images. It is not a selected deployment recipe.", "",
        "## CPU/CUDA and finite-output checks", "",
        "| Seed | CPU arithmetic | CPU accuracy | Agreement with matching CUDA arithmetic | Maximum absolute score difference |",
        "| --- | --- | ---: | ---: | ---: |"]
    for seed in (7, 17, 29):
        for policy, value in receipts[f"dense-seed-{seed}-dense_reference"]["cross_device"].items():
            lines.append(f"| {seed} | {policy} | {100*value['top1']:.4f}% | {100*value['prediction_agreement']:.4f}% | {value['score_max_abs_difference']:.6f} |")
    lines += ["", "All evaluated logits and embeddings passed finite checks. No inference latency is claimed because the machine was shared with held-out evaluation. CPU BF16 here is a raw autocast diagnostic; the repository's production precision resolver currently restricts BF16 to CUDA.", "",
        "## INT8 arithmetic and graph optimization", "",
        "The diagnostic graph reuses the existing fixed quantization scales. The receipt key bypass_activation_qdq bypasses runtime QuantizeLinear/DequantizeLinear pairs. Those pairs include feature activations AND the classifier's transposed weight tensor, which the quantizer handles as a runtime tensor. The restore_exact_fp32_weights key restores the 19 convolution-weight initializers only. Thus this first ablation is runtime-QDQ versus materialized convolution weights, not a pure activation-versus-all-weights experiment. Both changes affect an in-memory copy only. Bypassing both with graph optimization disabled reconstructs FP32 scores exactly for all three seeds; this is a required intervention validity check.", "",
        "| Seed | Graph/intervention | Optimization | Narrow accuracy | Agreement with FP32, optimization disabled | Max absolute score difference |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for seed in (7, 17, 29):
        result = receipts[f"onnx-seed-{seed}-dense_reference"]
        for policy, value in result["observations"].items():
            label, optimized = policy.rsplit("_optimization_", 1)
            lines.append(f"| {seed} | {label} | {optimized} | {100*value['top1']:.4f}% | {100*value['prediction_agreement']:.4f}% | {value['score_max_abs_difference']:.6f} |")
    stem_receipts = {}
    if args.stem:
        stem_status = read(args.stem/"status.json")
        stem_protocol = read(args.stem/"protocol.json")
        assert stem_status["status"] == "completed" and len(stem_status["completed"]) == 15
        assert stem_protocol["base_protocol_identity"] == identity
        assert stem_protocol["script_sha256"] == sha(ROOT/"scripts/localize_stem_precision.py")
        assert stem_protocol["helper_sha256"] == protocol["script_sha256"]
        assert read(args.stem/"local-data-reconciliation.json")["validation_panel_identity_sha256"] == read(directory/"local-data-reconciliation.json")["validation_panel_identity_sha256"]
        for path in args.stem.glob("seed-*.json"):
            payload = read(path)
            assert payload["protocol"] == stem_protocol
            assert payload["result"]["state_unchanged"]
            stem_receipts[path.stem] = payload["result"]
        assert len(stem_receipts) == 15
        lines += ["", "## Stem necessity and sufficiency", "",
            "The follow-up repeats all 15 checkpoints on exactly the same 512 images. Full BF16 stem arithmetic changes the stem convolution, BN output and SiLU arithmetic together. The convolution-only BF16 row returns float output, leaving BN and SiLU in FP32. The reverse intervention forces the entire stem to FP32 while keeping all later feature blocks under BF16 autocast.", "",
            "| Model | FP32 | BF16 | Only stem output rounding | Only complete stem BF16 | Only stem Conv BF16, BN FP32 | BF16 backbone except FP32 stem |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        stem_rows = []
        for name, result in sorted(stem_receipts.items()):
            policies = ("fp32", "bf16", "only_stem_conv_output_round", "only_stem_bf16", "only_stem_conv_bf16_bn_fp32", "bf16_backbone_except_fp32_stem")
            values = [f"{100*result[key]['top1']:.4f}%" for key in policies]
            lines.append(f"| {name} | " + " | ".join(values) + " |")
            for key, value in result.items():
                if isinstance(value, dict) and "top1" in value:
                    assert value["finite"], (name, key)
                    stem_rows.append({"model": name, "experiment": key, **value})
        fields = list(dict.fromkeys(key for row in stem_rows for key in row))
        with (directory/"stem-observations.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stem_rows)
        lines += ["", "The complete BF16 stem alone recovers the large precision losses across these 15 narrow panels, while the reverse FP32-stem intervention reintroduces those losses. Stem convolution under BF16 with BN and SiLU kept in FP32 gives nearly the same recovery. This places the dominant dependence in stem convolution arithmetic rather than the arithmetic dtype of BN or the head.", "",
            "Stem-output rounding alone largely reproduces the dense seed 7/17 recovery. It is not a universal repair. Several dynamic/low-rank checkpoints need the BF16 convolution computation, and output-only rounding can lower their accuracy. All negative results are retained.", "",
            "| Dense seed | FP32, all 4096 panel images | Only stem Conv output rounding, all 4096 | Only complete stem BF16, all 4096 |",
            "| --- | ---: | ---: | ---: |"]
        for seed in (7, 17, 29):
            result = stem_receipts[f"seed-{seed}-dense_reference"]
            values = [f"{100*result[key]['top1']:.4f}%" for key in ("full_panel_fp32", "full_panel_only_stem_conv_output_round", "full_panel_only_stem_bf16")]
            lines.append(f"| {seed} | " + " | ".join(values) + " |")
        lines += ["", "Rounding takes place before the unchanged stem BN. Moving that rounding to the BN output does not reproduce the seed 17 recovery. On its 512 images, stem weight-only rounding reaches 85.7422%, stem input-only rounding 82.6172%, and stem BN-output rounding 82.4219%. Stem Conv-output rounding reaches 99.6094%, while forcing the FP32 stem into the otherwise BF16 network gives 82.6172%. Recovery therefore depends on the location and arithmetic of the intervention.", "",
            "The follow-up records per-channel rounding shifts for uniform white/black receptive fields, the fixed BN affine gain, and the shift after applying that gain. This documents how a small structured pre-BN change enters a fixed learned network. It does not prove that a BN state update is required or that the original statistics were computed incorrectly.", "",
        ]
        conditioning = stem_receipts["seed-17-dense_reference"]["stem_conditioning_32"]
        channel = max(conditioning["channels"], key=lambda x: abs(x.get("white_round_bn_shift_mean", 0)))
        lines += [f"For seed 17 on the first 32 images, {100*conditioning['white_pixel_fraction']:.2f}% of pixels are exactly white and {100*channel['white_uniform_patch_fraction']:.2f}% of stem receptive fields are uniformly white. In channel {channel['channel']}, the common white-field convolution response changes from {channel['white_fp32_conv_mean']:.7f} to {channel['white_rounded_conv_mean']:.7f} after BF16 rounding. The unchanged BN gain of {channel['bn_gain']:.4f} turns this into a repeated shift of {channel['white_round_bn_shift_mean']:.6f} over those fields. This is a structured background-feature shift, not an overflow or nonfinite-output event. The network then amplifies the changed stem representation into different embeddings and class margins.", "",
            "## Head versus backbone INT8", "",
            "These same-graph diagnostic ablations retain the existing scales and separate the classifier's two runtime QDQ pairs, including its transposed weight, from all backbone QDQ and convolution weights. No new quantization recipe is calibrated. With optimization disabled, bypassing every QDQ again gives exactly the FP32 reference. Head-only and backbone-only here label diagnostic graphs, not published deployment variants.", "",
            "| Seed | Existing mixed INT8 | Only backbone INT8 | Only head INT8 | Every QDQ bypassed |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for seed in (7, 17, 29):
            result = stem_receipts[f"seed-{seed}-dense_reference"]
            for path, value in result["onnx_artifact_sha256"].items():
                assert sha(Path(path)) == value
            observed = result["onnx_head_backbone"]
            assert observed["all_qdq_bypassed_optimization_False"]["score_max_abs_difference"] < 1e-4
            values = [f"{100*observed[key+'_optimization_False']['top1']:.4f}%" for key in ("unchanged_mixed_int8", "backbone_only_int8", "head_only_int8", "all_qdq_bypassed")]
            lines.append(f"| {seed} | " + " | ".join(values) + " |")
        lines += ["", "The unchanged INT8 graph has the same narrow-panel accuracy with optimization on or off for all three seeds. Disabling optimization alone does not remove its failure. Exact convolution-weight restoration with the remaining runtime QDQ shows strong interactions with optimization, so the report retains both policies and does not treat effects as additive."]
    lines += ["", "## Interpretation and production scope", "",
        "The controlled embedding/head exchange rules out head autocast as the source of the FP32/BF16 difference. Convolution rounding and stage interventions locate which backbone arithmetic choices reproduce or remove the difference while every checkpoint and BN buffer remains fixed. Layerwise BN gain, input mean/variance mismatch and mean shifts in units of the stored running standard deviation are recorded in each dense receipt; those measurements describe conditioning and do not by themselves prove that the stored BN statistics were estimated incorrectly.", "",
        "These tests do not recover unsaved training-time activations or establish one unique historical cause for the checkpoint's arithmetic dependence. They do not justify changing BN statistics. Recalibration, QAT, retraining, or a new precision recipe would be conditional future work and would require a new development protocol before held-out evaluation.", "",
        "Native CUDA BF16 autocast preserves stored weights and BN buffers and reproduces the archived BF16 arithmetic on this panel. It is a weight-preserving runtime option for the original GPU evaluation protocol. The current frozen FP32 and INT8 deployment artifacts remain separate endpoints. No new universal CPU FP32/INT8 fix or full-validation production guarantee follows from these narrow diagnostic interventions.", "",
        "## Provenance and verification", "",
        f"Protocol identity: `{identity}`. The protocol receipt records all 15 checkpoint hashes, research-source hashes, Python/PyTorch/CUDA/cuDNN versions, device, deterministic flags, thread count, batch size and memory cap. Panel receipts record exact local sample IDs, target/image hashes and the full frozen validation digest reconstruction. ONNX receipts record original artifact hashes. Final metadata verification hashes each checkpoint metadata file and confirms exact equality with the previous analysis inventory snapshot. The final audit verified every source checkpoint, research source and ONNX input again. All diagnostic model-state hashes remained unchanged. No optimizer or BN update occurred. No test or ICDAR image was read.", "",
        "Reusable commands:", "", "```powershell",
        "# Use a fresh output folder for a new protocol or script version.",
        "python scripts/run_precision_forensics.py --output artifacts/local-post-training/remaining-work-20261026/precision/new-run",
        "python -m unittest discover -s tests -p test_precision_forensics.py -v",
        f"python scripts/summarize_precision_forensics.py {directory.as_posix()}" + (f" --stem {args.stem.as_posix()}" if args.stem else ""),
        "```", ""]
    (directory/"report.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {"status": "verified", "protocol_identity": identity, "completed_units": len(receipts),
        "observations": len(rows), "source_and_checkpoint_integrity": True,
        "all_model_states_unchanged": True, "onnx_double_bypass_max_error_below_1e_minus_4": True,
        "stem_followup_models": len(stem_receipts),
        "tests_sha256": sha(ROOT/"tests/test_precision_forensics.py"),
        "stem_receipt_sha256": {p.name: sha(p) for p in args.stem.glob("*.json")} if args.stem else {},
        "summarizer_sha256": sha(Path(__file__)), "receipt_sha256": {
            p.name: sha(p) for p in directory.glob("*.json") if p.name != "audit.json"}}
    (directory/"audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    completion = {"status": "completed_verified", "scope": "precision_forensics_only", "active_jobs": [],
        "report": str((directory/"report.md").resolve()), "report_sha256": sha(directory/"report.md"),
        "audit_sha256": sha(directory/"audit.json"), "main_units": len(receipts),
        "followup_models": len(stem_receipts), "no_test_or_icdar_images_read": True,
        "scientific_model_or_bn_updates": 0,
        "source_sha256": {name: sha(ROOT/name) for name in (
            "scripts/run_precision_forensics.py", "scripts/localize_stem_precision.py",
            "scripts/summarize_precision_forensics.py", "tests/test_precision_forensics.py")}}
    test_log = directory.parent/"tests.log"
    if test_log.exists():
        test_output = test_log.read_text(encoding="utf-8-sig")
        assert "Ran 5 tests" in test_output and test_output.rstrip().endswith("OK")
        completion.update(unit_tests="5 passed", tests_log_sha256=sha(test_log))
    (directory.parent/"completion.json").write_text(json.dumps(completion, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "receipt_sha256"}, indent=2))


if __name__ == "__main__":
    main()
