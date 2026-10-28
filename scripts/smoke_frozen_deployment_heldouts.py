"""Separate real-batch check for every frozen artifact; never writes formal outputs."""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import struct
import time

import evaluate_frozen_deployment_heldouts as run


def main():
    import numpy as np
    import torch
    import onnxruntime as ort
    from hccr.models.artifacts import load_model_from_run
    from hccr.benchmarking.quantization import load_classifier_deployment
    from hccr.data.gnt import load_source_image
    from hccr.preprocessing import EvalPreprocessor
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol = run.verified_protocol(run.OUT)
    destination = run.OUT / f"real-batch-check-{time.time_ns()}"
    destination.mkdir()
    result = {"protocol_file_sha256": run.sha(run.OUT / "protocol.json"), "runtime": run.runtime(),
              "formal_full_endpoint_metrics": False, "no_selection_or_tuning": True,
              "sample_count_per_endpoint": 96, "results": [], "inputs": {}}
    with run.owner(destination):
        batches = {}
        for endpoint in run.COUNTS:
            selected = list(itertools.islice(run.rows(run.OUT / "cache" / f"{endpoint}-rows.csv"), 96))
            prepared = []
            if endpoint == "test":
                progress = run.read(run.OUT / "cache/test-progress.json")
                if progress["completed"] < 96:
                    raise ValueError("No committed native-pixel-verified test prefix")
                cached = np.load(run.OUT / "cache/test-uint8.npy", mmap_mode="r")
                first = progress["blocks"][0]
                if hashlib.sha256(cached[first["start"]:first["end"]].tobytes()).hexdigest() != first["sha256"]:
                    raise ValueError("Committed cache prefix changed")
                images = torch.from_numpy(np.array(cached[:96], copy=True)).float().mul_(1 / 255)
                del cached
                result["inputs"][endpoint] = {"mode": "verified committed immutable cache prefix", "cache_block": first}
            else:
                for row in selected:
                    image = load_source_image(run.ROOT / "data/raw", row)
                    native = hashlib.sha256(struct.pack("<HH", image.width, image.height))
                    native.update(image.tobytes())
                    if native.hexdigest() != row["pixel_sha256"]:
                        raise ValueError("ICDAR smoke native pixel mismatch")
                    prepared.append(np.asarray(EvalPreprocessor(96, 4)(image), dtype="uint8"))
                images = torch.from_numpy(np.stack(prepared)[:, None]).float().mul_(1 / 255)
                result["inputs"][endpoint] = {"mode": "native GNT pixels rehashed then frozen preprocessing"}
            targets = np.asarray([int(r["class_id"]) for r in selected], dtype="int64")
            result["inputs"][endpoint].update(sample_identity_sha256=hashlib.sha256("\n".join(r["sample_id"] for r in selected).encode()).hexdigest(),
                                               targets_sha256=hashlib.sha256(targets.tobytes()).hexdigest())
            batches[endpoint] = images, targets
        with torch.no_grad():
            for seed in (7, 17, 29):
                source_artifact = next(a for a in protocol["artifacts"] if a["seed"] == seed and a["family"] == "native_fp32")
                source = load_model_from_run(Path(source_artifact["run"]), "cpu").float().eval()
                for artifact in [a for a in protocol["artifacts"] if a["seed"] == seed]:
                    family = artifact["family"]
                    if family == "native_fp32":
                        model = source
                    elif family == "native_head_int8":
                        model = load_classifier_deployment(Path(artifact["path"]))
                        backbone_hash = run.pair_identity(source, model)
                    else:
                        options = ort.SessionOptions()
                        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
                        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                        options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, artifact["optimization"])
                        session = ort.InferenceSession(artifact["path"], sess_options=options, providers=["CPUExecutionProvider"])
                    for endpoint, (images, targets) in batches.items():
                        outputs = []
                        for batch in images.split(32):
                            if family.startswith("native"):
                                scores = model(batch)
                                if family == "native_head_int8":
                                    embeddings = source._forward_embedding(batch)
                                    torch.testing.assert_close(embeddings, model._forward_embedding(batch), rtol=0, atol=0)
                                    torch.testing.assert_close(scores, model.classifier(embeddings), rtol=0, atol=0)
                            else:
                                scores = torch.from_numpy(session.run(None, {session.get_inputs()[0].name: batch.numpy()})[0])
                            if not torch.isfinite(scores).all():
                                raise ValueError("Nonfinite real-batch scores")
                            outputs.append(scores)
                        logits = torch.cat(outputs)
                        top5 = logits.topk(5, dim=1).indices.numpy().astype("int32")
                        path = destination / f"seed-{seed}-{family}-{endpoint}.npz"
                        np.savez_compressed(path, targets=targets, top5=top5)
                        record = {"seed": seed, "family": family, "endpoint": endpoint, "samples": len(targets),
                                  "top1_hits": int((top5[:, 0] == targets).sum()), "top5_hits": int((top5 == targets[:, None]).any(1).sum()),
                                  "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                                  "predictions_sha256": run.sha(path), "artifact_sha256": artifact["sha256"], "all_scores_finite": True}
                        if family == "native_head_int8":
                            record.update(backbone_sha256=backbone_hash, exact_embedding_and_direct_score_match=True)
                        result["results"].append(record)
                        run.atomic(destination / "results.json", result)
                        print(record, flush=True)
        result["status"] = "completed"
        result["completed_artifact_endpoint_checks"] = len(result["results"])
        run.atomic(destination / "results.json", result)
        print(f"Completed {len(result['results'])} real artifact/endpoint checks: {destination}", flush=True)


if __name__ == "__main__":
    main()
