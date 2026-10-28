"""Frozen deployment accuracy only: exact endpoint joins, bounded caches, resumable units.

No fitting, calibration, model selection, GPU inference, or latency benchmarking.
Run `freeze` before `run`; a changed runner/runtime/input requires a new protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import struct
import sys
import time
import traceback
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
OUT = ROOT / "artifacts/local-post-training/remaining-work-20261026/deployment-heldouts"
DOWNLOAD = ROOT / "artifacts/downloads/vastai-20261026"
ARCHIVE = DOWNLOAD / "57418784/primary/repo"
INDEX_ROOT = ARCHIVE / "artifacts/classifier-core-heldout-bf16"
NATIVE = ROOT / "artifacts/local-post-training/20261026-retry-af8b8078/results"
ONNX = ROOT / "artifacts/local-post-training/20261026-onnx-repair-verified/results"
MANIFEST = ROOT / "data/processed/hwdb1_0-1_1_official_five_splits/manifest.csv"
ICDAR = ROOT / "data/processed/icdar2013-hwdb1_0-1_1_four_splits/manifest.csv"
CLOUD = ARCHIVE / "data/processed/vast-primary-official/manifest.csv"
COUNTS = {"test": 533675, "icdar": 224419}
FIELDS = ["sample_id", "source_file", "record_offset", "writer_id", "raw_label",
          "unicode_label", "class_id", "width", "height", "split", "pixel_sha256",
          "source_version", "official_partition", "source_writer_id", "dataset_protocol"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def canonical(row):
    result = dict(row)
    if row["source_file"].startswith(("HWDB1.0_", "HWDB1.1_")):
        partition = "Train" if row["official_partition"] == "train" else "Test"
        result["source_file"] = f"HWDB1.0-1.1_{partition}/" + Path(row["source_file"]).name
        result["sample_id"] = hashlib.sha256((result["source_file"] + "\0" + row["record_offset"]).encode()).hexdigest()
    if result["source_file"].startswith("ICDAR2013_Test/"):
        result.update(source_version="ICDAR2013", official_partition="icdar",
                      source_writer_id=row["writer_id"], dataset_protocol="hwdb1.0-1.1-official")
    expected = hashlib.sha256((result["source_file"] + "\0" + result["record_offset"]).encode()).hexdigest()
    if expected != result["sample_id"]:
        raise ValueError("source/offset identity mismatch")
    return result


def runtime():
    import torch
    import onnxruntime as ort
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": str(torch.__version__), "onnxruntime": ort.__version__,
            "numpy": importlib.metadata.version("numpy"), "pillow": importlib.metadata.version("pillow"),
            "torchvision": importlib.metadata.version("torchvision"), "quantized_engine": torch.backends.quantized.engine,
            "cpu_threads": 4, "device": "cpu", "batch_size": 32,
            "torch_mkldnn": torch.backends.mkldnn.enabled,
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision()}


@contextmanager
def owner(output):
    import msvcrt
    path = output / "ownership.lock"
    stream = path.open("a+b")
    try:
        stream.seek(0)
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        atomic(output / "owner.json", {"pid": os.getpid(), "started": time.time(), "lock": "Windows byte-range exclusive; released by OS after process exit"})
        yield
    finally:
        stream.close()


def freeze(output):
    if (output / "protocol.json").exists():
        raise FileExistsError("Existing protocol is immutable; run it or choose a fresh output")
    current = runtime()
    if not current["torch"].startswith("2.13.") or current["onnxruntime"] != "1.31.0":
        raise ValueError("Requested local runtime versions are unavailable")
    inputs = [Path(__file__), MANIFEST, ICDAR, CLOUD, MANIFEST.with_name("labels.json"),
              ICDAR.with_name("labels.json"), NATIVE / "deployment-artifacts.json", ONNX / "input-inventory.json",
              DOWNLOAD / "consolidated/heldout-results.csv", INDEX_ROOT / "protocol.json"]
    source_files = ["models/artifacts.py", "models/efficient_hccr.py", "models/heads.py",
                    "benchmarking/quantization.py", "preprocessing/pipeline.py", "data/gnt.py", "data/lmdb_store.py"]
    inputs += [ROOT / "src/hccr" / p for p in source_files]
    inventory = read(ONNX / "input-inventory.json")["runs"]
    selected = [r for r in inventory if r["group"] == "final_core" and r["variant"] == "dense_reference"]
    if sorted(int(r["seed"]) for r in selected) != [7, 17, 29]:
        raise ValueError("Frozen three-seed source inventory missing")
    receipts = read(NATIVE / "deployment-artifacts.json")
    artifacts = []
    for row in selected:
        seed, run = int(row["seed"]), Path(row["run"])
        checkpoint = run / "checkpoint.pt"
        if sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from frozen inventory")
        metadata = read(run / "checkpoint_metadata.json")
        if metadata["preprocess"] != {"image_size": 96, "margin": 4}:
            raise ValueError(f"Unexpected source preprocessing: {metadata['preprocess']}")
        inputs += [checkpoint, run / "checkpoint_metadata.json", run / "labels.json"]
        artifacts.append({"seed": seed, "family": "native_fp32", "path": str(checkpoint),
                          "run": str(run), "sha256": row["checkpoint_sha256"], "engine": "PyTorch CPU FP32"})
        head = next(r for r in receipts if r["model"] == f"seed-{seed}-dense_reference" and r["kind"] == "dense_head_int8")
        if sha(head["file"]) != head["sha256"]:
            raise ValueError("INT8 deployment differs from frozen export receipt")
        artifacts.append({"seed": seed, "family": "native_head_int8", "path": head["file"], "sha256": head["sha256"],
                          "run": str(run), "engine": "PyTorch CPU dynamic INT8 dense head; unchanged FP32 backbone"})
        report_path = ONNX / f"onnx/seed-{seed}-dense_reference-report.json"
        inputs.append(report_path)
        report = read(report_path)
        for family, suffix, kind, policy in [("onnx_fp32", "fp32", "onnx_fp32", "ORT_DISABLE_ALL"),
                                            ("onnx_mixed_int8", "mixed-int8", "onnx_mixed_int8", "ORT_ENABLE_ALL")]:
            record = next(r for r in report["models"] if r["kind"] == kind)
            path = ONNX / f"onnx/seed-{seed}-dense_reference-{suffix}.onnx"
            if sha(path) != record["sha256"] or record["graph_optimization_level"] != policy:
                raise ValueError("ONNX differs from frozen export receipt or runtime policy")
            artifacts.append({"seed": seed, "family": family, "path": str(path), "sha256": record["sha256"],
                              "engine": "ONNX Runtime CPUExecutionProvider", "optimization": policy})
        for endpoint in COUNTS:
            metric = Path(row[endpoint + "_metrics"])
            saved = read(metric)
            index = INDEX_ROOT / f"{endpoint}-index-gpu-0.csv"
            if sha(index) != saved["sample_index_sha256"]:
                raise ValueError("Archived endpoint index does not match inference receipt")
            inputs += [metric, metric.with_name("predictions.npz"), index]
    inputs = sorted(set(inputs))
    hashes = {str(path): sha(path) for path in inputs}
    protocol = {"schema_version": 1, "frozen_at_unix": time.time(), "training": False,
                "heldout_pixels_read_before_freeze": 0, "artifacts": artifacts, "runtime": current,
                "inputs_sha256": hashes, "counts": COUNTS,
                "preprocess": {"image_size": 96, "margin": 4, "scaling": "torch uint8 to float32 then multiply by 1/255"},
                "order": "exact archived endpoint CSV order; SHA256 newline-joined sample IDs",
                "reconciliation": "all canonical source/offset, label, dimensions, native pixel SHA, version and writer fields equal archived full manifest; native decoded pixels rehashed",
                "native_pair_cache": "per-batch immutable FP32 embedding reused only after exact nonclassifier state and direct output checks; no persistent feature cache",
                "metrics": ["top1", "top5", "macro_recall_present_classes", "class_support", "class_recall", "by_source_version", "mean_confidence", "nll", "ece15"],
                "confidence_intervals": "No ICDAR writer intervals; no new inferential claim",
                "no_tuning": True, "not_a_latency_measurement": True, "all_24_passes_required": True}
    protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    atomic(output / "protocol.json", protocol)
    atomic(output / "protocol-sha256.json", {"canonical_json_sha256": protocol_digest, "file_sha256": sha(output / "protocol.json")})
    print(f"Frozen 12 artifacts, 24 full endpoint passes: {protocol_digest}", flush=True)


def verified_protocol(output):
    protocol = read(output / "protocol.json")
    if sha(output / "protocol.json") != read(output / "protocol-sha256.json")["file_sha256"]:
        raise ValueError("Frozen protocol was modified")
    if runtime() != protocol["runtime"]:
        raise ValueError("Runtime differs from frozen protocol")
    for path, expected in protocol["inputs_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen source/input changed: {path}")
    for artifact in protocol["artifacts"]:
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Frozen deployment artifact changed")
    return protocol


def status(output, **payload):
    payload.update(pid=os.getpid(), updated_unix=time.time())
    atomic(output / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def reconcile(output, protocol):
    """Only metadata reads. SQLite keeps 758094 records out of Python RAM."""
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    receipt = cache / "reconciliation.json"
    if receipt.exists():
        saved = read(receipt)
        for endpoint in COUNTS:
            if sha(cache / f"{endpoint}-rows.csv") != saved[endpoint]["rows_sha256"]:
                raise ValueError("Reconciled row cache changed")
        return saved
    database = cache / "metadata.sqlite"
    # A partial metadata join is disposable, while completed receipts are immutable.
    database.unlink(missing_ok=True)
    conn = sqlite3.connect(database)
    conn.execute("PRAGMA cache_size=-32768")
    conn.execute("CREATE TABLE samples (id TEXT PRIMARY KEY, local_id TEXT, data TEXT, verified INTEGER DEFAULT 0, seen INTEGER DEFAULT 0)")
    inserted = 0
    for manifest, endpoint in [(MANIFEST, "test"), (ICDAR, "icdar")]:
        for row in rows(manifest):
            if row["split"] != endpoint:
                continue
            mapped = canonical(row)
            conn.execute("INSERT INTO samples(id,local_id,data) VALUES (?,?,?)",
                         (mapped["sample_id"], row["sample_id"], json.dumps(mapped, ensure_ascii=False, sort_keys=True)))
            inserted += 1
            if inserted % 10000 == 0:
                conn.commit()
                status(output, status="reconciling_metadata", local_records=inserted)
    conn.commit()
    matched = 0
    for row in rows(CLOUD):
        if row["split"] not in COUNTS:
            continue
        found = conn.execute("SELECT data,verified FROM samples WHERE id=?", (row["sample_id"],)).fetchone()
        if found is None or found[1]:
            raise ValueError("Missing or duplicated canonical archived source/offset identity")
        local = json.loads(found[0])
        for field in FIELDS:
            if local[field] != row[field]:
                raise ValueError(f"Native/archive metadata mismatch {row['sample_id']} {field}: {local[field]} != {row[field]}")
        conn.execute("UPDATE samples SET verified=1 WHERE id=?", (row["sample_id"],))
        matched += 1
        if matched % 10000 == 0:
            conn.commit()
            status(output, status="reconciling_metadata", local_records=inserted, archived_matches=matched)
    conn.commit()
    if inserted != sum(COUNTS.values()) or matched != inserted:
        raise ValueError("Full metadata reconciliation count mismatch")
    receipt_data = {"protocol_file_sha256": sha(output / "protocol.json"), "metadata_records": inserted}
    import numpy as np
    for endpoint, count in COUNTS.items():
        identity = hashlib.sha256()
        count_found = 0
        targets = np.lib.format.open_memmap(cache / f"{endpoint}-targets.npy", mode="w+", dtype="int64", shape=(count,))
        with (cache / f"{endpoint}-rows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS + ["local_sample_id"])
            writer.writeheader()
            for i, index in enumerate(rows(INDEX_ROOT / f"{endpoint}-index-gpu-0.csv")):
                found = conn.execute("SELECT local_id,data,verified,seen FROM samples WHERE id=?", (index["sample_id"],)).fetchone()
                if found is None or found[2] != 1 or found[3] != 0:
                    raise ValueError("Index record absent, unverified, or duplicated")
                local_id, encoded = found[:2]
                row = json.loads(encoded)
                if row["split"] != endpoint or any(index[k] != row[k] for k in index):
                    raise ValueError("Archived endpoint index metadata mismatch")
                if i:
                    identity.update(b"\n")
                identity.update(row["sample_id"].encode())
                targets[i] = int(row["class_id"])
                writer.writerow({**row, "local_sample_id": local_id})
                conn.execute("UPDATE samples SET seen=1 WHERE id=?", (row["sample_id"],))
                count_found += 1
        targets.flush()
        if count_found != count:
            raise ValueError("Endpoint count mismatch")
        for artifact in protocol["artifacts"]:
            if artifact["family"] != "native_fp32":
                continue
            source = next(r for r in read(ONNX / "input-inventory.json")["runs"] if r["group"] == "final_core" and r["variant"] == "dense_reference" and int(r["seed"]) == artifact["seed"])
            metric = read(source[endpoint + "_metrics"])
            if metric["sample_identity_sha256"] != identity.hexdigest():
                raise ValueError("Ordered image identity differs from archived heldout evaluation")
            with np.load(Path(source[endpoint + "_metrics"]).with_name("predictions.npz"), allow_pickle=False) as predictions:
                if not np.array_equal(predictions["targets"], targets):
                    raise ValueError("Targets differ from archived heldout predictions")
        receipt_data[endpoint] = {"samples": count, "sample_identity_sha256": identity.hexdigest(),
                                  "rows_sha256": sha(cache / f"{endpoint}-rows.csv"),
                                  "targets_sha256": sha(cache / f"{endpoint}-targets.npy"),
                                  "writer_metadata_equal": True, "writer_CI_permitted": endpoint == "test"}
        conn.commit()
    if conn.execute("SELECT COUNT(*) FROM samples WHERE verified != 1 OR seen != 1").fetchone()[0]:
        raise ValueError("Unmatched endpoint record")
    conn.close()
    atomic(receipt, receipt_data)
    return receipt_data


def prepare_cache(output, endpoint, protocol, reconciliation):
    import numpy as np
    import torch
    from torchvision.transforms.v2 import functional as tvf
    from hccr.data.lmdb_store import LMDBImageStore
    from hccr.data.gnt import load_source_image
    from hccr.preprocessing import EvalPreprocessor
    cache = output / "cache"
    path = cache / f"{endpoint}-uint8.npy"
    receipt_path = cache / f"{endpoint}-cache.json"
    progress_path = cache / f"{endpoint}-progress.json"
    count = COUNTS[endpoint]
    if receipt_path.exists():
        receipt = read(receipt_path)
        if sha(path) != receipt["sha256"]:
            raise ValueError("Completed uint8 cache changed")
        return receipt
    progress = read(progress_path) if progress_path.exists() else {"completed": 0, "blocks": []}
    mode = "r+" if path.exists() else "w+"
    images = np.lib.format.open_memmap(path, mode=mode, dtype="uint8", shape=(count, 1, 96, 96))
    for block in progress["blocks"]:
        if hashlib.sha256(images[block["start"]:block["end"]].tobytes()).hexdigest() != block["sha256"]:
            raise ValueError("Partial cache block changed")
    transform = EvalPreprocessor(96, 4)
    store = LMDBImageStore(MANIFEST.with_name("images.lmdb"), protocol["inputs_sha256"][str(MANIFEST)]) if endpoint == "test" else None
    started, initial, last = time.time(), progress["completed"], progress["completed"]
    try:
        for i, row in enumerate(rows(cache / f"{endpoint}-rows.csv")):
            if i < initial:
                continue
            image = store.read_image(row["local_sample_id"]) if store else load_source_image(ROOT / "data/raw", row)
            digest = hashlib.sha256(struct.pack("<HH", image.width, image.height))
            digest.update(image.tobytes())
            if digest.hexdigest() != row["pixel_sha256"] or image.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Native pixels differ from archived manifest: {row['sample_id']}")
            prepared = transform(image)
            images[i, 0] = np.asarray(prepared, dtype="uint8")
            if i == initial:
                reference = tvf.to_dtype(tvf.to_image(prepared), torch.float32, scale=True)
                cached = torch.from_numpy(np.array(images[i], copy=True)).float().mul_(1 / 255)
                torch.testing.assert_close(reference, cached, rtol=0, atol=0)
            if (i + 1) % 4096 == 0 or i + 1 == count:
                images.flush()
                end = i + 1
                progress["blocks"].append({"start": last, "end": end, "sha256": hashlib.sha256(images[last:end].tobytes()).hexdigest()})
                progress["completed"] = end
                atomic(progress_path, progress)
                rate = (end - initial) / max(time.time() - started, 1e-9)
                status(output, status="preparing_cache", endpoint=endpoint, verified_native_pixels=end, total=count,
                       images_per_second=rate, eta_seconds=(count - end) / rate)
                last = end
                # Release mapped working-set pages; keep only one small block resident.
                del images
                images = np.lib.format.open_memmap(path, mode="r+")
        if progress["completed"] != count:
            raise ValueError("Incomplete image cache")
    finally:
        if store:
            store.close()
    del images
    receipt = {"samples": count, "shape": [count, 1, 96, 96], "dtype": "uint8", "sha256": sha(path),
               "all_native_pixels_verified": True, "protocol_file_sha256": sha(output / "protocol.json"),
               "rows_sha256": reconciliation[endpoint]["rows_sha256"], "float_conversion_exact_check": True,
               "source": "native LMDB" if endpoint == "test" else "native source GNT files",
               "not_a_latency_measurement": True}
    atomic(receipt_path, receipt)
    return receipt


def unit_state(output, artifact, endpoint):
    import numpy as np
    directory = output / "evaluations" / f"seed-{artifact['seed']}" / artifact["family"] / endpoint
    directory.mkdir(parents=True, exist_ok=True)
    metrics = directory / "metrics.json"
    if metrics.exists():
        saved = read(metrics)
        for name, expected in saved["outputs_sha256"].items():
            if sha(directory / name) != expected:
                raise ValueError("Completed prediction output changed")
        return {"directory": directory, "done": True, "completed": COUNTS[endpoint]}
    progress = read(directory / "progress.json") if (directory / "progress.json").exists() else {"completed": 0, "blocks": [], "elapsed_seconds": 0}
    arrays = {}
    for name, shape, dtype in [("top5", (COUNTS[endpoint], 5), "int32"), ("confidence", (COUNTS[endpoint],), "float32"), ("nll", (COUNTS[endpoint],), "float32")]:
        path = directory / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=shape)
    for block in progress["blocks"]:
        digest = hashlib.sha256()
        for name in arrays:
            digest.update(arrays[name][block["start"]:block["end"]].tobytes())
        if digest.hexdigest() != block["sha256"]:
            raise ValueError("Partial prediction block changed")
    return {"directory": directory, "done": False, **progress, "arrays": arrays, "last": progress["completed"]}


def save_scores(unit, start, scores, targets):
    import torch
    scores = scores.float().cpu()
    if not torch.isfinite(scores).all() or scores.shape != (len(targets), 3755):
        raise ValueError("Nonfinite or malformed deployment scores")
    stop = start + len(targets)
    unit["arrays"]["top5"][start:stop] = scores.topk(5, dim=1).indices.numpy()
    log_probs = scores.log_softmax(1)
    unit["arrays"]["confidence"][start:stop] = log_probs.max(1).values.exp().numpy()
    unit["arrays"]["nll"][start:stop] = -log_probs[torch.arange(len(targets)), torch.as_tensor(targets)].numpy()
    unit["completed"] = stop


def flush_unit(unit, elapsed):
    digest = hashlib.sha256()
    for name, array in unit["arrays"].items():
        array.flush()
        digest.update(array[unit["last"]:unit["completed"]].tobytes())
    if unit["last"] != unit["completed"]:
        unit["blocks"].append({"start": unit["last"], "end": unit["completed"], "sha256": digest.hexdigest()})
    unit["elapsed_seconds"] += elapsed
    atomic(unit["directory"] / "progress.json", {k: unit[k] for k in ("completed", "blocks", "elapsed_seconds")})
    unit["last"] = unit["completed"]


def finish_unit(output, unit, artifact, endpoint, reconciliation, cache_receipt):
    import numpy as np
    targets = np.load(output / "cache" / f"{endpoint}-targets.npy", mmap_mode="r")
    predictions = unit["arrays"]["top5"]
    correct = predictions[:, 0] == targets
    top5 = np.any(predictions == targets[:, None], axis=1)
    support = np.bincount(targets, minlength=3755)
    hits = np.bincount(targets[correct], minlength=3755)
    recalls = np.divide(hits, support, out=np.zeros(3755), where=support != 0)
    versions = {}
    for i, row in enumerate(rows(output / "cache" / f"{endpoint}-rows.csv")):
        value = versions.setdefault(row["source_version"], {"samples": 0, "top1_hits": 0, "top5_hits": 0})
        value["samples"] += 1
        value["top1_hits"] += int(correct[i])
        value["top5_hits"] += int(top5[i])
    for value in versions.values():
        value["top1"] = value["top1_hits"] / value["samples"]
        value["top5"] = value["top5_hits"] / value["samples"]
    confidence = unit["arrays"]["confidence"]
    ece = 0.0
    for b in range(15):
        selected = (confidence >= b / 15) & ((confidence <= 1) if b == 14 else (confidence < (b + 1) / 15))
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    result = {"status": "completed", "samples": len(targets), "top1": float(correct.mean()), "top5": float(top5.mean()),
              "macro_recall_present_classes": float(recalls[support > 0].mean()), "class_support": support.tolist(),
              "class_recall": recalls.tolist(), "by_source_version": versions,
              "mean_confidence": float(np.mean(confidence, dtype=np.float64)), "nll": float(np.mean(unit["arrays"]["nll"], dtype=np.float64)), "ece15": ece,
              "seed": artifact["seed"], "family": artifact["family"], "split": endpoint,
              "artifact": artifact, "sample_identity_sha256": reconciliation[endpoint]["sample_identity_sha256"],
              "targets_sha256": reconciliation[endpoint]["targets_sha256"], "cache_sha256": cache_receipt["sha256"],
              "protocol_file_sha256": sha(output / "protocol.json"), "runtime": runtime(),
              "elapsed_processing_seconds": unit["elapsed_seconds"], "not_a_latency_measurement": True,
              "outputs_sha256": {f"{name}.npy": sha(unit["directory"] / f"{name}.npy") for name in unit["arrays"]}}
    atomic(unit["directory"] / "metrics.json", result)
    print(f"COMPLETED seed {artifact['seed']} {artifact['family']} {endpoint} n={len(targets)} top1={result['top1']:.8f} top5={result['top5']:.8f}", flush=True)


def pair_identity(source, quantized):
    import torch
    a = {k: v for k, v in source.state_dict().items() if not k.startswith("classifier.")}
    b = {k: v for k, v in quantized.state_dict().items() if not k.startswith("classifier.")}
    if a.keys() != b.keys():
        raise ValueError("Native pair backbone key mismatch")
    digest = hashlib.sha256()
    for key in sorted(a):
        if a[key].dtype != b[key].dtype or not torch.equal(a[key], b[key]):
            raise ValueError("Native pair backbone differs")
        digest.update(key.encode())
        digest.update(str((a[key].dtype, tuple(a[key].shape))).encode())
        digest.update(a[key].reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def infer(output, endpoint, artifacts, reconciliation, cache_receipt, initial_batches=0):
    import numpy as np
    import torch
    import onnxruntime as ort
    from hccr.models.artifacts import load_model_from_run
    from hccr.benchmarking.quantization import load_classifier_deployment
    images = np.load(output / "cache" / f"{endpoint}-uint8.npy", mmap_mode="r")
    targets = np.load(output / "cache" / f"{endpoint}-targets.npy", mmap_mode="r")
    units = [(artifact, unit_state(output, artifact, endpoint)) for artifact in artifacts]
    units = [(a, u) for a, u in units if not u["done"]]
    if not units:
        return
    session = source = quantized = None
    if artifacts[0]["family"].startswith("native"):
        native_artifact = next(a for a in artifacts if a["family"] == "native_fp32")
        int8_artifact = next(a for a in artifacts if a["family"] == "native_head_int8")
        source = load_model_from_run(Path(native_artifact["run"]), "cpu").float().eval()
        quantized = load_classifier_deployment(Path(int8_artifact["path"])).eval()
        backbone_hash = pair_identity(source, quantized)
    else:
        artifact = artifacts[0]
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, artifact["optimization"])
        session = ort.InferenceSession(artifact["path"], sess_options=options, providers=["CPUExecutionProvider"])
    start_at = min(u["completed"] for _, u in units)
    clock = last_flush = time.time()
    initial = start_at
    proof_written = False
    with torch.inference_mode():
        for start in range(start_at, len(images), 32):
            stop = min(start + 32, len(images))
            image = torch.from_numpy(np.array(images[start:stop], copy=True)).float().mul_(1 / 255)
            if source is not None:
                embeddings = source._forward_embedding(image)
                scores = {"native_fp32": source.classifier(embeddings), "native_head_int8": quantized.classifier(embeddings)}
                if not proof_written:
                    torch.testing.assert_close(scores["native_fp32"], source(image), rtol=0, atol=0)
                    torch.testing.assert_close(embeddings, quantized._forward_embedding(image), rtol=0, atol=0)
                    torch.testing.assert_close(scores["native_head_int8"], quantized(image), rtol=0, atol=0)
                    proof_path = output / "native-pair-proofs" / f"seed-{artifacts[0]['seed']}-{endpoint}-{start}.json"
                    proof_path.parent.mkdir(exist_ok=True)
                    if not proof_path.exists():
                        atomic(proof_path, {"backbone_sha256": backbone_hash, "exact_state_match": True,
                            "exact_embedding_output_match": True, "exact_direct_model_score_match": True,
                            "samples": len(image), "batch_start": start, "immutable_embedding_reuse": True,
                            "not_a_latency_measurement": True})
                    proof_written = True
            else:
                result = session.run(None, {session.get_inputs()[0].name: image.numpy()})[0]
                scores = {artifacts[0]["family"]: torch.from_numpy(result)}
            for artifact, unit in units:
                if start >= unit["completed"]:
                    save_scores(unit, start, scores[artifact["family"]], targets[start:stop])
            if (start - initial) // 32 < 3:
                print(f"INITIAL_BATCH {endpoint} seed={artifacts[0]['seed']} families={[a['family'] for a in artifacts]} rows={stop}", flush=True)
            if stop % 2048 == 0 or stop == len(images) or initial_batches and stop >= initial + initial_batches * 32:
                now = time.time()
                for _, unit in units:
                    flush_unit(unit, now - last_flush)
                rate = (stop - initial) / max(now - clock, 1e-9)
                status(output, status="evaluating", endpoint=endpoint, seed=artifacts[0]["seed"],
                       families=[a["family"] for a, _ in units], completed=stop, total=len(images),
                       endpoint_images_per_second=rate, eta_seconds=(len(images) - stop) / rate,
                       not_a_latency_measurement=True)
                last_flush = now
                del images
                images = np.load(output / "cache" / f"{endpoint}-uint8.npy", mmap_mode="r")
            if initial_batches and stop >= initial + initial_batches * 32:
                return
    for artifact, unit in units:
        if unit["completed"] != len(images):
            raise ValueError("Incomplete inference output")
        finish_unit(output, unit, artifact, endpoint, reconciliation, cache_receipt)


def run(output, initial_batches=0):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol = verified_protocol(output)
    reconciliation = reconcile(output, protocol)
    caches = {}
    for endpoint in COUNTS:
        caches[endpoint] = prepare_cache(output, endpoint, protocol, reconciliation)
    atomic(output / "input-ready.json", {"reconciliation": reconciliation, "caches": caches, "runtime": runtime()})
    failures = []
    for seed in (7, 17, 29):
        by_family = {a["family"]: a for a in protocol["artifacts"] if a["seed"] == seed}
        groups = [[by_family["native_fp32"], by_family["native_head_int8"]], [by_family["onnx_fp32"]], [by_family["onnx_mixed_int8"]]]
        for group in groups:
            for endpoint in COUNTS:
                try:
                    infer(output, endpoint, group, reconciliation, caches[endpoint], initial_batches)
                except Exception as exc:
                    failure = {"seed": seed, "families": [a["family"] for a in group], "endpoint": endpoint,
                               "error": str(exc), "traceback": traceback.format_exc(), "time": time.time()}
                    failures.append(failure)
                    atomic(output / "failures" / f"{time.time_ns()}.json", failure)
                    traceback.print_exc()
    completed = []
    for path in sorted((output / "evaluations").glob("*/*/*/metrics.json")):
        saved = read(path)
        completed.append({"seed": saved["seed"], "family": saved["family"], "endpoint": saved["split"],
                          "samples": saved["samples"], "top1": saved["top1"], "top5": saved["top5"], "metrics": str(path)})
    atomic(output / "summary.json", {"completed_passes": len(completed), "required_passes": 24, "results": completed,
                                     "failures_this_invocation": failures, "initial_batches_only": initial_batches})
    status(output, status="initial_batches_completed" if initial_batches else "completed" if len(completed) == 24 else "incomplete",
           completed_passes=len(completed), required_passes=24, failures=len(failures))
    return 0 if initial_batches and not failures or len(completed) == 24 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--initial-batches", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with owner(args.output):
        try:
            if args.command == "freeze":
                freeze(args.output)
                return 0
            return run(args.output, args.initial_batches)
        except Exception as exc:
            failure = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            atomic(args.output / "failures" / f"{time.time_ns()}.json", failure)
            status(args.output, **failure)
            traceback.print_exc()
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
