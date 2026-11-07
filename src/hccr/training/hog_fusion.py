"""One frozen-backbone HOG fusion pilot with paired validation evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.v2 import functional as VF

from hccr.artifacts.checkpoints import file_digest
from hccr.data.lmdb_store import LMDBImageStore
from hccr.models.artifacts import load_model_from_run, read_checkpoint_metadata
from hccr.models.hog_fusion import HOGCNNFusion, HOGConfig, load_hog_fusion_run
from hccr.preprocessing.pipeline import EvalPreprocessor


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class FusionImages(Dataset):
    """Stream manifest metadata and retain only one split's compact identities."""

    def __init__(self, manifest, split, labels, preprocess, manifest_sha):
        if split not in {"train", "validation"}:
            raise ValueError("the pilot only opens train and validation images")
        self.ids, targets, self.writers = [], [], []
        writer_pool = {}
        with Path(manifest).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    continue
                target = int(row["class_id"])
                if labels.get(str(target)) != row["unicode_label"]:
                    raise ValueError("manifest class order differs from the checkpoint")
                writer = row["writer_id"].strip()
                if not writer:
                    raise ValueError("writer IDs are required for this pilot")
                self.ids.append(row["sample_id"])
                self.writers.append(writer_pool.setdefault(writer, writer))
                targets.append(target)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("split must contain nonempty unique sample IDs")
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = EvalPreprocessor(**preprocess)
        self.store = LMDBImageStore(
            Path(manifest).with_name("images.lmdb"), manifest_sha
        )
        _ = self.store.metadata
        self.store.close()
        digest = hashlib.sha256()
        for sample_id in self.ids:
            digest.update(sample_id.encode() + b"\n")
        self.ordered_ids_sha256 = digest.hexdigest()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = self.transform(self.store.read_image(self.ids[index]))
        tensor = VF.to_dtype(VF.to_image(image), torch.float32, scale=True)
        return tensor, int(self.targets[index])


def extract_cache(model, dataset, folder, identity, args, report):
    folder.mkdir(parents=True, exist_ok=True)
    identity = {
        **identity,
        "ordered_ids_sha256": dataset.ordered_ids_sha256,
        "samples": len(dataset),
        "dtype": "float32",
    }
    receipt = folder / "metadata.json"
    prior = json.loads(receipt.read_text("utf-8")) if receipt.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("feature-cache provenance changed; use a new output directory")
    mode = "r+" if prior else "w+"
    arrays = {
        name: np.lib.format.open_memmap(
            folder / f"{name}.npy",
            mode=mode,
            dtype=np.float32,
            shape=(len(dataset), dimension),
        )
        for name, dimension in (("cnn", 320), ("hog", model.hog.config.dimension))
    }
    start = int(prior["completed_rows"]) if prior else 0
    if not 0 <= start <= len(dataset):
        raise ValueError("invalid feature-cache progress")
    if prior and prior["complete"]:
        for name, array in arrays.items():
            if file_digest(folder / f"{name}.npy") != prior["sha256"][name]:
                raise ValueError("completed feature-cache checksum differs")
            array.flags.writeable = False
        return arrays
    progress = {"identity": identity, "completed_rows": start, "complete": False}
    write_json(receipt, progress)
    loader = DataLoader(
        Subset(dataset, range(start, len(dataset))),
        batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        **({"multiprocessing_context": "spawn"} if args.num_workers else {}),
    )
    model.eval()
    offset = start
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader):
            cnn, hog = model.extract(images.to(args.device, non_blocking=True))
            end = offset + len(images)
            arrays["cnn"][offset:end] = cnn.cpu().numpy()
            arrays["hog"][offset:end] = hog.cpu().numpy()
            offset = end
            if batch_index % 100 == 0 or offset == len(dataset):
                for array in arrays.values():
                    array.flush()
                progress["completed_rows"] = offset
                write_json(receipt, progress)
                report(
                    "extract", split=folder.name, completed=offset, total=len(dataset)
                )
    progress.update(
        complete=True,
        sha256={name: file_digest(folder / f"{name}.npy") for name in arrays},
    )
    write_json(receipt, progress)
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


def cached_batches(arrays, targets, device, batch_size, *, rng=None):
    """Shuffle bounded contiguous disk blocks, then rows within each block."""
    block_rows = 4096
    starts = np.arange(0, len(targets), block_rows)
    if rng is not None:
        rng.shuffle(starts)
    for start in starts:
        stop = min(int(start) + block_rows, len(targets))
        tensors = [
            torch.from_numpy(np.array(arrays[name][start:stop], copy=True)).to(device)
            for name in ("cnn", "hog")
        ]
        labels = torch.from_numpy(np.array(targets[start:stop], copy=True)).to(device)
        order = np.arange(stop - start)
        if rng is not None:
            rng.shuffle(order)
        order = torch.as_tensor(order, device=device)
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            yield *(tensor[indices] for tensor in tensors), labels[indices]


def evaluate_pair(model, arrays, targets, device, batch_size, *, ablate_hog=False):
    baseline, fusion, losses = [], [], 0.0
    model.eval()
    with torch.inference_mode():
        for cnn, hog, labels in cached_batches(arrays, targets, device, batch_size):
            base_logits = model.cnn.classifier(cnn)
            if ablate_hog:
                embedding = model.fusion(
                    torch.cat((cnn, cnn.new_zeros((len(cnn), model.hog_dim))), dim=1)
                )
                logits = model.cnn.classifier(embedding)
            else:
                logits = model.forward_cached(cnn, hog)
            baseline.append(base_logits.argmax(1).cpu().numpy())
            fusion.append(logits.argmax(1).cpu().numpy())
            losses += F.cross_entropy(logits, labels, reduction="sum").item()
    baseline, fusion = np.concatenate(baseline), np.concatenate(fusion)
    base_ok, fused_ok = baseline == targets, fusion == targets
    repairs = int((~base_ok & fused_ok).sum())
    damage = int((base_ok & ~fused_ok).sum())
    return (
        {
            "samples": len(targets),
            "baseline_top1": float(base_ok.mean()),
            "fusion_top1": float(fused_ok.mean()),
            "delta_pp": 100 * (repairs - damage) / len(targets),
            "repairs": repairs,
            "damage": damage,
            "prediction_changes": int((baseline != fusion).sum()),
            "fusion_cross_entropy": losses / len(targets),
        },
        baseline,
        fusion,
    )


def run_hog_fusion(args) -> dict:
    source, manifest, output = (
        Path(args.source_run).resolve(),
        Path(args.manifest).resolve(),
        Path(args.output_dir).resolve(),
    )
    if output.exists() and not args.resume:
        raise FileExistsError("output already exists; use --resume for the same run")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def report(stage, **values):
        status = {"stage": stage, "elapsed_seconds": time.time() - started, **values}
        write_json(output / "status.json", status)
        print(json.dumps(status), flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    metadata = read_checkpoint_metadata(source)
    if metadata["model"].get("num_classes") != 3755:
        raise ValueError("this pilot requires all 3,755 official classes")
    labels_payload = json.loads((source / "labels.json").read_text("utf-8"))
    local_labels = manifest.with_name("labels.json")
    if json.loads(local_labels.read_text("utf-8")) != labels_payload:
        raise ValueError("local and checkpoint label mappings differ")
    precision = metadata["training"]["precision"]["resolved"]
    preprocess = metadata["preprocess"]
    cnn = load_model_from_run(source, device=args.device)
    model = HOGCNNFusion(
        cnn,
        HOGConfig(image_size=preprocess["image_size"]),
        backbone_precision=precision,
    ).to(args.device)
    frozen_before = {
        name: value.detach().cpu().clone() for name, value in cnn.state_dict().items()
    }
    identity = {
        "source_checkpoint_sha256": file_digest(source / "checkpoint.pt"),
        "source_metadata_sha256": file_digest(source / "checkpoint_metadata.json"),
        "source_manifest_sha256": metadata.get("manifest_digest"),
        "manifest_sha256": file_digest(manifest),
        "labels_sha256": file_digest(source / "labels.json"),
        "preprocess": preprocess,
        "model": model.specification(),
        "model_code_sha256": file_digest(
            Path(__file__).parents[1] / "models/hog_fusion.py"
        ),
        "runner_code_sha256": file_digest(Path(__file__)),
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else None
        ),
        "augmentation": "none_deterministic_eval",
        "cache_precision": "float32",
        "fusion_precision": "float32",
    }
    run_identity = output / "identity.json"
    if (
        run_identity.exists()
        and json.loads(run_identity.read_text("utf-8")) != identity
    ):
        raise ValueError("run identity differs; do not resume into different settings")
    write_json(run_identity, identity)
    write_json(output / "labels.json", labels_payload)
    report("read_manifest")
    datasets = {
        split: FusionImages(
            manifest,
            split,
            labels_payload["labels"],
            preprocess,
            identity["manifest_sha256"],
        )
        for split in ("train", "validation")
    }
    if set(datasets["train"].writers) & set(datasets["validation"].writers):
        raise ValueError("train and validation writers overlap")
    support = {
        split: {
            "samples": len(dataset),
            "writers": len(set(dataset.writers)),
            "classes": len(np.unique(dataset.targets)),
            "ordered_ids_sha256": dataset.ordered_ids_sha256,
        }
        for split, dataset in datasets.items()
    }
    write_json(output / "support.json", support)
    validation = datasets["validation"]
    caches = {
        "validation": extract_cache(
            model, validation, output / "cache/validation", identity, args, report
        )
    }
    initial, _, _ = evaluate_pair(
        model, caches["validation"], validation.targets, args.device, args.batch_size
    )
    if initial["prediction_changes"] != 0:
        raise RuntimeError("identity-initialized fusion did not preserve baseline")
    write_json(output / "initial_metrics.json", initial)
    if initial["baseline_top1"] < args.min_baseline_top1:
        raise RuntimeError("local baseline accuracy is below the declared pilot guard")
    caches["train"] = extract_cache(
        model, datasets["train"], output / "cache/train", identity, args, report
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.hog_projection.parameters(), "weight_decay": 1e-4},
            {"params": model.hog_normalization.parameters(), "weight_decay": 0.0},
            {"params": model.fusion.parameters(), "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    artifact_metadata = {
        "schema_version": 1,
        "model": {**model.specification(), "cnn": metadata["model"]},
        "preprocess": preprocess,
        "source_run": str(source),
        "identity": identity,
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "optimizer": "adamw",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "whole_model_parameters": sum(p.numel() for p in model.parameters()),
            "support": support,
        },
        "test": {"status": "not_run"},
        "icdar": {"status": "not_run"},
    }
    best = (initial["fusion_top1"], -initial["fusion_cross_entropy"])
    best_epoch, first_epoch, curves = 0, 1, []

    def save_selected(epoch, scores):
        torch.save(model.state_dict(), output / "checkpoint.pt.tmp")
        os.replace(output / "checkpoint.pt.tmp", output / "checkpoint.pt")
        artifact_metadata["selection"] = {
            "epoch": epoch,
            "split": "validation",
            "metrics": scores,
            "policy": "max_top1_then_min_cross_entropy_including_epoch_zero",
        }
        write_json(output / "checkpoint_metadata.json", artifact_metadata)

    latest = output / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_epoch = state["epoch"] + 1
        best, best_epoch, curves = (
            tuple(state["best"]),
            state["best_epoch"],
            state["curves"],
        )
    else:
        save_selected(0, initial)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        rng = np.random.default_rng(args.seed + epoch)
        for cnn_features, hog, targets in cached_batches(
            caches["train"],
            datasets["train"].targets,
            args.device,
            args.batch_size,
            rng=rng,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_cached(cnn_features, hog, targets)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite fusion training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += loss.item() * len(targets)
            seen += len(targets)
            if seen % (args.batch_size * 200) < args.batch_size:
                report(
                    "train", epoch=epoch, completed=seen, total=len(datasets["train"])
                )
        scores, _, _ = evaluate_pair(
            model,
            caches["validation"],
            validation.targets,
            args.device,
            args.batch_size,
        )
        candidate = (scores["fusion_top1"], -scores["fusion_cross_entropy"])
        if candidate > best:
            best, best_epoch = candidate, epoch
            save_selected(epoch, scores)
        curves.append({"epoch": epoch, "train_loss": total_loss / seen, **scores})
        write_json(output / "curves.json", {"epochs": curves})
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "best_epoch": best_epoch,
                "curves": curves,
            },
            output / "latest.pt.tmp",
        )
        os.replace(output / "latest.pt.tmp", latest)
        report("epoch_complete", epoch=epoch, best_epoch=best_epoch, **scores)
    if any(
        not torch.equal(frozen_before[name], value.detach().cpu())
        for name, value in model.cnn.state_dict().items()
    ):
        raise RuntimeError("the frozen CNN or classifier changed")
    selected = load_hog_fusion_run(output, args.device)
    scores, baseline, fused = evaluate_pair(
        selected,
        caches["validation"],
        validation.targets,
        args.device,
        args.batch_size,
    )
    without_hog, _, _ = evaluate_pair(
        selected,
        caches["validation"],
        validation.targets,
        args.device,
        args.batch_size,
        ablate_hog=True,
    )
    write_json(
        output / "hog_ablation.json",
        {
            "intervention": "zero_projected_hog_keep_learned_cnn_projection",
            "natural": {key: value for key, value in scores.items()},
            "without_hog": without_hog,
            "natural_minus_without_hog_pp": 100
            * (scores["fusion_top1"] - without_hog["fusion_top1"]),
            "limitation": "post_training_intervention_may_be_out_of_distribution",
        },
    )
    np.savez_compressed(
        output / "validation_predictions.npz",
        targets=validation.targets,
        baseline=baseline,
        fusion=fused,
        sample_ids=np.asarray(validation.ids),
        writers=np.asarray(validation.writers),
    )
    writer_scores = {}
    writer_array = np.asarray(validation.writers)
    for writer in sorted(set(validation.writers)):
        mask = writer_array == writer
        writer_scores[writer] = {
            "samples": int(mask.sum()),
            "repairs": int(
                (
                    (baseline != validation.targets)
                    & (fused == validation.targets)
                    & mask
                ).sum()
            ),
            "damage": int(
                (
                    (baseline == validation.targets)
                    & (fused != validation.targets)
                    & mask
                ).sum()
            ),
        }
    scores.update(
        selected_epoch=best_epoch,
        frozen_state_unchanged=True,
        interpretation="single_seed_validation_pilot_not_independent_test_evidence",
        test_status="not_run",
        icdar_status="not_run",
        writers=writer_scores,
    )
    write_json(output / "metrics.json", scores)
    report(
        "completed",
        selected_epoch=best_epoch,
        **{
            key: value
            for key, value in scores.items()
            if key != "writers" and key != "selected_epoch"
        },
    )
    for split, dataset in datasets.items():
        dataset.store.close()
        for array in caches[split].values():
            array._mmap.close()
    return scores
