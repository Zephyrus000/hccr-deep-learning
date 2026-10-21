"""Reuse verified development arrays after a storage-only publication repair."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_embedding_ph_study import exclusive_worker
from scripts.run_embedding_ph_study import main as run_study

from hccr.tda.embedding.study import EmbeddingStudy, load_config
from hccr.tda.features import (
    digest_file,
    load_cache,
    publish_cache,
    signature,
    write_json,
)


def verify_storage_patch(snapshot: Path, pinned_hash: str) -> dict:
    """Reject numerical changes, allowing only the reviewed publication repair."""
    before = snapshot.read_text(encoding="utf-8")
    hashes = {
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in (before, before.replace("\n", "\r\n"))
    }
    if pinned_hash not in hashes:
        raise ValueError("prior source snapshot does not match pinned implementation")
    current = Path(__file__).parents[1] / "src/hccr/tda/features.py"

    class StorageOnly(ast.NodeTransformer):
        def visit_Import(self, node):
            node.names = [
                a for a in node.names if a.name not in {"logging", "shutil", "time"}
            ]
            return node if node.names else None

        def visit_ImportFrom(self, node):
            if node.module == "tempfile":
                node.names = [a for a in node.names if a.name != "mkdtemp"]
            return node

        def visit_FunctionDef(self, node):
            if node.name == "publish_cache":
                return None
            return self.generic_visit(node)

        def visit_ClassDef(self, node):
            if node.name == "CacheWriter":
                node.body = [
                    child
                    for child in node.body
                    if not isinstance(child, ast.FunctionDef)
                    or child.name not in {"__enter__", "__exit__"}
                ]
            return self.generic_visit(node)

        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "staged"
                and node.func.attr == "rename"
            ):
                return ast.Call(
                    ast.Name("publish_cache", ast.Load()),
                    [node.func.value, *node.args],
                    node.keywords,
                )
            return node

    old = ast.dump(StorageOnly().visit(ast.parse(before)), include_attributes=False)
    new = ast.dump(
        StorageOnly().visit(ast.parse(current.read_text(encoding="utf-8"))),
        include_attributes=False,
    )
    if old != new:
        raise ValueError(
            "not a storage-only change; refuse to migrate feature provenance"
        )
    return {"before": pinned_hash, "after": digest_file(current)}


def reuse_development_caches(prior: Path, config, snapshot: Path) -> dict:
    prior_study = json.loads((prior / "seed-7/study.json").read_text(encoding="utf-8"))
    if signature(prior_study["identity"]) != prior_study["fingerprint"]:
        raise ValueError("invalid prior study fingerprint")
    expected = copy.deepcopy(prior_study["identity"])
    source_hashes = expected["implementation"]["shared"]["source"]
    repair = verify_storage_patch(snapshot, source_hashes["tda/features.py"])
    source_hashes["tda/features.py"] = repair["after"]
    with EmbeddingStudy(config) as study:
        if signature(study.identity) != signature(expected):
            raise ValueError(
                "source, scientific config or other implementation changed; "
                "refuse cache reuse"
            )
        old_bank_path = prior / "seed-7/bank"
        old_bank = json.loads(
            (old_bank_path / "metadata.json").read_text(encoding="utf-8")
        )["identity"]
        if (
            old_bank["study"] != prior_study["fingerprint"]
            or old_bank["fit_split"] != "train"
        ):
            raise ValueError("reference bank is not pinned to prior TRAIN study")
        new_bank = {**old_bank, "study": study.fingerprint}
        inventories = {}
        for relative, old_identity, new_identity in (
            ("bank", old_bank, new_bank),
            (
                "features/train",
                {
                    "study": prior_study["fingerprint"],
                    "source_view": study.train.identity,
                    "bank": signature(old_bank),
                },
                {
                    "study": study.fingerprint,
                    "source_view": study.train.identity,
                    "bank": signature(new_bank),
                },
            ),
        ):
            source, target = prior / "seed-7" / relative, study.output / relative
            with load_cache(source, old_identity) as arrays:
                metadata = json.loads(
                    (source / "metadata.json").read_text(encoding="utf-8")
                )
                inventory = metadata["arrays"]
                for name, spec in inventory.items():
                    if (
                        list(arrays[name].shape) != spec["shape"]
                        or str(arrays[name].dtype) != spec["dtype"]
                    ):
                        raise ValueError(f"invalid array schema: {source / name}")
            if target.exists():
                with load_cache(target, new_identity):
                    existing = json.loads(
                        (target / "metadata.json").read_text(encoding="utf-8")
                    )
                    if existing["arrays"] != inventory:
                        raise ValueError("existing migration contains different arrays")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(prefix=".reuse-", dir=target.parent) as tmp:
                    staged = Path(tmp) / "cache"
                    staged.mkdir()
                    for name in inventory:
                        os.link(source / f"{name}.npy", staged / f"{name}.npy")
                    write_json(
                        staged / "metadata.json",
                        {
                            "identity": new_identity,
                            "signature": signature(new_identity),
                            "arrays": inventory,
                        },
                    )
                    publish_cache(staged, target)
            inventories[relative] = inventory
        report = {
            "prior_run": str(prior.resolve()),
            "prior_fingerprint": prior_study["fingerprint"],
            "fingerprint": study.fingerprint,
            "storage_repair": repair,
            "method": (
                "read-only arrays hard-linked after full SHA256 verification; "
                "new metadata records ancestry"
            ),
            "reused_arrays": inventories,
            "test_or_icdar_opened": False,
        }
        write_json(study.output / "cache_recovery.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-run", required=True, type=Path)
    parser.add_argument("--prior-features-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    prior = args.prior_run.resolve()
    root = args.output_dir.resolve()
    if root == prior or root.is_relative_to(prior) or prior.is_relative_to(root):
        parser.error("recovery requires a separate output directory")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_worker(root / "dispatch.lock"):
        state = {
            "job_id": root.name,
            "status": "running",
            "stage": "verification",
            "worker_pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
            "prior_run": str(prior),
        }

        def notify():
            state["updated_at"] = datetime.now(UTC).isoformat()
            write_json(root / "dispatch_status.json", state)
            print(json.dumps(state), flush=True)

        try:
            notify()
            repository = Path(__file__).resolve().parents[1]
            changed = [
                "src/hccr/tda/features.py",
                "scripts/recover_embedding_ph_cache.py",
                "tests/test_cache_publication.py",
                "tests/test_embedding_recovery.py",
            ]
            commands = [
                ["black", "--check", "--workers", "1", *changed],
                ["ruff", "check", *changed],
                *[
                    ["unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
                    for pattern in (
                        "test_cache_publication.py",
                        "test_embedding*.py",
                        "test_stage2*.py",
                    )
                ],
            ]
            for command in commands:
                subprocess.run(
                    [sys.executable, "-m", *command], cwd=repository, check=True
                )
            state["stage"] = "recover_train_cache"
            state["verification_passed"] = True
            notify()
            config = replace(
                load_config(prior / "resolved_config.json"), output_dir=str(root)
            )
            if config.seed != 7:
                raise ValueError(
                    "this recovery is restricted to the failed seed-7 study"
                )
            resolved = root / "resolved_config.json"
            from dataclasses import asdict

            write_json(resolved, asdict(config))
            reuse_development_caches(prior, config, args.prior_features_source)
            state["stage"] = "full_class_study"
            notify()
            code = run_study(["--config", str(resolved), "--threads", "12", "--resume"])
            if code:
                raise RuntimeError(
                    "full-class worker failed; inspect seed-7/status.json"
                )
            state["status"], state["stage"] = "completed", "completed"
            return 0
        except Exception as error:
            state.update(
                status="failed",
                failed_stage=state["stage"],
                error=f"{type(error).__name__}: {error}",
            )
            traceback.print_exc()
            return 1
        finally:
            notify()


if __name__ == "__main__":
    raise SystemExit(main())
