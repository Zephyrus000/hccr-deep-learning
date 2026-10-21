"""Build and audit the official four-split HWDB manifests in sequence."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from scripts.build_casia_manifest import main as build_manifest

from hccr.data.lmdb_store import manifest_digest
from hccr.tda.features import write_json

JOBS = (
    (
        "icdar2013-hwdb1.0-1.1",
        "icdar2013-hwdb1_0-1_1_four_splits",
        3755,
        False,
    ),
    (
        "icdar2013-hwdb1.0-1.2",
        "icdar2013-hwdb1_0-1_2_four_splits",
        3755,
        False,
    ),
    (
        "hwdb1.0-1.2",
        "hwdb1_0-1_2_four_splits",
        None,
        True,
    ),
)


def verify_published_manifest(
    path: Path, protocol: str, class_count: int | None
) -> dict:
    report = json.loads((path / "audit_report.json").read_text(encoding="utf-8"))
    expected_splits = {"train", "validation", "test", "icdar"}
    if (
        report["protocol"] != protocol
        or report["writer_disjoint_verified"] is not True
        or set(report["split_counts"]) != expected_splits
        or set(report["writer_counts"]) != expected_splits
        or any(count <= 0 for count in report["split_counts"].values())
        or any(report["pairwise_writer_overlap_counts"].values())
        or report["writer_counts"]["icdar"] != 60
        or report["split_counts"]["icdar"] != 224419
        or report["image_overlap_check"]["status"] != "passed"
        or report["image_overlap_check"]["cross_split_duplicates"] != 0
        or report["image_overlap_check"]["checked_samples"]
        != sum(report["split_counts"].values())
        or (class_count is not None and report["class_count"] != class_count)
        or manifest_digest(path / "manifest.csv") != report["manifest_digest"]
        or manifest_digest(path / "class_mapping_full.json")
        != report["label_mapping_digest"]
    ):
        raise ValueError(f"published manifest failed four-split audit: {path}")
    labels = json.loads((path / "class_mapping_full.json").read_text(encoding="utf-8"))
    if len(labels["labels"]) != report["class_count"]:
        raise ValueError(f"class mapping length disagrees with audit: {path}")
    return {
        "output_dir": str(path.resolve()),
        "class_count": report["class_count"],
        "split_counts": report["split_counts"],
        "writer_counts": report["writer_counts"],
        "manifest_digest": report["manifest_digest"],
        "label_mapping_digest": report["label_mapping_digest"],
        "exact_image_overlap": report["image_overlap_check"]["cross_split_duplicates"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path("artifacts/hwdb-mapping-rebuild-20261021"),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(
            json.dumps(
                [
                    {"protocol": protocol, "output_dir": str(args.output_root / folder)}
                    for protocol, folder, _, _ in JOBS
                ],
                indent=2,
            )
        )
        return 0
    args.status_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status_dir / "status.json"
    if status_path.exists():
        raise FileExistsError(f"rebuild already dispatched: {status_path}")
    outputs = [args.output_root / folder for _, folder, _, _ in JOBS]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing manifests: {existing}")
    state = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "jobs": [
            {
                "protocol": protocol,
                "output_dir": str(output.resolve()),
                "status": "pending",
            }
            for (protocol, _, _, _), output in zip(JOBS, outputs, strict=True)
        ],
    }

    def publish_status() -> None:
        state["updated_at"] = datetime.now(UTC).isoformat()
        write_json(status_path, state)
        print(
            json.dumps({"status": state["status"], "jobs": state["jobs"]}), flush=True
        )

    try:
        publish_status()
        for index, ((protocol, _, classes, all_hanzi), output) in enumerate(
            zip(JOBS, outputs, strict=True)
        ):
            job = state["jobs"][index]
            job["status"] = "building"
            publish_status()
            command = [
                "--protocol",
                protocol,
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(output),
            ]
            if all_hanzi:
                command += ["--icdar-dir", str(args.data_root / "ICDAR2013_Test")]
            if build_manifest(command) != 0:
                raise RuntimeError(f"manifest builder failed for {protocol}")
            job.update(verify_published_manifest(output, protocol, classes))
            job["status"] = "completed"
            publish_status()
        state["status"] = "completed"
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        state["failed_job"] = next(
            (job["protocol"] for job in state["jobs"] if job["status"] == "building"),
            None,
        )
        traceback.print_exc()
        return 1
    finally:
        state["finished_at"] = datetime.now(UTC).isoformat()
        publish_status()


if __name__ == "__main__":
    raise SystemExit(main())
