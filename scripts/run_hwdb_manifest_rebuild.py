"""Build and audit proposal manifests, with optional vocabulary scaling."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from hccr.data.lmdb_store import manifest_digest
from hccr.data.manifest import read_manifest, validate_training_manifest
from hccr.tda.features import write_json

if __package__:
    from scripts.build_casia_manifest import main as build_manifest
else:
    from build_casia_manifest import main as build_manifest

JOBS = (
    (
        "hwdb1.0-1.1-official",
        "hwdb1_0-1_1_official_five_splits",
        3755,
        False,
    ),
    (
        "hwdb1.0-1.2-official",
        "hwdb1_0-1_2_official_four_splits",
        7185,
        True,
    ),
)


def verify_published_manifest(
    path: Path, protocol: str, class_count: int | None
) -> dict:
    report = json.loads((path / "audit_report.json").read_text(encoding="utf-8"))
    expected_splits = {"train", "calibration", "validation", "test"}
    if report.get("split_counts", {}).get("icdar"):
        expected_splits.add("icdar")
    if (
        report["protocol"] != protocol
        or report.get("hwdb_writer_disjoint_verified") is not True
        or set(report["split_counts"]) != expected_splits
        or set(report["writer_counts"]) != expected_splits
        or any(count <= 0 for count in report["split_counts"].values())
        or any(report["pairwise_writer_overlap_counts"].values())
        or ("icdar" in expected_splits and report["writer_counts"]["icdar"] != 60)
        or ("icdar" in expected_splits and report["split_counts"]["icdar"] != 224419)
        or report["image_overlap_check"]["status"] != "passed"
        or report["image_overlap_check"]["cross_split_duplicates"] != 0
        or report["image_overlap_check"]["checked_samples"]
        != sum(report["split_counts"].values())
        or (class_count is not None and report["class_count"] != class_count)
        or manifest_digest(path / "manifest.csv") != report["manifest_digest"]
        or manifest_digest(path / "class_mapping_full.json")
        != report["label_mapping_digest"]
    ):
        raise ValueError(f"published manifest failed proposal split audit: {path}")
    validate_training_manifest(
        path / "manifest.csv", read_manifest(path / "manifest.csv")
    )
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
        "icdar_identity_reconciliation": report.get("icdar_identity_reconciliation"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path("artifacts/hwdb-mapping-rebuild-20261022"),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--without-icdar", action="store_true")
    parser.add_argument("--source-version-map", type=Path)
    parser.add_argument("--writer-identity-map", type=Path)
    parser.add_argument("--secondary-vocabulary", type=Path)
    args = parser.parse_args(argv)
    jobs = list(JOBS if args.include_secondary else JOBS[:1])
    if args.without_icdar:
        jobs[0] = (
            jobs[0][0],
            jobs[0][1].replace("five_splits", "four_splits"),
            3755,
            False,
        )
    if args.plan_only:
        print(
            json.dumps(
                [
                    {"protocol": protocol, "output_dir": str(args.output_root / folder)}
                    for protocol, folder, _, _ in jobs
                ],
                indent=2,
            )
        )
        return 0
    args.status_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status_dir / "status.json"
    if status_path.exists():
        raise FileExistsError(f"rebuild already dispatched: {status_path}")
    outputs = [args.output_root / folder for _, folder, _, _ in jobs]
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
            for (protocol, _, _, _), output in zip(jobs, outputs, strict=True)
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
            zip(jobs, outputs, strict=True)
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
            for flag, value in (
                ("--source-version-map", args.source_version_map),
                ("--writer-identity-map", args.writer_identity_map),
                ("--vocabulary", args.secondary_vocabulary if all_hanzi else None),
            ):
                if value is not None:
                    command += [flag, str(value)]
            if args.without_icdar:
                command += ["--without-icdar"]
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
