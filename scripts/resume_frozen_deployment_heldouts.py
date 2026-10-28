"""Resume the immutable scoring runner after checking every derived input identity.

This operational wrapper does not alter the frozen scoring code or protocol.
Always use this entry point for a restart, including after a process interruption.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import evaluate_frozen_deployment_heldouts as scoring


def verify_derived(output):
    import numpy as np
    checked = {}
    receipt_path = output / "cache/reconciliation.json"
    if not receipt_path.exists():
        return {"metadata_reconciliation": "not committed; original runner reconstructs it"}
    reconciliation = scoring.read(receipt_path)
    protocol_hash = scoring.sha(output / "protocol.json")
    if reconciliation["protocol_file_sha256"] != protocol_hash:
        raise ValueError("Reconciliation belongs to another protocol")
    for endpoint in scoring.COUNTS:
        for name, expected in [(f"{endpoint}-rows.csv", reconciliation[endpoint]["rows_sha256"]),
                               (f"{endpoint}-targets.npy", reconciliation[endpoint]["targets_sha256"])]:
            if scoring.sha(output / "cache" / name) != expected:
                raise ValueError(f"Derived endpoint identity changed: {name}")
            checked[name] = expected
        cache_receipt = output / "cache" / f"{endpoint}-cache.json"
        if cache_receipt.exists():
            cache = scoring.read(cache_receipt)
            if (cache["protocol_file_sha256"] != protocol_hash or
                    cache["rows_sha256"] != reconciliation[endpoint]["rows_sha256"]):
                raise ValueError("Image cache provenance changed")
            image_path = output / "cache" / f"{endpoint}-uint8.npy"
            if scoring.sha(image_path) != cache["sha256"]:
                raise ValueError("Derived image cache changed")
            checked[image_path.name] = cache["sha256"]
    for path in sorted((output / "evaluations").glob("*/*/*/metrics.json")):
        result = scoring.read(path)
        endpoint = result["split"]
        if (result["protocol_file_sha256"] != protocol_hash or
                result["targets_sha256"] != reconciliation[endpoint]["targets_sha256"] or
                result["sample_identity_sha256"] != reconciliation[endpoint]["sample_identity_sha256"] or
                result["samples"] != scoring.COUNTS[endpoint]):
            raise ValueError("Completed evaluation input provenance changed")
        for name, expected in result["outputs_sha256"].items():
            if scoring.sha(path.parent / name) != expected:
                raise ValueError("Completed output changed")
        targets = np.load(output / "cache" / f"{endpoint}-targets.npy", mmap_mode="r")
        top5 = np.load(path.parent / "top5.npy", mmap_mode="r")
        if (abs(float((top5[:, 0] == targets).mean()) - result["top1"]) > 1e-12 or
                abs(float((top5 == targets[:, None]).any(1).mean()) - result["top5"]) > 1e-12):
            raise ValueError("Completed metrics differ from saved predictions")
        checked[str(path.relative_to(output))] = scoring.sha(path)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=scoring.OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    with scoring.owner(output):
        protocol = scoring.verified_protocol(output)
        checked = verify_derived(output)
        scoring.atomic(output / "resume-checks" / f"{time.time_ns()}.json", {
            "wrapper_sha256": scoring.sha(__file__), "protocol_file_sha256": scoring.sha(output / "protocol.json"),
            "derived_inputs_and_completed_outputs": checked, "scientific_protocol_changed": False})
        return scoring.run(output)


if __name__ == "__main__":
    raise SystemExit(main())
