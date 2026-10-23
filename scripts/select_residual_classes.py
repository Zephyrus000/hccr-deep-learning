"""Select an auditable residual mask from paired calibration prediction JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from hccr.models.allocation import load_paired_predictions, select_residual_classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-predictions", type=Path, required=True)
    parser.add_argument("--low-rank-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "penalty",
            "pair",
            "pair_only",
            "random",
            "frequency",
            "raw_difficulty",
            "directional_svd",
        ),
        default="penalty",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--control-scores",
        type=Path,
        help=(
            "JSON with class_mapping_digest, values and provenance; frequency also "
            "requires split=fit, official_partition=train, manifest_digest"
        ),
    )
    args = parser.parse_args()
    protected = [args.dense_predictions, args.low_rank_predictions]
    if args.control_scores is not None:
        protected.append(args.control_scores)
    if args.output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Mask output must not overwrite prediction or control inputs")
    paired = load_paired_predictions(args.dense_predictions, args.low_rank_predictions)
    options = {}
    control_provenance = None
    if args.control_scores is not None:
        if args.method not in ("frequency", "directional_svd"):
            parser.error("--control-scores is only used by frequency/directional_svd")
        contents = args.control_scores.read_bytes()
        control = json.loads(contents)
        if (
            control.get("class_mapping_digest")
            != paired.dense_metadata["class_mapping_digest"]
        ):
            raise ValueError("Control-score class mapping mismatch")
        if not isinstance(control.get("provenance"), dict) or not control["provenance"]:
            raise ValueError("Control scores require provenance")
        if args.method == "frequency" and (
            control.get("split") != "fit"
            or control.get("official_partition") != "train"
            or control.get("manifest_digest")
            != paired.dense_metadata["fit_manifest_digest"]
        ):
            raise ValueError("Frequency scores must come from the paired fit manifest")
        if args.method == "directional_svd" and (
            control["provenance"].get("checkpoint_digest")
            != paired.dense_metadata["checkpoint_digest"]
            or type(control.get("rank")) is not int
            or control["rank"] <= 0
            or control["rank"] != paired.low_rank_metadata.get("classifier_rank")
        ):
            raise ValueError(
                "Directional scores must match the paired dense checkpoint "
                "and low-rank rank"
            )
        key = "frequencies" if args.method == "frequency" else "directional_errors"
        options[key] = control["values"]
        control_provenance = {
            "file_digest": hashlib.sha256(contents).hexdigest(),
            "metadata": {
                key: value for key, value in control.items() if key != "values"
            },
        }
    result = select_residual_classes(
        paired, args.k, method=args.method, seed=args.seed, eta=args.eta, **options
    )
    if control_provenance is not None:
        result["provenance"]["control_scores"] = control_provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        f"Selected {args.k} classes; {len(result['zero_penalty_selected'])} "
        f"have zero compression penalty. Wrote {args.output}"
    )


if __name__ == "__main__":
    main()
