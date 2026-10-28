"""Post-training statistics with paired, source-stratified writer resampling."""

from __future__ import annotations

import numpy as np


def prediction_statistics(targets, top5, selected_classes=(), classes=3755):
    targets, top5 = np.asarray(targets), np.asarray(top5)
    if targets.ndim != 1 or top5.shape != (len(targets), 5) or not len(targets):
        raise ValueError("Expected nonempty targets and five predictions per image")
    if (targets < 0).any() or (targets >= classes).any():
        raise ValueError("Target class outside vocabulary")
    if (top5 < 0).any() or (top5 >= classes).any():
        raise ValueError("Predicted class outside vocabulary")
    correct = targets == top5[:, 0]
    support = np.bincount(targets, minlength=classes)
    hits = np.bincount(targets[correct], minlength=classes)
    predicted = np.bincount(top5[:, 0], minlength=classes)
    selected = np.isin(targets, selected_classes)
    pairs, counts = np.unique(
        np.stack((targets[~correct], top5[~correct, 0]), axis=1),
        axis=0, return_counts=True,
    )
    order = np.argsort(-counts, kind="stable")[:30]
    return {
        "samples": len(targets), "top1": float(correct.mean()),
        "top5": float((top5 == targets[:, None]).any(1).mean()),
        "macro_recall": float((hits[support > 0] / support[support > 0]).mean()),
        "in_H": {"samples": int(selected.sum()),
                 "top1": float(correct[selected].mean()) if selected.any() else None},
        "outside_H": {"samples": int((~selected).sum()),
                      "top1": float(correct[~selected].mean()) if (~selected).any() else None},
        "per_class_support": support.tolist(), "per_class_hits": hits.tolist(),
        "per_class_prediction_frequency": predicted.tolist(),
        "per_class_false_positives": (predicted - hits).tolist(),
        "top_confusions_exploratory": [
            {"true": int(pairs[i, 0]), "predicted": int(pairs[i, 1]),
             "count": int(counts[i])} for i in order
        ],
    }


def paired_writer_interval(differences, writers, versions, *, repetitions=2000, seed=20261026):
    """Resample whole writers within source signatures; retain paired outcomes.

    A writer appearing in several source versions stays one cluster. Writers
    are stratified by their complete source signature rather than split apart.
    """
    delta = np.asarray(differences, dtype=np.float64)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Differences must be a finite, nonempty vector")
    if len(writers) != len(delta) or len(versions) != len(delta):
        raise ValueError("Writer/source arrays must align with paired outcomes")
    if repetitions < 1 or any(not str(w).strip() for w in writers):
        raise ValueError("Writer IDs and positive repetition count are required")
    ids, inverse = np.unique(np.asarray(writers, dtype=str), return_inverse=True)
    sources = np.asarray(versions, dtype=str)
    signatures = [set() for _ in ids]
    for index, source in zip(inverse, sources, strict=True):
        signatures[index].add(source)
    strata = {}
    for index, signature in enumerate(signatures):
        strata.setdefault(tuple(sorted(signature)), []).append(index)
    numerator = np.bincount(inverse, weights=delta, minlength=len(ids))
    denominator = np.bincount(inverse, minlength=len(ids))
    rng = np.random.default_rng(seed)
    sums = np.zeros(repetitions)
    sizes = np.zeros(repetitions)
    for members in strata.values():
        draw = rng.choice(members, size=(repetitions, len(members)), replace=True)
        sums += numerator[draw].sum(1)
        sizes += denominator[draw].sum(1)
    low, high = np.quantile(sums / sizes, [0.025, 0.975])
    return {
        "difference_pp": float(delta.mean() * 100),
        "ci95_pp": [float(low * 100), float(high * 100)],
        "samples": len(delta), "writers": len(ids),
        "repetitions": repetitions, "seed": seed,
        "method": "paired_writer_cluster_percentile_stratified_by_source_signature",
        "strata_writer_counts": {"+".join(k): len(v) for k, v in strata.items()},
        "multiplicity_adjusted": False,
    }


def overlap_statistics(masks):
    sets = {str(k): set(v) for k, v in masks.items()}
    result = []
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            intersection, union = sets[a] & sets[b], sets[a] | sets[b]
            result.append({"seed_a": a, "seed_b": b,
                           "intersection": len(intersection),
                           "jaccard": len(intersection) / len(union)})
    return result
