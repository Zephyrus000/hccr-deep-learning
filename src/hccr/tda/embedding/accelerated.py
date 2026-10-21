"""Exhaustive FP64 matrix screening with the original CPU neighbor ordering."""

from __future__ import annotations

import numpy as np

from .neighborhood import EMBEDDING_DIM, NeighborBatch, normalize_rows

SCREENING_PROTOCOL = "fp64_gamma_2d8_x32_closed_2error_legacy_rerank_v1"


class ExactMatrixSearch:
    def __init__(
        self, bank, *, device="cuda", max_distance_bytes=1024**3, candidate_cap=8192
    ):
        import torch

        self.bank = bank
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("exact matrix search requires CPU or CUDA FP64")
        if max_distance_bytes < 8 * len(bank.targets) or candidate_cap < 1:
            raise ValueError("matrix budget/candidate cap is too small")
        self.max_queries = max_distance_bytes // (8 * len(bank.targets))
        self.candidate_cap = candidate_cap
        self.squared_norms = np.empty(len(bank.targets), dtype=np.float64)
        for start in range(0, len(bank.targets), 4096):
            values = np.asarray(bank.vectors[start : start + 4096])
            self.squared_norms[start : start + len(values)] = np.sum(values**2, axis=1)
        self.references = torch.tensor(
            np.asarray(bank.vectors), dtype=torch.float64, device=self.device
        )
        self.norms = torch.tensor(self.squared_norms, device=self.device)
        self.id_to_index = {str(value): i for i, value in enumerate(bank.sample_ids)}
        # For unit vectors, Cauchy bounds sum(abs(q_i*r_i)) by their norms.
        # gamma(2D+8) covers both reduction paths and distance arithmetic;
        # x32 is conservative for IEEE FP64 GEMM/FMA and fixed-order einsum.
        # No autocast/FP32/TF32 dot products are used. This is an absolute bound
        # against the old squared-distance expression, including cancellation.
        unit_roundoff = np.finfo(np.float64).eps / 2
        operations = 2 * EMBEDDING_DIM + 8
        gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
        self.error_bound = 32 * gamma * max(1.0, float(self.squared_norms.max()))
        self.diagnostics = {
            "protocol": SCREENING_PROTOCOL,
            "device": str(self.device),
            "squared_distance_error_bound": self.error_bound,
            "queries": 0,
            "reranked_candidates": 0,
            "oracle_fallback_queries": 0,
        }

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def close(self):
        self.references = self.norms = None

    def neighbors(
        self, queries, query_ids, k, *, reference_batch_size=2048, query_batch_size=32
    ):
        import torch

        if not 1 <= k < len(self.bank.targets):
            raise ValueError("neighbors require k+1 or more reference vectors")
        if (
            len(queries) != len(query_ids)
            or min(query_batch_size, reference_batch_size) < 1
        ):
            raise ValueError("invalid query IDs or search batch sizes")
        indices = np.empty((len(queries), k), dtype=np.int64)
        distances = np.empty((len(queries), k), dtype=np.float64)
        batch_size = min(query_batch_size, self.max_queries)
        with torch.inference_mode():
            for start in range(0, len(queries), batch_size):
                normalized = normalize_rows(queries[start : start + batch_size])
                ids = list(map(str, query_ids[start : start + len(normalized)]))
                norms = np.sum(normalized**2, axis=1)
                tensor = torch.tensor(normalized, device=self.device)
                matrix = tensor @ self.references.T
                matrix.mul_(-2)
                matrix.add_(torch.tensor(norms, device=self.device)[:, None])
                matrix.add_(self.norms[None, :]).clamp_min_(0)
                excluded = [self.id_to_index.get(value, -1) for value in ids]
                valid = np.flatnonzero(np.array(excluded) >= 0)
                if len(valid):
                    matrix[
                        torch.tensor(valid, device=self.device),
                        torch.tensor(np.array(excluded)[valid], device=self.device),
                    ] = torch.inf
                # Padding only avoids a full threshold scan in the usual case.
                # It never decides membership: a closed 2E band certifies it.
                width = min(k + 8, len(self.bank.targets))
                values, chosen = torch.topk(matrix, width, largest=False, sorted=True)
                values, chosen = values.cpu().numpy(), chosen.cpu().numpy()
                for row, query in enumerate(normalized):
                    threshold = values[row, k - 1]
                    if not np.isfinite(threshold):
                        raise ValueError(
                            "not enough neighbors after sample-ID exclusion"
                        )
                    if values[row, -1] <= threshold + 2 * self.error_bound:
                        mask = matrix[row] <= threshold + 2 * self.error_bound
                        candidates = torch.nonzero(mask).flatten().cpu().numpy()
                    else:
                        candidates = chosen[row]
                    if excluded[row] >= 0:
                        candidates = candidates[candidates != excluded[row]]
                    if len(candidates) > self.candidate_cap:
                        answer = self.bank.neighbors(
                            queries[start + row : start + row + 1],
                            [ids[row]],
                            k,
                            reference_batch_size=reference_batch_size,
                        )
                        indices[start + row] = answer.indices[0]
                        distances[start + row] = answer.distances[0]
                        self.diagnostics["oracle_fallback_queries"] += 1
                        continue
                    references = np.asarray(self.bank.vectors[candidates])
                    # Reproduce the reference arithmetic, not the GEMM distances.
                    dot = np.einsum(
                        "id,jd->ij", query[None], references, optimize=False
                    )[0]
                    exact = np.maximum(
                        norms[row] + np.sum(references**2, axis=1) - 2 * dot, 0
                    )
                    order = np.lexsort((self.bank.ranks[candidates], exact))[:k]
                    selected = candidates[order]
                    indices[start + row] = selected
                    distances[start + row] = np.linalg.norm(
                        self.bank.vectors[selected] - query, axis=1
                    )
                    self.diagnostics["reranked_candidates"] += len(candidates)
                self.diagnostics["queries"] += len(normalized)
                del matrix, tensor
        return NeighborBatch(indices, distances)
