"""Coverage / de-redundancy KV-cache prune selectors (B13 Direction A).

Reframes the memory-budget problem from "rank tokens by a scalar importance and
keep top-K" to **budget-constrained subset SELECTION for coverage**: pick the K
tokens that best *span* the (highly redundant) history, with an explicit
anti-redundancy term. See ``logs/raw/B13/abstract-problem-query.md`` and memory
``[[b13-direction-a-coverage-selection]]`` for the framing and the four
"fingerprints" (esp. #4: strong temporal redundancy => this is a coverage problem,
not a top-K one).

``CoverageTokenPruner`` is drop-in for ``train_stream_mtp.PrunedAnticipativeModel``:
same ``prune(feats) -> (kept_feats, kept_idx)`` interface as ``TokenPruner``, kept
in original chronological order.

Design (first cut, frozen-safe -- no reader change):

* Operate at **temporal-slot granularity**. Encoder tokens are temporal-major and
  gp-contiguous (gp = grid^2 spatial tokens per 2-frame slot), so slot ``s`` is
  tokens ``[s*gp : (s+1)*gp]``. Video redundancy is dominated by the temporal axis,
  so covering diverse *slots* captures most of it; keeping whole slots also keeps
  the block structure the predictor's RoPE positions index, and makes the budget
  quantize naturally (K is a multiple of gp for free).
* Represent each slot by its **mean token feature**.
* Select ``K/gp`` slots by **farthest-point sampling** (k-center greedy, the
  standard 2-approx to the k-center coverage objective): seed, then repeatedly add
  the slot whose min-distance to the already-selected set is largest.
* **Seed = most recent slot** by default: recency is a known-useful prior (the
  ``recent`` and ``loss_aware`` baselines exploit it), so "keep *now*, then cover
  the history diversely backward" is the most defensible single design and blends
  recency with coverage. ``seed="centroid"`` (slot nearest the mean) gives a
  recency-free pure-coverage variant.
"""
from __future__ import annotations

import torch


def _round_keep(keep_count: int, gp: int) -> int:
    return max(int(gp), (int(keep_count) // int(gp)) * int(gp))


def _facility_location_greedy(slot_feat: torch.Tensor, k: int) -> torch.Tensor:
    """Facility-location (k-medoid-style) greedy over ``slot_feat`` [B, n, D].
    Returns selected slot indices [B, k] (selection order).

    Maximizes the submodular coverage ``sum_i max_{j in S} sim(i, j)`` with
    ``sim = -squared distance`` (== minimize each slot's distance to its nearest
    selected representative). Unlike k-center/FPS this picks **representatives of
    dense regions**, not extremes/outliers -- the right objective for "keep one
    typical exemplar per redundant cluster". Seed-free.
    """
    B, n, _ = slot_feat.shape
    device = slot_feat.device
    sim = -torch.cdist(slot_feat, slot_feat).pow(2)  # [B, n(point), n(candidate)]
    best = torch.full((B, n), float("-inf"), device=device)  # best[i] = max sim to selected
    selected = torch.empty(B, k, dtype=torch.long, device=device)
    chosen = torch.zeros(B, n, dtype=torch.bool, device=device)
    barange = torch.arange(B, device=device)
    for i in range(k):
        gain = torch.clamp(sim - best.unsqueeze(1), min=0).sum(dim=1)  # [B, n_candidates]
        gain = gain.masked_fill(chosen, float("-inf"))
        nxt = gain.argmax(dim=1)  # [B]
        selected[:, i] = nxt
        chosen[barange, nxt] = True
        best = torch.maximum(best, sim[barange, :, nxt])  # sim to the new representative
    return selected


def _farthest_point_sampling(slot_feat: torch.Tensor, k: int, seed_idx: torch.Tensor) -> torch.Tensor:
    """k-center greedy over ``slot_feat`` [B, n, D]. Returns selected slot indices
    [B, k] (unsorted, selection order). ``seed_idx`` [B] is the first pick.

    Distances are squared-Euclidean in feature space. Already-selected slots are
    masked to -1 so they are never re-picked.
    """
    B, n, _ = slot_feat.shape
    device = slot_feat.device
    selected = torch.empty(B, k, dtype=torch.long, device=device)
    selected[:, 0] = seed_idx
    barange = torch.arange(B, device=device)
    # min squared-distance from every slot to the current selected set
    cur = slot_feat[barange, seed_idx]  # [B, D]
    dmin = (slot_feat - cur.unsqueeze(1)).pow(2).sum(-1)  # [B, n]
    dmin[barange, seed_idx] = -1.0
    for i in range(1, k):
        nxt = dmin.argmax(dim=1)  # [B]
        selected[:, i] = nxt
        cur = slot_feat[barange, nxt]  # [B, D]
        newd = (slot_feat - cur.unsqueeze(1)).pow(2).sum(-1)  # [B, n]
        dmin = torch.minimum(dmin, newd)
        dmin[barange, nxt] = -1.0
    return selected


class CoverageTokenPruner:
    """Direction A: budget-constrained coverage / de-redundancy selection via
    slot-level farthest-point sampling. Frozen-safe; no encoder/predictor patch."""

    def __init__(self, keep_count: int, gp: int, *, seed: str = "recent",
                 objective: str = "kcenter"):
        self.gp = int(gp)
        self.keep_count = _round_keep(keep_count, gp)
        if seed not in ("recent", "centroid"):
            raise ValueError(f"seed must be 'recent' or 'centroid', got {seed!r}")
        if objective not in ("kcenter", "facility"):
            raise ValueError(f"objective must be 'kcenter' or 'facility', got {objective!r}")
        self.seed = seed
        self.objective = objective

    def _seed_idx(self, slot_feat: torch.Tensor) -> torch.Tensor:
        B, n, _ = slot_feat.shape
        if self.seed == "recent":
            return torch.full((B,), n - 1, dtype=torch.long, device=slot_feat.device)
        # centroid: slot nearest the per-sample mean feature
        centroid = slot_feat.mean(dim=1, keepdim=True)  # [B, 1, D]
        return (slot_feat - centroid).pow(2).sum(-1).argmin(dim=1)  # [B]

    @torch.no_grad()
    def prune(self, feats: torch.Tensor):
        B, N, D = feats.shape
        gp = self.gp
        n_slots = N // gp
        k_slots = min(self.keep_count // gp, n_slots)
        if k_slots >= n_slots:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(B, -1)
            return feats, idx

        slot_feat = feats[:, : n_slots * gp, :].view(B, n_slots, gp, D).mean(dim=2)  # [B, n_slots, D]
        if self.objective == "facility":
            sel_slots = _facility_location_greedy(slot_feat, k_slots)  # seed-free representatives
        else:
            seed = self._seed_idx(slot_feat)
            sel_slots = _farthest_point_sampling(slot_feat, k_slots, seed)  # [B, k_slots]
        sel_slots = sel_slots.sort(dim=1).values  # chronological

        # Expand selected slots back to their gp token indices, chronological.
        offs = torch.arange(gp, device=feats.device)  # [gp]
        idx = (sel_slots.unsqueeze(-1) * gp + offs.view(1, 1, gp)).reshape(B, k_slots * gp)  # [B, K]
        gathered = feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        return gathered, idx

    def remove(self):  # symmetry with TokenPruner
        return None
