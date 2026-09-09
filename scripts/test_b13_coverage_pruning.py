"""CPU logic tests for B13 Direction-A coverage / de-redundancy prune selector.

Validates the index math of ``coverage_pruning`` in isolation (no model / video):

* ``_round_keep`` floors the budget to a multiple of gp;
* ``_farthest_point_sampling`` (k-center greedy) covers distinct clusters and
  never re-picks a selected slot;
* ``CoverageTokenPruner.prune`` returns gp-contiguous, chronological token indices
  for the selected slots, honours the budget, and passes-through when K >= N;
* seed selection ('recent' = last slot, 'centroid' = slot nearest the mean).
"""
import torch

from app.hdepic_lora_action_anticipation import coverage_pruning as cp


def test_round_keep():
    assert cp._round_keep(4100, 256) == 4096  # 16 * 256
    assert cp._round_keep(3900, 256) == 3840  # 15 * 256
    assert cp._round_keep(100, 256) == 256     # >= gp floor


def test_fps_avoids_redundancy():
    # two near-duplicate clusters A={0,1}~0, B={2,3}~10; seed at recent (slot 3).
    slot_feat = torch.tensor([[[0.0], [0.05], [10.0], [10.05]]])  # [1, 4, 1]
    seed = torch.tensor([3])
    sel = cp._farthest_point_sampling(slot_feat, k=2, seed_idx=seed)
    # must add the farthest slot (from the other cluster), not a near-duplicate
    assert set(sel[0].tolist()) == {3, 0}
    # k=3: after {3,0} the next farthest is the remaining most-distinct slot
    sel3 = cp._farthest_point_sampling(slot_feat, k=3, seed_idx=seed)
    assert len(set(sel3[0].tolist())) == 3  # no re-pick


def test_facility_picks_representatives():
    # dense cluster {0,1,2} near 0 + one far outlier {3}. facility-location with
    # k=1 must pick a DENSE representative (0/1/2), NOT the outlier -- the opposite
    # of k-center/FPS which would chase the extreme.
    slot_feat = torch.tensor([[[0.0], [0.1], [-0.1], [50.0]]])  # [1, 4, 1]
    sel1 = cp._facility_location_greedy(slot_feat, k=1)
    assert sel1[0, 0].item() in (0, 1, 2)
    # FPS from the outlier seed would keep the outlier; contrast documented here.
    sel_fps = cp._farthest_point_sampling(slot_feat, k=1, seed_idx=torch.tensor([3]))
    assert sel_fps[0, 0].item() == 3
    # k=2 facility covers both the dense region and the outlier (once forced to add)
    sel2 = cp._facility_location_greedy(slot_feat, k=2)
    assert 3 in sel2[0].tolist() and len(set(sel2[0].tolist())) == 2


def test_facility_via_pruner():
    gp = 2
    vals = [0.0, 0.1, -0.1, 50.0]
    feats = torch.zeros(1, len(vals) * gp, 3)
    for s, v in enumerate(vals):
        feats[0, s * gp:(s + 1) * gp, :] = v
    _, idx = cp.CoverageTokenPruner(keep_count=2, gp=gp, objective="facility").prune(feats)
    # k_slots=1 -> a dense representative slot (0/1/2), gp-contiguous
    assert idx.shape == (1, gp)
    assert idx[0, 0].item() // gp in (0, 1, 2)


def test_seed():
    slot_feat = torch.tensor([[[0.0], [0.0], [0.0], [10.0]]])
    assert cp.CoverageTokenPruner(2, 2, seed="recent")._seed_idx(slot_feat).tolist() == [3]
    # centroid mean = 2.5 -> nearest is a 0-slot (argmin picks the first)
    assert cp.CoverageTokenPruner(2, 2, seed="centroid")._seed_idx(slot_feat).tolist() == [0]


def test_coverage_pruner():
    gp = 2
    vals = [0.0, 0.05, 10.0, 10.05]  # 4 slots, two clusters
    feats = torch.zeros(1, len(vals) * gp, 3)
    for s, v in enumerate(vals):
        feats[0, s * gp:(s + 1) * gp, :] = v
    # keep_count 4 -> k_slots = 2; recent seed (slot 3) + farthest (slot 0)
    kept, idx = cp.CoverageTokenPruner(keep_count=4, gp=gp, seed="recent").prune(feats)
    assert idx.tolist() == [[0, 1, 6, 7]]  # slots 0 and 3, gp-contiguous, chronological
    assert torch.equal(kept, feats[:, [0, 1, 6, 7], :])
    # K >= N -> pass-through (all tokens, original order)
    _, idx_all = cp.CoverageTokenPruner(keep_count=100, gp=gp).prune(feats)
    assert idx_all.tolist() == [list(range(len(vals) * gp))]


def test_batched_independent_seeds():
    # two samples with clusters swapped; recent seed differs in effect per row
    a = torch.tensor([[[0.0], [0.1], [9.0], [9.1]]])
    b = torch.tensor([[[9.0], [9.1], [0.0], [0.1]]])
    slot_feat = torch.cat([a, b], dim=0)  # [2, 4, 1]
    seed = torch.tensor([3, 3])
    sel = cp._farthest_point_sampling(slot_feat, k=2, seed_idx=seed)
    # seed is slot 3 (near cluster {2,3}); the 2nd pick is the single farthest slot,
    # which lies in the far cluster {0,1} for both rows.
    assert sel[:, 0].tolist() == [3, 3]
    assert sel[0, 1].item() in (0, 1) and sel[1, 1].item() in (0, 1)


if __name__ == "__main__":
    test_round_keep()
    print("round-keep tests OK")
    test_fps_avoids_redundancy()
    print("fps tests OK")
    test_facility_picks_representatives()
    test_facility_via_pruner()
    print("facility-location tests OK")
    test_seed()
    print("seed tests OK")
    test_coverage_pruner()
    print("coverage-pruner tests OK")
    test_batched_independent_seeds()
    print("batched-fps tests OK")
    print("ALL TESTS PASSED")
