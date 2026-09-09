#!/usr/bin/env python3
"""Export frozen CPU summaries as explicit pooled masks and received-group ratios."""
import json
import os
from pathlib import Path
import numpy as np
from scripts.egtea.analyze_b18_q3_token_self_return import GROUPS, mask_stats, digest, write_json

root = Path(os.environ["TS_REVIEW_INPUT"])
out = Path(os.environ["TS_REVIEW_OUTPUT"])
out.mkdir(parents=True, exist_ok=False)
summary = json.loads((root / "summary.json").read_text())
assert summary["n_rows"] == 86 and not summary["partial"]
assert digest(Path(__file__).with_name("analyze_b18_q3_token_self_return.py")) == summary["analysis_sha256"]
with np.load(root / "pooled_sums.npz") as archive:
    pooled = {k: archive[k] for k in archive.files}
masks, received, provenance = {}, {}, {}
for suffix in ("actual", "no_rope_readout"):
    received[suffix] = {}
    for route in ("legacy", "fp32", "self_excluded_fp32"):
        stats, mask = mask_stats(pooled["score_" + route + "_" + suffix])
        assert mask.sum() == 4096
        assert stats == summary["population"][suffix]["selection"][route]
        masks[route + "__" + suffix] = mask
    flow = pooled["flow_fp32_" + suffix].sum(0)[:, :64]
    columns = pooled["head_fp32_" + suffix][:, :64*256].reshape(12,64,256).sum((0,2))
    diagonal = pooled["row_self_" + suffix].sum((0,2))
    same = np.diag(flow[:64])
    for name, ids in GROUPS.items():
        total = float(columns[ids].sum())
        exact = float(diagonal[ids].sum())
        own = float(same[ids].sum())
        received[suffix][name] = dict(total_received_mass=total, exact_self_mass=exact,
            same_slot_other_mass=own-exact, other_sources_mass=total-own,
            exact_self_share=exact/total, same_slot_other_share=(own-exact)/total,
            whole_same_slot_share=own/total, other_sources_share=(total-own)/total,
            exact_self_share_of_same_slot=exact/own)
np.savez_compressed(out / "pooled_masks.npz", **masks)
write_json(out / "received_group_metrics.json", received)
for name in ("summary.json", "pooled_sums.npz", "individual_masks.npz"):
    provenance[name] = dict(path=str(root/name), sha256=digest(root/name))
write_json(out / "review_export.json", dict(protocol=summary["protocol"], n_rows=86,
    job_id=os.environ.get("SLURM_JOB_ID"), code_sha256=digest(__file__), source=provenance,
    interpretation="Derivative export only; same audited probabilities, ratios of sums; no new capture",
    artifacts={name:digest(out/name) for name in ("pooled_masks.npz","received_group_metrics.json")}))
print("SELF_REVIEW_EXPORT", str(out), flush=True)
