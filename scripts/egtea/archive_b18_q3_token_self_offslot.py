#!/usr/bin/env python3
"""Formal CPU derivative: off-self context-key uniformity per query slot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    pooled=a.source/"pooled_sums.npz"; source_summary=a.source/"summary.json"
    with np.load(pooled) as z:
        assert "flow_fp32_actual" in z.files
        raw=z["flow_fp32_actual"]
    assert raw.shape==(12,65,65) and raw.dtype==np.float64 and np.isfinite(raw).all() and (raw>=0).all()
    flow=raw.sum(0)[:64,:64]
    rows=[]
    for q in range(64):
        keys=np.arange(64)!=q; off=flow[q,keys]; key_ids=np.arange(64)[keys]
        assert len(off)==63 and off.sum()>0 and (off>0).all()
        prob=off/off.sum(); mean=off.mean()
        rows.append(dict(query_slot=q,cv=float(off.std(ddof=0)/mean),
            normalized_entropy=float(-(prob*np.log(prob)).sum()/np.log(63)),max_over_mean=float(off.max()/mean),
            strongest_offslot_key=int(key_ids[off.argmax()]),strongest_probability=float(prob.max()),offslot_mass=float(off.sum())))
    with (a.out/"per_query_slot.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def summarize(name,idx):
        selected=[rows[i] for i in idx]; pooled_key=flow[np.asarray(idx)].copy()
        for local,q in enumerate(idx): pooled_key[local,q]=0
        key_mass=pooled_key.sum(0)
        return dict(name=name,n_query_slots=len(idx),query_slots=list(idx),
            mean_cv=float(np.mean([x["cv"] for x in selected])),
            mean_normalized_entropy=float(np.mean([x["normalized_entropy"] for x in selected])),
            mean_max_over_mean=float(np.mean([x["max_over_mean"] for x in selected])),
            strongest_offslot_key_by_group_mass=int(key_mass.argmax()),strongest_offslot_key_group_mass=float(key_mass.max()))
    groups={"slot0":summarize("slot0",range(0,1)),"slot1":summarize("slot1",range(1,2)),
            "middle16_48":summarize("middle16_48",range(16,48)),"all64":summarize("all64",range(64))}
    assert groups["slot0"]["strongest_offslot_key_by_group_mass"]==1
    assert groups["slot1"]["strongest_offslot_key_by_group_mass"]==0
    assert abs(groups["slot0"]["mean_cv"]-0.9153)<5e-4 and abs(groups["slot1"]["mean_cv"]-1.0782)<5e-4
    assert abs(groups["slot0"]["mean_max_over_mean"]-6.0375)<5e-4 and abs(groups["slot1"]["mean_max_over_mean"]-6.7185)<5e-4
    assert abs(groups["middle16_48"]["mean_cv"]-0.7874)<5e-4 and abs(groups["middle16_48"]["mean_max_over_mean"]-4.3640)<5e-4
    report=dict(protocol="b18-predictor-prune/egtea-q3-token-self-return-v1",derivative="off-self-context-key-uniformity-v1",
        definition="sum flow_fp32_actual over 12 heads; for each context query slot remove its same-index context key slot, normalize remaining 63 positive masses; population CV, entropy/log(63), and max/mean",
        interpretation_boundary="pooled received score over query sources can be temporally fairly flat while each fixed query slot has nonuniform off-self routing",
        source=dict(directory=str(a.source),pooled_sums_sha256=sha(pooled),summary_sha256=sha(source_summary),array="flow_fp32_actual",shape=list(raw.shape),dtype=str(raw.dtype)),
        groups=groups,per_query_slot=rows,gates=dict(shape_exact=True,finite_nonnegative=True,all_offslot_positive=True,
            expected_slot01_strongest_keys=True,independent_expected_value_checks=True))
    (a.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    report["artifacts"]={"summary_sha256":sha(a.out/"summary.json"),"per_query_slot_sha256":sha(a.out/"per_query_slot.csv")}
    (a.out/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(groups=groups,summary_sha256=sha(a.out/"summary.json"),audit_sha256=sha(a.out/"audit.json"))),flush=True)


if __name__=="__main__": main()
