#!/usr/bin/env python3
"""Independent audit and paired inference for B18 Q3 random token drop."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.hdepic_lora_action_anticipation.eval_b18_q3_random_drop_accuracy import (
    PROTOCOL, ARMS, ANCHORS, RANDOM_ARMS, HORIZONS, KEEP, make_masks, model_args, sha, write_json,
)
from app.hdepic_lora_action_anticipation import train_stream_mtp as T

METRICS = ["top1", "top3", "top5", "ce"]


def monitor_stats(run, timing):
    records = []
    with (run.parent / "gpu.csv").open() as f:
        for line in f:
            fields = [v.strip() for v in line.split(",")]
            if len(fields) == 5:
                try: records.append((datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(fields[2])))
                except ValueError: pass
    arr = np.asarray(records)
    begin = timing[1]["start"] if len(timing) > 1 else timing[0]["start"]
    end = timing[-1]["compute_end"]
    selected = arr[(arr[:, 0] >= begin) & (arr[:, 0] <= end), 1]
    return dict(path=str(run.parent / "gpu.csv"), interval_seconds=5, whole_samples=len(arr),
                whole_mean=float(arr[:, 1].mean()), steady_samples=len(selected),
                steady_mean=float(selected.mean()) if len(selected) else None,
                steady_start_epoch=begin, steady_end_epoch=end,
                steady_definition="second measured batch start through last compute end")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partial", action="store_true"); p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text()); assert manifest["protocol"] == PROTOCOL
    masks, mask_audit = make_masks()
    with np.load(args.manifest.parent / "fixed_masks.npz") as frozen:
        assert set(frozen.files) == set(ARMS) and all(np.array_equal(frozen[a], masks[a]) for a in ARMS)
    maps = T.load_action_maps(model_args().train_csv)
    records, run_reports, source_versions = [], [], []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        assert summary["protocol"] == PROTOCOL and summary["manifest_sha256"] == sha(args.manifest)
        assert summary["class_counts"] == [len(x) for x in maps]
        assert summary["target_positions"]["first_target"] == 6144 and summary["target_positions"]["context"] == list(range(KEEP))
        rows = [json.loads(x) for x in (run / "predictions.jsonl").read_text().splitlines()]
        assert len(rows) == summary["rows"] == summary["stop"] - summary["start"]
        expected = manifest["samples"][summary["start"]:summary["stop"]]
        assert [r["sample_id"] for r in rows] == [s["sample_id"] for s in expected]
        cursor = 0
        for path in sorted(run.glob("logits_*.npz")):
            with np.load(path) as z:
                ids = z["sample_ids"].tolist(); batch = rows[cursor:cursor+len(ids)]
                assert ids == [r["sample_id"] for r in batch] and set(z.files) == {"sample_ids", *ARMS}
                for arm in ARMS:
                    logits = z[arm]; assert logits.shape == (len(ids), 3, len(maps[2])) and np.isfinite(logits).all()
                    for bi, r in enumerate(batch):
                        for hi, horizon in enumerate(HORIZONS):
                            h = f"{horizon:g}s"; x = torch.from_numpy(logits[bi, hi]); saved = r["arms"][arm][h]
                            top = x.topk(5).indices.tolist(); assert top == saved["top5_indices"]
                            ties = {}
                            for k in [1, 3, 5]:
                                threshold = float(x.topk(k).values[-1]); above = int((x > threshold).sum()); equal = int((x == threshold).sum())
                                ties[f"top{k}"] = dict(boundary=above < k < above+equal, n_equal=equal, threshold=threshold)
                            if r["labels"][h]["valid"]:
                                y = r["labels"][h]["label"]; assert saved["label"] == y
                                for k in [1, 3, 5]: assert saved[f"top{k}"] == (y in top[:k])
                                ce = float(torch.nn.functional.cross_entropy(x[None], torch.tensor([y]))); assert abs(ce-saved["ce"]) < 1e-6
                            r.setdefault("tie_audit", {}).setdefault(arm, {})[h] = ties
            cursor += len(ids)
        assert cursor == len(rows)
        timing = [json.loads(x) for x in (run / "timings.jsonl").read_text().splitlines()]
        assert sum(x["rows"] for x in timing) == len(rows)
        run_reports.append(dict(run=str(run), job_id=summary["job_id"], start=summary["start"], stop=summary["stop"], rows=len(rows),
            gpu=summary["gpu"], seconds=summary["seconds"], row_seconds=summary["seconds"]/len(rows),
            peak_cuda_bytes=summary["peak_cuda_bytes"], hardware_comparison=summary.get("hardware_comparison"),
            monitor=monitor_stats(run, timing), summary_sha256=sha(run/"summary.json"), predictions_sha256=sha(run/"predictions.jsonl")))
        records.extend(rows); source_versions.append(summary["source_sha256"])
    assert all(x == source_versions[0] for x in source_versions)
    order = {s["sample_id"]: i for i, s in enumerate(manifest["samples"])}
    records.sort(key=lambda r: order[r["sample_id"]])
    assert len({r["sample_id"] for r in records}) == len(records)
    if not args.partial: assert [order[r["sample_id"]] for r in records] == list(range(4000))
    extended = ARMS + ["random_mean"]
    values = np.full((len(records), len(extended), 3, 4), np.nan); valid = np.zeros((len(records), 3), bool)
    boundary = np.zeros((len(records), len(ARMS), 3, 3), bool)
    for ri, r in enumerate(records):
        for hi, horizon in enumerate(HORIZONS):
            h = f"{horizon:g}s"; valid[ri, hi] = r["labels"][h]["valid"]
            if not valid[ri, hi]: continue
            for ai, arm in enumerate(ARMS):
                values[ri, ai, hi] = [r["arms"][arm][h][m] for m in METRICS]
                boundary[ri, ai, hi] = [r["tie_audit"][arm][h][m]["boundary"] for m in METRICS[:3]]
    ridx = [extended.index(a) for a in RANDOM_ARMS]
    values[:, -1] = np.nanmean(values[:, ridx], axis=1)
    sessions = sorted({r["video_id"] for r in records}); si = {s:i for i,s in enumerate(sessions)}
    denom = np.zeros((len(sessions),3)); sums = np.zeros((len(sessions),len(extended),3,4))
    for ri,r in enumerate(records): denom[si[r["video_id"]]] += valid[ri]; sums[si[r["video_id"]]] += np.nan_to_num(values[ri])
    mean = sums.sum(0)/denom.sum(0)[None,:,None]
    rng=np.random.default_rng(20260908); weights=rng.multinomial(len(sessions),np.full(len(sessions),1/len(sessions)),size=args.bootstrap)
    boot=np.einsum("bs,sahm->bahm",weights,sums)/(weights@denom)[:,None,:,None]
    estimates=[]
    for ai,arm in enumerate(extended):
        for hi,h in enumerate(HORIZONS):
            for mi,m in enumerate(METRICS):
                scale=1 if m=="ce" else 100
                estimates.append(dict(arm=arm,horizon=h,metric=m,n=int(denom[:,hi].sum()),estimate=float(mean[ai,hi,mi]*scale),
                    ci95=(np.quantile(boot[:,ai,hi,mi],[.025,.975])*scale).tolist(),units="nats" if m=="ce" else "percent"))
    comparisons=[]
    for other in ANCHORS:
        a,b=extended.index("random_mean"),extended.index(other)
        for hi,h in enumerate(HORIZONS):
            for mi,m in enumerate(METRICS):
                scale=1 if m=="ce" else 100; d=values[:,a,hi,mi]-values[:,b,hi,mi]
                comparisons.append(dict(a="random_mean",b=other,horizon=h,metric=m,n=int(valid[:,hi].sum()),
                    delta=float((mean[a,hi,mi]-mean[b,hi,mi])*scale),
                    ci95=(np.quantile(boot[:,a,hi,mi]-boot[:,b,hi,mi],[.025,.975])*scale).tolist(),
                    units="nats" if m=="ce" else "percentage_points",wins=int(np.nansum(d>0)),losses=int(np.nansum(d<0))))
    participant_sensitivity=[]
    participants=sorted({r["participant_id"] for r in records})
    for other in ["recent4s","offline_L0"]:
        a,b=extended.index("random_mean"),extended.index(other)
        for hi,h in enumerate(HORIZONS):
            for mi,m in enumerate(METRICS):
                deltas=[]
                for participant in participants:
                    keep=np.asarray([r["participant_id"] != participant for r in records]) & valid[:,hi]
                    deltas.append(float(np.nanmean(values[keep,a,hi,mi]-values[keep,b,hi,mi])*(1 if m=="ce" else 100)))
                participant_sensitivity.append(dict(a="random_mean",b=other,horizon=h,metric=m,
                    leave_one_participant_out_min=min(deltas),leave_one_participant_out_max=max(deltas),
                    sign_stable=all(x>=0 for x in deltas) or all(x<=0 for x in deltas),n_participants=len(participants)))
    report=dict(protocol=PROTOCOL,is_partial=args.partial,population=dict(evaluated_n=len(records),sessions=len(sessions),participants=len(participants),
        valid_denominators=valid.sum(0).astype(int).tolist()), manifest=str(args.manifest),manifest_sha256=sha(args.manifest),
        masks=mask_audit,random_definition=manifest["random_definition"],input_runs=run_reports,estimates=estimates,comparisons=comparisons,
        participant_sensitivity=participant_sensitivity,bootstrap=dict(unit="session/video",clusters=len(sessions),draws=args.bootstrap,seed=20260908,
            interval="paired percentile95%; exploratory reused validation"),
        tie_audit=dict(boundary_tie_rows={arm:{f"{h:g}s":{f"top{k}":int(boundary[:,ai,hi,ki].sum()) for ki,k in enumerate([1,3,5])}
            for hi,h in enumerate(HORIZONS)} for ai,arm in enumerate(ARMS)}),
        gates=dict(exact_full_union=not args.partial,unique_rows=True,source_versions_identical=True,logits_recomputed=True,
                   masks_exact=True,constant_positions=True,native_labels=True))
    write_json(args.out/"summary.json",report)
    write_json(args.out/"primary.json",dict(population=report["population"],
        estimates=[e for e in estimates if e["horizon"]==2 and e["metric"]=="top5"],
        comparisons=[c for c in comparisons if c["horizon"]==2 and c["metric"]=="top5"],
        participant_sensitivity=[x for x in participant_sensitivity if x["horizon"]==2 and x["metric"]=="top5"],jobs=run_reports))
    np.savez_compressed(args.out/"paired_metrics.npz",values=values,valid=valid,arms=np.asarray(extended),metrics=np.asarray(METRICS),
                        horizons=np.asarray(HORIZONS),sample_ids=np.asarray([r["sample_id"] for r in records]),sessions=np.asarray([r["video_id"] for r in records]))
    fig,ax=plt.subplots(figsize=(9,4.8)); ix=np.arange(len(extended)); vals=mean[:,0,2]*100
    lo=np.quantile(boot[:,:,0,2],.025,axis=0)*100; hi=np.quantile(boot[:,:,0,2],.975,axis=0)*100
    ax.errorbar(ix,vals,yerr=np.vstack([vals-lo,hi-vals]),fmt="o",capsize=4,color="#245a84")
    ax.set_xticks(ix,extended,rotation=25,ha="right"); ax.set_ylabel("Native Action Top-5 @2s (%)"); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(args.out/"top5_2s.png",dpi=180); plt.close(fig)
    print(json.dumps(dict(rows=len(records),out=str(args.out),primary=str(args.out/"primary.json"))),flush=True)


if __name__ == "__main__": main()
