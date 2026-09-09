#!/usr/bin/env python3
"""CPU-only paired target-query evaluation and post-softmax cross-slot readouts.

Run inside the project Slurm/container environment. Readout deletion never
changes model attention outputs. All diagnostic normalizations are explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

import analyze_b18_q1_paired_hybrid as U

PROTOCOL = "b18-predictor-prune/egtea-ctx16-target-crossslot-v1"
HORIZONS = ["2s", "4s", "6s"]
TARGET = "hybrid_target_L11"
RANDOM = [f"hybrid_target_random_seed{s}" for s in (1701, 1702, 1703)]
ARMS = ["recent", "hybrid_allquery_L11", TARGET, *RANDOM, "hybrid_target_donor"]
KINDS = ["context_original", "context_offslot", "allquery_original", "allquery_residual", "targetquery"]
CONTRASTS = {
    "target_minus_allquery": {TARGET: 1, "hybrid_allquery_L11": -1},
    "target_minus_random_mean3": {TARGET: 1, **{a: -1/3 for a in RANDOM}},
    "target_minus_recent": {TARGET: 1, "recent": -1},
    "target_minus_donor_secondary": {TARGET: 1, "hybrid_target_donor": -1},
    **{f"target_minus_random_seed{s}_secondary": {TARGET: 1, f"hybrid_target_random_seed{s}": -1}
       for s in (1701, 1702, 1703)},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def close(a, b, message, rtol=.003, atol=.03):
    U.require(np.allclose(a, b, rtol=rtol, atol=atol), message)


def mean_weighted(parts, weights):
    return sum(np.asarray(x, dtype=np.float64)*w for x,w in zip(parts,weights))/sum(weights)


def load(runs, partial):
    runs = sorted(runs, key=lambda p: read_json(p/"metadata.json")["start"])
    metadata, summaries, rows, arrays, weights = [], [], [], [], []
    for path in runs:
        for name in ("metadata.json","summary.json","execution_order.json","predictions.jsonl",
                     "token_scores.npy","crossslot_profiles.npz"):
            U.require((path/name).is_file(), f"Missing completed artifact {path/name}")
        meta, summary = read_json(path/"metadata.json"), read_json(path/"summary.json")
        order = read_json(path/"execution_order.json")
        records = [json.loads(x) for x in (path/"predictions.jsonl").read_text().splitlines()]
        n = meta["stop"]-meta["start"]
        U.require(len(records)==len(order)==summary["n_rows"]==n, "Shard row-count mismatch")
        U.require(meta["start"]%4==meta["stop"]%4==0, "Split original four-row execution batch")
        U.require(meta["evaluation_protocol"]==summary["evaluation_protocol"]==PROTOCOL, "Protocol mismatch")
        U.require(meta["arms"]==ARMS and meta["score_block_order"]==[0,11] and
                  meta["score_kind_order"]==["all_query","target_query","context_offslot"] and
                  meta["diagnostic_kind_order"]==KINDS and meta["diagnostic_topk"]==4096, "Arm/score order mismatch")
        U.require(meta["query_group_order"]==["context","target"], "Query group order mismatch")
        for key,value in {"n_context_tokens":16384,"n_target_tokens":256,"tokens_per_slot":256,
                          "target_array_start_slot":64,"target_rope_start_slot":72}.items():
            U.require(meta["token_layout"][key]==value, f"Unexpected token layout {key}")
        for r,o in zip(records,order):
            U.require(all(r[k]==v for k,v in o.items()), "Prediction/execution identity mismatch")
        data = dict(np.load(path/"crossslot_profiles.npz"))
        h = data["raw_group_mass"].shape[3]
        shapes = {"raw_group_mass":(n,2,2,h,65), "offslot_context_mass":(n,2,h,65),
                  "conditional_context_mass":(n,2,h,65), "diagnostic_keep_per_slot":(n,2,5,64),
                  "mean_flow_raw":(2,h,65,65), "mean_flow_offslot":(2,h,64,65),
                  "mean_flow_conditional":(2,h,64,65), "mean_head_token_maps":(2,3,h,16384)}
        for name in ("diagonal_context_mass","diagonal_context_mass_fp32","self_token_context_mass","survival_mean","survival_min",
                     "survival_zero","survival_nearzero","original_row_sum_mean","original_row_sum_min"):
            shapes[name]=(n,2,h,64)
        for name,shape in shapes.items():
            U.require(data[name].shape==shape, f"Unexpected {name} shape {data[name].shape}")
            U.require(np.isfinite(data[name]).all() and (data[name]>=0).all(), f"Invalid {name}")
        U.require(all(meta["token_layout"]["num_heads"][str(b)]==h for b in (0,11)), "Head-count mismatch")
        raw,off,cond=data["raw_group_mass"].astype(np.float64),data["offslot_context_mass"].astype(np.float64),data["conditional_context_mass"].astype(np.float64)
        context=raw[:,:,0]
        U.require((off <= context+.003*context+.03).all(), "Raw offslot exceeds original context received mass")
        close(context[...,:64]-off[...,:64],data["diagonal_context_mass"],"Removed diagonal/context received mismatch")
        close(context[...,64:],off[...,64:],"Context-query target-key raw mass changed",rtol=2e-5,atol=.003)
        U.require((data["self_token_context_mass"]<=data["diagonal_context_mass_fp32"]*1.00001+.001).all(), "Self-token exceeds same-slot FP32 mass")
        close(256*(data["original_row_sum_mean"]-data["survival_mean"]),data["diagonal_context_mass_fp32"],
              "FP32 original/survival/same-slot mass mismatch",rtol=1e-5,atol=.002)
        close(context.sum(-1),16384,"Original context query mass invalid",rtol=.01)
        close(raw[:,:,1].sum(-1),256,"Original target query mass invalid",rtol=.01)
        close(cond.sum(-1),16384-data["survival_zero"].sum(-1),"Per-query-renormalized mass/count mismatch",rtol=1e-4)
        close(off.sum(-1),256*data["survival_mean"].sum(-1),"Raw survivor mass/denominator mismatch",rtol=.005)
        U.require((data["survival_zero"]<=data["survival_nearzero"]).all(),"Near-zero count excludes exact zeros")
        U.require((data["survival_nearzero"]<=256).all(),"Impossible survival count")
        U.require((data["survival_min"]<=data["survival_mean"]+1e-7).all(),"Survival min exceeds mean")
        for name in ("mean_flow_offslot","mean_flow_conditional"):
            U.require(np.max(np.abs(np.diagonal(data[name][...,:64],axis1=-2,axis2=-1)))==0,"Nonzero context-slot diagonal after deletion")
        close(data["mean_flow_raw"][...,:64,64:],data["mean_flow_offslot"][...,64:],"Flow target-key mass changed",rtol=2e-5,atol=1e-6)
        close(data["mean_flow_raw"][...,:64,:].sum(-1),data["original_row_sum_mean"].mean(0),"Raw row-sum flow mismatch",rtol=.005,atol=2e-4)
        close(data["mean_flow_offslot"].sum(-1),data["survival_mean"].mean(0),"Survival flow mismatch",rtol=.005,atol=2e-4)
        close(data["mean_flow_conditional"].sum(-1),1-data["survival_zero"].mean(0)/256,"Conditional flow count mismatch",rtol=1e-4,atol=1e-5)
        counts=data["diagnostic_keep_per_slot"]
        U.require(np.issubdtype(counts.dtype,np.integer) and (counts<=256).all() and (counts.sum(-1)==4096).all(),"Invalid diagnostic GPU top-K counts")
        scores=np.load(path/"token_scores.npy",mmap_mode="r")
        U.require(scores.shape==(n,2,3,16384) and scores.dtype==np.float32,"Token-score shape mismatch")
        U.require(np.isfinite(scores).all() and (scores>=0).all(),"Invalid token scores")
        temporal=scores.reshape(n,2,3,64,256).sum(-1,dtype=np.float64)
        close(temporal[:,:,0],raw[...,:64].sum((2,3)),"All-query token/profile mismatch")
        close(temporal[:,:,1],raw[:,:,1,:,:64].sum(2),"Target-query token/profile mismatch")
        close(temporal[:,:,2],off[...,:64].sum(2),"Offslot token/profile mismatch")
        close(data["mean_head_token_maps"].sum(2),scores.mean(0,dtype=np.float64),"Mean head-token map mismatch")
        parity=summary["parity"]
        U.require(parity["score_exact_batches"]==n//4,"Missing all-query reference parity batches")
        U.require(parity["anchor_ce_max_abs"]<=1e-5,"GPU anchor CE parity tolerance exceeded")
        if meta["arguments"]["parity_check"]:
            U.require(parity["same_mask_model_output_exact"] is True,"Missing direct original-forward parity")
        metadata.append(meta); summaries.append(summary);rows.extend(records); arrays.append(data);weights.append(n)
    common=metadata[0]
    keys=("evaluation_protocol","sample_manifest_protocol","manifest_sha256","arms","source_sha256",
          "train_csv_sha256","val_csv_sha256","checkpoint_paths","checkpoint_file_identity","reference_dir",
          "reference_metadata_sha256","token_layout","selection","conditional_definition","near_zero_threshold")
    for meta in metadata[1:]:
        U.require(all(meta[k]==common[k] for k in keys),"Cross-shard provenance mismatch")
    manifest_path=Path(common["manifest"])
    manifest_meta=read_json(manifest_path.with_suffix(".meta.json"))
    U.require(U.sha(manifest_path)==common["manifest_sha256"]==manifest_meta["manifest_sha256"],"Manifest bytes changed")
    U.require(manifest_meta["evaluation_protocol"]==common["sample_manifest_protocol"],"Sampling vs evaluation protocol conflated")
    for key in ("train_csv","val_csv"):
        U.require(U.sha(common["arguments"][key])==common[key+"_sha256"],f"CSV changed: {key}")
    U.require(common["val_csv_sha256"]==manifest_meta["source_csv_sha256"],"Manifest source hash mismatch")
    for path,identity in common["checkpoint_file_identity"].items():
        stat=Path(path).stat()
        U.require(identity=={"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns},"Checkpoint attributes changed")
    with manifest_path.open() as f: manifest_rows=list(csv.DictReader(f))
    with Path(common["arguments"]["val_csv"]).open() as f: source=list(csv.DictReader(f))
    vocabulary={}
    with Path(common["arguments"]["train_csv"]).open() as f:
        for row in csv.DictReader(f):
            for v,n,m in zip(row["mtp_verbs"].split(","),row["mtp_nouns"].split(","),row["mtp_mask"].split(",")):
                pair=(int(v),int(n))
                if float(m)>=.5 and min(pair)>=0 and pair not in vocabulary:vocabulary[pair]=len(vocabulary)
    expected=[]
    for entry in manifest_rows:
        i=int(entry["original_csv_row_index"]);r=source[i]
        U.require(hashlib.sha256(json.dumps(r,sort_keys=True,separators=(",",":")).encode()).hexdigest()==entry["row_sha256"],"Source row hash mismatch")
        expected.append({"source_index":i,"selection_index":int(entry["selection_index"]),
                         "sample_id":hashlib.sha256((r["video_id"]+"|"+r["frame_indices"]).encode()).hexdigest(),
                         "video_id":r["video_id"],"participant_id":entry["participant_id"]})
    ordered=sorted(expected,key=lambda r:(r["video_id"],r["selection_index"]))
    paired=[];half=len(ordered)//2
    for i in range(half):
        a,b=dict(ordered[i]),dict(ordered[i+half])
        a.update(donor_sample_id=b["sample_id"],pair_id=i);b.update(donor_sample_id=a["sample_id"],pair_id=i)
        paired.extend((a,b))
    positions=[i for m in metadata for i in range(m["start"],m["stop"])]
    U.require(len(set(positions))==len(rows)==len({r["sample_id"] for r in rows}),"Overlapping ranges or rows")
    if not partial:U.require(positions==list(range(len(paired))),"Incomplete full manifest")
    for i,r in zip(positions,rows):
        U.require(all(r[k]==v for k,v in paired[i].items()),"Unexpected manifest/donor identity")
        source_row=source[r["source_index"]]
        for hi,h in enumerate(HORIZONS):
            pair=(int(source_row["mtp_verbs"].split(",")[hi]),int(source_row["mtp_nouns"].split(",")[hi]))
            state="masked" if float(source_row["mtp_mask"].split(",")[hi])<=.5 else "valid" if pair in vocabulary else "out_of_training_vocabulary"
            U.require(r["label_validity"][h]==state,"Source-derived validity mismatch")
            if state=="valid":U.require(all(r["arms"][a]["metrics"][h]["label"]==vocabulary[pair] for a in ARMS),"Source-derived label mismatch")
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        U.require(a["donor_sample_id"]==b["sample_id"] and b["donor_sample_id"]==a["sample_id"] and a["video_id"]!=b["video_id"],"Broken cross-video donor pair")
    reference_dir=Path(common["reference_dir"])
    U.require(U.sha(reference_dir/"metadata.json")==common["reference_metadata_sha256"],"Reference metadata changed")
    reference={r["sample_id"]:r for r in (json.loads(x) for x in (reference_dir/"predictions.jsonl").read_text().splitlines())}
    refscore=np.load(reference_dir/"scores.npy",mmap_mode="r")
    for path,meta in zip(runs,metadata):
        score=np.load(path/"token_scores.npy",mmap_mode="r")
        U.require(np.array_equal(score[:,:,0],refscore[meta["start"]:meta["stop"]]),"Stored all-query score not exact reference")
    for r in rows:
        for current,old in (("recent","recent"),("hybrid_allquery_L11","hybrid_online_L11")):
            metrics=r["arms"][current]["metrics"];prior=reference[r["sample_id"]]["arms"][old]["metrics"]
            U.require(set(metrics)==set(prior),"Reference horizon validity differs")
            for h,v in metrics.items():
                U.require(v["label"]==prior[h]["label"] and v["top5"]==prior[h]["top5"] and abs(v["ce"]-prior[h]["ce"])<=1e-5,"Reference anchor output mismatch")
    joined={name:(mean_weighted([d[name] for d in arrays],weights) if name.startswith("mean_") else np.concatenate([d[name] for d in arrays])) for name in arrays[0]}
    coverage={"rows":len(rows),"manifest_rows":len(paired),"complete":len(rows)==len(paired),
              "sessions":len({r["video_id"] for r in rows}),"participant_ids":len({r["participant_id"] for r in rows}),
              "donor_pairs":len(rows)//2,"ranges":[[m["start"],m["stop"]] for m in metadata],
              "manifest_sha256":common["manifest_sha256"],"val_csv_sha256":common["val_csv_sha256"],
              "reference_anchor_parity":"All stored all-query scores exactly equal prior full4000; both anchor Top5/labels equal on every valid horizon, CE abs<=1e-5.",
              "checkpoint_identity":"Paths/byte sizes/mtime_ns agree; no checkpoint content hash recorded."}
    return runs,common,metadata,summaries,rows,joined,coverage


def outcomes(rows,metas,summaries):
    correct=np.zeros((len(rows),3,len(ARMS)));loss=np.zeros_like(correct)
    valid=np.zeros((len(rows),3),bool);kept=np.zeros((len(rows),len(ARMS),64),np.int16)
    exclusions={h:Counter() for h in HORIZONS}
    for ri,r in enumerate(rows):
        U.require(set(r["arms"])==set(ARMS),"Missing/extra arm")
        for hi,h in enumerate(HORIZONS):
            state=r["label_validity"][h];valid[ri,hi]=state=="valid";exclusions[h][state]+=1
        for ai,a in enumerate(ARMS):
            item=r["arms"][a];counts=np.asarray(item["keep_per_slot"])
            U.require(counts.shape==(64,) and np.issubdtype(counts.dtype,np.integer) and (counts>=0).all() and (counts<=256).all() and counts.sum()==4096,"Bad arm budget")
            kept[ri,ai]=counts
            if a=="recent":U.require((counts[:48]==0).all() and (counts[48:]==256).all(),"Recent pattern mismatch")
            else:U.require((counts[52:]==256).all() and counts[:52].sum()==1024,"Hybrid anchor mismatch")
            for hi,h in enumerate(HORIZONS):
                m=item["metrics"].get(h)
                U.require((m is not None)==valid[ri,hi],"Arm validity mismatch")
                if m is not None:
                    U.require(isinstance(m["top5"],bool) and np.isfinite(m["ce"]) and m["ce"]>=0,"Invalid metric")
                    correct[ri,hi,ai]=m["top5"];loss[ri,hi,ai]=m["ce"]
        for a in RANDOM+["hybrid_target_donor"]:
            U.require(np.array_equal(kept[ri,ARMS.index(a)],kept[ri,ARMS.index(TARGET)]),"Matched temporal counts differ")
    cursor=0
    for meta,summary in zip(metas,summaries):
        n=meta["stop"]-meta["start"];sl=slice(cursor,cursor+n)
        U.require(summary["parity"]["anchor_top5_equal_examples"]==2*int(valid[sl].sum()),"Incomplete GPU anchor parity count")
        for ai,a in enumerate(ARMS):
            for hi,h in enumerate(HORIZONS):
                s=summary["results"][a]
                U.require(s.get(f"n@{h}",0)==valid[sl,hi].sum() and s.get(f"correct@{h}",0)==correct[sl,hi,ai].sum(),"Summary count mismatch")
                close(s.get(f"ce_sum@{h}",0),loss[sl,hi,ai].sum(),"Summary CE mismatch",rtol=1e-7,atol=1e-5)
        cursor+=n
    result={a:{h:{"n":int(valid[:,hi].sum()),"correct":int(correct[:,hi,ai].sum()),
                  "action_top5":float(correct[:,hi,ai].sum()/valid[:,hi].sum()),
                  "mean_ce":float(loss[:,hi,ai].sum()/valid[:,hi].sum())} for hi,h in enumerate(HORIZONS)} for ai,a in enumerate(ARMS)}
    selection={a:{"mean_keep_per_slot":kept[:,ai].mean(0).tolist(),
                  "outside_recent16_count":U.distribution(kept[:,ai,:48].sum(-1)),
                  "displaced_recent4_count":U.distribution(kept[:,ai,48:52].sum(-1)),
                  "recent8_count":U.distribution(kept[:,ai,56:].sum(-1)),
                  "recent16_count":U.distribution(kept[:,ai,48:].sum(-1))} for ai,a in enumerate(ARMS)}
    return correct,loss,valid,result,selection,{h:dict(x) for h,x in exclusions.items()}


def paired_bootstrap(rows,correct,loss,valid,reps,seed):
    names=list(CONTRASTS)
    w=np.array([[CONTRASTS[name].get(a,0) for a in ARMS] for name in names])
    values=np.stack([correct@w.T,loss@w.T],-1)
    point=values.sum(0)/valid.sum(0)[:,None,None]
    result={name:{h:{"n":int(valid[:,hi].sum()),"top5_delta_pp":float(100*point[hi,ci,0]),"ce_delta":float(point[hi,ci,1])} for hi,h in enumerate(HORIZONS)} for ci,name in enumerate(names)}
    for key,offset in (("video_id",0),("participant_id",1)):
        clusters=sorted({r[key] for r in rows});mapping={v:i for i,v in enumerate(clusters)}
        idx=np.array([mapping[r[key]] for r in rows]);g=len(clusters)
        sums=np.zeros((g,)+values.shape[1:]);denom=np.zeros((g,3))
        np.add.at(sums,idx,values);np.add.at(denom,idx,valid)
        rng=np.random.default_rng(seed+offset);samples=np.empty((reps,)+point.shape)
        for start in range(0,reps,256):
            stop=min(start+256,reps);draw=rng.multinomial(g,np.full(g,1/g),size=stop-start)
            numerator=(draw@sums.reshape(g,-1)).reshape((stop-start,)+point.shape)
            denominator=draw@denom;U.require((denominator>0).all(),"Empty bootstrap horizon")
            samples[start:stop]=numerator/denominator[:,:,None,None]
        ci=np.quantile(samples,[.025,.975],axis=0)
        for ni,name in enumerate(names):
            for hi,h in enumerate(HORIZONS):
                result[name][h][key]={"cluster_n":g,"top5_95ci_pp":(100*ci[:,hi,ni,0]).tolist(),
                                     "ce_95ci":ci[:,hi,ni,1].tolist(),"top5_bootstrap_se_pp":float(100*samples[:,hi,ni,0].std(ddof=1))}
    return {"contrasts":result,"primary":names[:3],"primary_horizon":"2s","reps":reps,"seed":seed,
            "method":"Paired cluster percentile bootstrap; each replicate uses ratio of resampled outcome sums to valid counts, sharing cluster draws across arms.",
            "limits":["Repeated exploratory reuse of the same validation manifest; not independent confirmation.",
                      "Marginal95% intervals, no simultaneous coverage for multiple contrasts/horizons.",
                      "Donor intervals condition on fixed pairing and omit full shared-source dependency and reassignment uncertainty.",
                      "Random seeds averaged within receiver, not independent validation rows; three-seed SD is descriptive."]}


def temporal_summary(raw,query_count):
    context=np.asarray(raw[...,:64],dtype=np.float64)
    return {"raw_context_probability_per_query_head":U.distribution(context.sum(-1)/query_count),
            "raw_target_probability_per_query_head":U.distribution(np.asarray(raw[...,64:],dtype=np.float64).sum(-1)/query_count),
            "pooled_raw_context_profile_per_query_head":(context.sum((0,1))/(len(context)*context.shape[1]*query_count)).tolist(),
            "per_sample_per_head_context_conditional":U.concentration(context),
            "per_sample_head_pooled_context_conditional":U.concentration(context.sum(1)),
            "pooled_context_conditional_profile":U.normalized(context.sum((0,1))).tolist(),
            "pooling":U.pooling_decomposition(context),
            "denominator":query_count,"normalization":"Raw: divide by unchanged original query count per head. Conditional: normalize received mass over64 context keys after aggregation. Per-query renormalization, where named, occurred separately before this readout."}


def diagnostics(data):
    result={"definition":"C=context-query received, T=target-query received, O=C after zeroing same-time-slot contextQ/contextK blocks post-softmax. Compare C→O and C+T→O+T with query source unchanged.",
            "layout":"Context key slots0–63; appended target array index64/RoPE start72. Cross-slot includes adjacent slots.",
            "self_vs_same_slot_dtype":"Self-token and same-slot removal fractions both usefloat32 sums of the sameBF16 softmax values. A separate legacyBF16 query-summed diagonal verifies rawreceived conservation."}
    for bi,b in enumerate((0,11)):
        raw=data["raw_group_mass"][:,bi].astype(np.float64);c,t=raw[:,0],raw[:,1]
        off=data["offslot_context_mass"][:,bi].astype(np.float64)
        conditional=data["conditional_context_mass"][:,bi].astype(np.float64)
        maps={"context_original":(c,16384),"context_offslot":(off,16384),
              "allquery_original":(c+t,16640),"allquery_residual":(off+t,16640),
              "targetquery":(t,256),"context_per_query_renorm":(conditional,16384),
              "allquery_per_query_renorm":(conditional+t,16640)}
        block={"temporal":{name:temporal_summary(v,q) for name,(v,q) in maps.items()}}
        original=data["original_row_sum_mean"][:,bi].sum(-1,dtype=np.float64)*256
        removed=data["diagonal_context_mass_fp32"][:,bi].sum(-1,dtype=np.float64)
        selfmass=data["self_token_context_mass"][:,bi].sum(-1,dtype=np.float64)
        context_destination=c[...,:64].sum(dtype=np.float64)
        block["mass_accounting"]={
            "original_raw_probability_sum":float(original.sum()),
            "same_slot_removed_share_of_original_context_query_mass":float(removed.sum()/original.sum()),
            "exact_self_token_share_of_original_context_query_mass":float(selfmass.sum()/original.sum()),
            "other_same_slot_share_of_original_context_query_mass":float((removed-selfmass).sum()/original.sum()),
            "same_slot_removed_share_conditioned_on_context_destination":float(removed.sum()/context_destination),
            "exact_self_token_share_conditioned_on_context_destination":float(selfmass.sum()/context_destination),
            "other_same_slot_share_conditioned_on_context_destination":float((removed-selfmass).sum()/context_destination),
            "self_token_share_of_removed_same_slot_mass":float(selfmass.sum()/removed.sum()),
            "conditional_destination_denominator":"Global raw received mass from context queries to64context key slots; legacyBF16 query reduction. Numerators useFP32 sums of the same softmaxvalues; small reduction differences tolerated. This global ratio is not the prior per-head/query-slot mean of conditioned ratios.",
            "remaining_all_key_share":float(256*data["survival_mean"][:,bi].sum(dtype=np.float64)/original.sum()),
            "remaining_all_key_share_legacy_received":float(off.sum()/original.sum()),
            "remaining_context_key_share":float(off[...,:64].sum()/original.sum()),
            "unchanged_target_key_share":float(off[...,64:].sum()/original.sum()),
            "same_slot_removed_per_sample_head_share":U.distribution(removed/original),
            "exact_zero_survival_queries":int(data["survival_zero"][:,bi].sum()),
            "near_zero_survival_queries":int(data["survival_nearzero"][:,bi].sum()),
            "minimum_survival":float(data["survival_min"][:,bi].min()),
            "per_slot_mean_survival_distribution":U.distribution(data["survival_mean"][:,bi]),
            "caveat":"Global raw removed share differs from averaging context-conditioned query-slot profiles. Do not reconstruct it from the earlier conditional same-slot percentage."}
        block["diagnostic_topk"]={name:{"mean_keep_per_slot":data["diagnostic_keep_per_slot"][:,bi,ki].mean(0).tolist(),
                                               "recent16_mean_count":float(data["diagnostic_keep_per_slot"][:,bi,ki,48:].sum(-1).mean()),
                                               "topk":4096,"scope":"Actual GPU selection on diagnostic score only; no corresponding classifier intervention"} for ki,name in enumerate(KINDS)}
        flows={"original":data["mean_flow_raw"][bi,:,:64,:],"offslot":data["mean_flow_offslot"][bi],"per_query_renorm":data["mean_flow_conditional"][bi]}
        q,k=np.indices((64,64));distance=abs(q-k)
        block["flow"]={}
        for name,f in flows.items():
            context=f[...,:64];pooled=context.mean(0);cond=U.normalized(context)
            block["flow"][name]={"head_pooled_raw_context_matrix":pooled.tolist(),
                                 "head_pooled_row_context_conditional_matrix":U.normalized(pooled).tolist(),
                                 "raw_context_mass_per_source_query":float(context.sum(-1).mean()),
                                 "raw_target_mass_per_source_query":float(f[...,64:].sum(-1).mean()),
                                 "same_slot_raw":float((context*(distance==0)).sum(-1).mean()),
                                 "adjacent_only_raw":float((context*(distance==1)).sum(-1).mean()),
                                 "distance_ge8_raw":float((context*(distance>=8)).sum(-1).mean()),
                                 "adjacent_only_conditional":float((cond*(distance==1)).sum(-1).mean()),
                                 "distance_ge8_conditional":float((cond*(distance>=8)).sum(-1).mean()),
                                 "aggregation":"Saved flow averages actual query probabilities within each256-query slot then overexamples; matrix shown head-pooled. Row context conditioning is after pooling and is not the per-query sensitivity transform.",
                                 "conditional_scalar_order":"For each head and source query slot, normalize its example-mean row over context keys; then average scalars across heads/source slots.",
                                 "conditional_matrix_order":"First average example-mean flow matrices across heads; then normalize each pooled source-slot row over context keys. This conditioning order differs from the conditional scalars."}
        result[f"L{b}"]=block
    return result


def joint_score_diagnostics(runs):
    accum={k:np.zeros((2,16384)) for k in KINDS};normacc={k:np.zeros((2,16384)) for k in KINDS}
    ent={k:[] for k in KINDS};mass={k:[] for k in KINDS};spatial_examples=[]
    for path in runs:
        scores=np.load(path/"token_scores.npy",mmap_mode="r")
        order=read_json(path/"execution_order.json")
        for start in range(0,len(scores),32):
            x=np.array(scores[start:start+32],dtype=np.float64)
            values={"context_original":x[:,:,0]-x[:,:,1],"context_offslot":x[:,:,2],
                    "allquery_original":x[:,:,0],"allquery_residual":x[:,:,2]+x[:,:,1],"targetquery":x[:,:,1]}
            for name,v in values.items():
                U.require(v.min()>=-1e-3,"Negative algebraic context score beyond rounding")
                v=np.maximum(v,0);s=v.sum(-1);p=v/s[...,None]
                accum[name]+=v.sum(0);normacc[name]+=p.sum(0)
                ent[name].append(-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(16384));mass[name].append(s)
            if len(spatial_examples)<2:
                for j in range(min(2-len(spatial_examples),len(x))):
                    spatial_examples.append({"sample_id":order[start+j]["sample_id"],"video_id":order[start+j]["video_id"],
                        "selection":"First two execution rows, fixed before outcome inspection; descriptive examples only",
                        "maps":{name:[U.normalized(v[j,bi].reshape(64,256).sum(0)).reshape(16,16).tolist() for bi in range(2)] for name,v in values.items()}})
    output={"units":"Joint entropy divided bylog16384; pooling weights proportional to raw context-key received mass",
            "examples":spatial_examples}
    for bi,b in enumerate((0,11)):
        output[f"L{b}"]={}
        for name in KINDS:
            e=np.concatenate(ent[name])[:,bi];w=np.concatenate(mass[name])[:,bi];w=w/w.sum()
            pooled=U.normalized(accum[name][bi]);equal=U.normalized(normacc[name][bi])
            output[f"L{b}"][name]={"mean_per_sample_entropy":float(e.mean()),"raw_pooled_entropy":float(U.entropy(pooled)),
                                    "raw_weighted_sample_pooling_js":float(U.entropy(pooled)-w@e),
                                    "equal_sample_pooling_js":float(U.entropy(equal)-e.mean()),
                                    "raw_pooled_spatial_context_conditional":pooled.reshape(64,256).sum(0).reshape(16,16).tolist()}
    return output


def plots(out,selection,diag,joint):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slots=np.arange(64)
    fig,ax=plt.subplots(figsize=(9,4))
    for name in ("recent","hybrid_allquery_L11",TARGET):ax.plot(slots,selection[name]["mean_keep_per_slot"],label=name)
    ax.axvline(51.5,color="gray",lw=1);ax.set(xlabel="Context slot:0oldest→63newest",ylabel="Retained tokens per slot",title="Accuracy arms: actual GPU selection, mean across manifest rows")
    ax.legend();fig.tight_layout();fig.savefig(out/"accuracy_arm_retained_counts.png",dpi=160);plt.close(fig)
    settings=[("temporal_absolute", "pooled_raw_context_profile_per_query_head", "Probability per original source query/head; context keys only"),
              ("temporal_conditional", "pooled_context_conditional_profile", "Received mass conditioned on64context key slots after raw pooling")]
    for filename,key,label in settings:
        fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True)
        for bi,b in enumerate((0,11)):
            for col,names in enumerate((("context_original","context_offslot","context_per_query_renorm"),("allquery_original","allquery_residual","targetquery"))):
                ax=axes[bi,col]
                for name in names:ax.plot(slots,diag[f"L{b}"]["temporal"][name][key],label=name)
                ax.set_title(f"L{b}: "+("context query source" if col==0 else "all-query residual and target source"));ax.legend(fontsize=7)
                ax.set_xlabel("Context key slot");ax.set_ylabel("Probability / context-slot probability")
        fig.suptitle(label+"\nRenorm curve: each context query normalized over all surviving keys before aggregation",fontsize=10)
        fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/(filename+".png"),dpi=160);plt.close(fig)
    for mode,key in (("absolute","head_pooled_raw_context_matrix"),("conditional","head_pooled_row_context_conditional_matrix")):
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for bi,b in enumerate((0,11)):
            matrices=[np.array(diag[f"L{b}"]["flow"][name][key]) for name in ("original","offslot","per_query_renorm")]
            vmax=max(m.max() for m in matrices)
            for col,(name,m) in enumerate(zip(("original context","same-slot zeroed","per-query renorm sensitivity"),matrices)):
                ax=axes[bi,col];im=ax.imshow(m,origin="lower",vmin=0,vmax=vmax,aspect="auto")
                ax.set(xlabel="Context key slot",ylabel="Context query slot",title=f"L{b}: {name}");fig.colorbar(im,ax=ax,fraction=.045)
        fig.suptitle("Query-slot flow: "+("absolute probability averaged over heads/examples" if mode=="absolute" else "each head-pooled mean row conditioned on context keys")+"\nColumns2/3exclude the entire same-slot temporalblock; adjacent slots remain",fontsize=10)
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/f"crossslot_flow_{mode}.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for bi,b in enumerate((0,11)):
        for col,names in enumerate((("context_original","context_offslot"),("allquery_original","allquery_residual","targetquery"))):
            for name in names:axes[bi,col].plot(slots,diag[f"L{b}"]["diagnostic_topk"][name]["mean_keep_per_slot"],label=name)
            axes[bi,col].set(xlabel="Context key slot",ylabel="Mean retained token count",title=f"L{b}: diagnostic topK4096 (no accuracy arm)");axes[bi,col].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"diagnostic_topk_counts.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for bi,b in enumerate((0,11)):
        names=("context_original","context_offslot","targetquery")
        matrices=[np.array(joint[f"L{b}"][name]["raw_pooled_spatial_context_conditional"]) for name in names]
        vmax=max(m.max() for m in matrices)
        for col,(name,m) in enumerate(zip(names,matrices)):
            im=axes[bi,col].imshow(m,vmin=0,vmax=vmax);axes[bi,col].set_title(f"L{b}: {name}");fig.colorbar(im,ax=axes[bi,col],fraction=.045)
    fig.suptitle("Spatial received-score marginal: raw pool across samples/heads/time, normalize over256spatial positions\nContext-key-conditioned; sources differ for targetquery column; shape alone does not establish utility",fontsize=10)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/"pooled_spatial_maps.png",dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--runs",nargs="+",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--allow-partial",action="store_true");p.add_argument("--bootstrap-reps",type=int,default=9999)
    p.add_argument("--bootstrap-seed",type=int,default=2026090521);p.add_argument("--no-plots",action="store_true")
    args=p.parse_args();U.require(args.bootstrap_reps>=199,"Need at least199bootstrap draws")
    source_hash={str(Path(__file__).resolve()):U.sha(__file__),str(Path(U.__file__).resolve()):U.sha(U.__file__)}
    runs,common,metas,summaries,rows,data,coverage=load(args.runs,args.allow_partial)
    correct,loss,valid,result,selection,exclusions=outcomes(rows,metas,summaries)
    paired=paired_bootstrap(rows,correct,loss,valid,args.bootstrap_reps,args.bootstrap_seed)
    diag=diagnostics(data);joint=joint_score_diagnostics(runs)
    random={h:{"seed_top5":[result[a][h]["action_top5"] for a in RANDOM],
               "mean_top5":float(np.mean([result[a][h]["action_top5"] for a in RANDOM])),
               "seed_sd_top5_pp":float(100*np.std([result[a][h]["action_top5"] for a in RANDOM],ddof=1))} for h in HORIZONS}
    report={"evaluation_protocol":PROTOCOL,"analysis_job_id":os.environ.get("SLURM_JOB_ID"),"analysis_source_sha256":source_hash,
            "input_jobs":[m["job_id"] for m in metas],"input_dirs":[str(x) for x in runs],"metric_scope":"native",
            "eval_path":common["eval_path"],"coverage":coverage,"exclusions":exclusions,"results":result,
            "paired_inference":paired,"random_seed_summary":random,"selection":selection,"crossslot_diagnostics":diag,
            "joint_scores":joint,"smoke_only":args.allow_partial,
            "limits":["Readout deletion/renormalization does not modify model outputs or test causal utility of the surviving pattern.",
                      "All arms share full-context encoder outputs; older information may already exist in recent-position tokens.",
                      "Cross-slot includes neighboring slots and is not synonymous with long-range memory.",
                      "Same validation manifest reused after earlier inspection; this is exploratory follow-up, not untouched confirmation."]}
    args.out_dir.mkdir(parents=True,exist_ok=True);target=args.out_dir/"summary.json"
    U.require(not target.exists(),"Refusing to overwrite analysis artifacts")
    target.write_text(json.dumps(report,indent=2)+"\n")
    brief={"summary":str(target),"coverage":coverage,"results_2s":{a:v["2s"] for a,v in result.items()},
           "contrasts_2s":{k:v["2s"] for k,v in paired["contrasts"].items()},
           "mass_accounting":{f"L{b}":diag[f"L{b}"]["mass_accounting"] for b in (0,11)},
           "smoke_only":args.allow_partial}
    (args.out_dir/"brief.json").write_text(json.dumps(brief,indent=2)+"\n")
    (args.out_dir/"merged_execution_order.json").write_text(json.dumps([{k:r[k] for k in ("sample_id","video_id","participant_id","source_index","selection_index","donor_sample_id","pair_id")} for r in rows],indent=2)+"\n")
    if not args.no_plots:plots(args.out_dir,selection,diag,joint)
    print(json.dumps(brief,indent=2))


if __name__=="__main__":main()
