#!/usr/bin/env python3
"""Verb / noun / action val metrics for a --no-kv (vanilla clip -> probe) checkpoint.

The trainer's evaluate() logs only action Top-5/Top-1. This rebuilds the
model from the checkpoint's saved CLI args (V-JEPA 2.0, or 2.1 with
``--vjepa21 1``), loads probe + trainable LoRA params the way the trainer
resumes, and scores the val split with the trainer's own Dataset, class maps
and label filter (``map_labels``), single GPU, batch 1, no sampler padding.

Reported per head (verb / noun / action), in %:
  * micro Top-1/3/5 accuracy (Top-1/5 action = trainer's logged metric, up to
    DistributedSampler padding);
  * class-mean recall@1/@3/@5 (mean over ground-truth classes present in val);
  * verb/noun Top-1/5 derived from the action head (softmax over actions,
    marginalised onto verb / noun).

  python scripts/eval_kvprune_nokv_vna.py --ckpt RUN/best.pt --video-root VIDEOS \
      [--val-csv CSV] --out result.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import run_local_kvprune_joint_2s as T


def build(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    g = ck["args"]
    if not int(g.get("no_kv", 0)) or int(g.get("rope", 0)) or int(g.get("gaze_imu", 0)):
        raise SystemExit("checkpoint is not a --no-kv --rope 0 RGB run")
    vjepa21 = bool(int(g.get("vjepa21", 0))) or "vjepa2_1" in Path(str(g["checkpoint"])).name
    maps = T.build_class_maps(Path(g["train_csv"]))
    verb_map, noun_map, action_map = maps
    backbone = T.build_model(device, int(g["frames"]), int(g["fps"]), int(g["img_size"]),
                             str(g["checkpoint"]), no_predictor=True, vjepa21=vjepa21)
    for p in backbone.parameters():
        p.requires_grad = False
    T.inject_encoder_lora(backbone, rank=int(g["lora_rank"]), alpha=float(g["lora_alpha"]),
                          dropout=0.0, last_n_blocks=12, target_suffixes=("attn.qkv", "attn.proj"))
    clf = T.AttentiveClassifier(verb_classes=verb_map, noun_classes=noun_map,
                                action_classes=action_map, embed_dim=int(backbone.embed_dim),
                                num_heads=int(g["probe_heads"]), depth=int(g["probe_depth"]),
                                use_activation_checkpointing=False).to(device)
    bundle = T.JointBundle(backbone, clf, chunk=int(g["chunk"]), stream_steps=1, protect_hist=0,
                           protect_k_frames=34, no_kv=True, self_only=False, prune_mode="last",
                           score_hist=1).to(device)
    pk = bundle.clf.load_state_dict(ck["probe"], strict=False)
    assert not pk.missing_keys and not pk.unexpected_keys, (pk.missing_keys[:5], pk.unexpected_keys[:5])
    bk = bundle.backbone.load_state_dict(ck["backbone"], strict=False)
    assert not [k for k in bk.missing_keys if "lora" in k] and not bk.unexpected_keys
    bundle.eval()
    meta = {k: ck.get(k) for k in ("epoch", "global_step", "best_top5")}
    meta.update({k: g.get(k) for k in ("checkpoint", "img_size", "frames", "fps", "horizon",
                                       "train_csv", "val_csv")})
    meta["vjepa21"] = vjepa21
    return bundle, maps, meta, g


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, default=None, help="default: the run's --val-csv")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    t0 = time.time()
    bundle, maps, meta, g = build(args.ckpt, device)
    verb_map, noun_map, action_map = maps
    val_csv = args.val_csv or Path(g["val_csv"])
    ds = T.ClipAnticipationDataset(val_csv, args.video_root, horizon_sec=float(g["horizon"]),
                                   model_fps=float(g["fps"]), img_size=int(g["img_size"]),
                                   n_model=int(g["frames"]))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         **T.loader_kwargs(args.num_workers))
    # action_map keys are (verb, noun) raw ids; map to verb/noun head indices.
    act_v = {a: verb_map[v] for (v, n), a in action_map.items()}
    act_n = {a: noun_map[n] for (v, n), a in action_map.items()}
    av = torch.tensor([act_v[a] for a in range(len(action_map))], device=device)
    an = torch.tensor([act_n[a] for a in range(len(action_map))], device=device)

    hits = defaultdict(float)
    per_class = {h: defaultdict(lambda: [0, 0, 0, 0]) for h in ("verb", "noun", "action")}
    n = 0
    for batch in loader:
        clips = T.normalize_clip(batch["clip"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = bundle(clips)
        v, nn_, a, keep = T.map_labels(batch["verb"].to(device), batch["noun"].to(device),
                                       verb_map, noun_map, action_map, device)
        if not keep:
            continue
        labs = {"verb": v, "noun": nn_, "action": a}
        for head, lab in labs.items():
            logits = out[head][keep].float()
            order = logits.argsort(dim=-1, descending=True)
            for k in (1, 3, 5):
                hit = (order[:, :k] == lab[:, None]).any(-1)
                hits[f"{head}_top{k}"] += float(hit.sum())
                slot = {1: 0, 3: 1, 5: 2}[k]
                for c, h in zip(lab.tolist(), hit.tolist()):
                    per_class[head][c][slot] += int(h)
            for c in lab.tolist():
                per_class[head][c][3] += 1
        # verb / noun derived from the action head.
        pa = out["action"][keep].float().softmax(-1)
        pv = torch.zeros(pa.size(0), len(verb_map), device=device).index_add_(1, av, pa)
        pn = torch.zeros(pa.size(0), len(noun_map), device=device).index_add_(1, an, pa)
        for head, prob, lab in (("verb_from_action", pv, v), ("noun_from_action", pn, nn_)):
            order = prob.argsort(dim=-1, descending=True)
            for k in (1, 5):
                hits[f"{head}_top{k}"] += float((order[:, :k] == lab[:, None]).any(-1).sum())
        n += len(keep)

    res = {"n": n}
    for key, val in sorted(hits.items()):
        res[key] = round(100.0 * val / n, 4)
    for head, pc in per_class.items():
        for slot, k in ((0, 1), (1, 3), (2, 5)):
            res[f"{head}_classmean_recall{k}"] = round(
                100.0 * sum(c[slot] / c[3] for c in pc.values()) / len(pc), 4)
        res[f"{head}_n_classes_in_val"] = len(pc)
    result = {"ckpt": str(args.ckpt), "ckpt_meta": meta, "val_csv": str(val_csv),
              "val_items": len(ds), "metrics": res,
              "note": "micro accuracy and class-mean recall over class-eligible val samples "
                      "(map_labels keep); single process, no DistributedSampler padding",
              "seconds": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
