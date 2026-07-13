#!/usr/bin/env python
"""Presentation overlay: kept-token mask on real frames + ground-truth and predicted verb/noun/action.

Uses the cached `_idx` (kept token positions) + re-decodes the exact clip (compute_clip_window +
front edge-pad to 480), so display frame p ↔ encoder slot p//2 (tubelet=2). Kept 16x16 patches are
shown at full brightness, dropped patches dimmed. If a checkpoint is given, also runs the trained
model on the cached tokens and prints GT vs predicted verb/noun/action in the title.
"""
import glob
import os
import random

import numpy as np
import torch

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M  # noqa: E402

CACHE = os.environ.get("IDX_CACHE", f"{SHARED}/data/preproc_cache_vjepa/nf480_fps8.0_px256_keep4096_idx")
VIDEO_ROOT = f"{SHARED}/data/hdepic_vjepa_videos"
ANN = f"{SHARED}/data/hdepic_vjepa_annotations/phd_split"
CLS = f"{SHARED}/data/hd-epic-annotations/narrations-and-action-segments"
OUT = os.environ.get("OUT_DIR", f"{SHARED}/outputs/prune_pattern")
CKPT = os.environ.get("CKPT", f"{SHARED}/outputs/vjepa_prune_anticipation/b12_1min_truepos/b12_vjepa_1min_truepos-best.pt")
POSMODE = os.environ.get("POSITION_MODE", "true")
GRID, SLOTS, PATCH, IMG = 16, 240, 16, 256
GP = GRID * GRID
N_CLIPS = int(os.environ.get("N_CLIPS", "4"))
N_FRAMES_SHOW = 6
os.makedirs(OUT, exist_ok=True)


def parse(fname):
    pid, vid, sf = os.path.basename(fname)[:-3].split("__")
    return pid, vid, int(sf)


def decode_padded(pid, vid, start_frame):
    from decord import VideoReader, cpu
    vr = VideoReader(f"{VIDEO_ROOT}/{pid}/{vid}.MP4", num_threads=1, ctx=cpu(0), width=IMG, height=IMG)
    win = np.clip(M.compute_clip_window(start_frame, vr.get_avg_fps(), 480, 8.0), 0, len(vr) - 1)
    frames = vr.get_batch(win).asnumpy()
    if frames.shape[0] < 480:
        frames = np.concatenate([np.repeat(frames[:1], 480 - frames.shape[0], 0), frames], 0)
    return frames


def main():
    import csv
    # GT rows keyed by (video_id, start_frame)
    gt = {}
    for split in ("train", "val", "test"):
        for r in csv.DictReader(open(f"{ANN}/HD_EPIC_{split}_vjepa.csv", newline="")):
            gt[(r["video_id"], int(r["start_frame"]))] = r
    verb_names = M.load_class_vocab(f"{CLS}/HD_EPIC_verb_classes.csv")
    noun_names = M.load_class_vocab(f"{CLS}/HD_EPIC_noun_classes.csv")

    device = torch.device("cuda")
    model = M.build_model(device, 480, 8, IMG, f"{SHARED}/checkpoints/vitl.pt")
    gp = int(model.grid_size) ** 2
    M.apply_lora_to_predictor(model.predictor, rank=8, alpha=16.0)
    model._position_mode = POSMODE
    model._full_n = (480 // int(model.tubelet_size)) * gp
    if POSMODE == "true":
        model.predictor.num_patches = ((model._full_n // gp) + 8) * gp
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    nv = ck["probe"]["verb_head.weight"].shape[0]
    nn_ = ck["probe"]["noun_head.weight"].shape[0]
    na = ck["probe"]["action_head.weight"].shape[0]
    probe = M.HDEpicProbe(model.embed_dim, nv, nn_, na).to(device)
    probe.load_state_dict(ck["probe"]); probe.eval()
    model.predictor.load_state_dict(ck["predictor_lora"], strict=False); model.predictor.eval()
    inv_action = {v: k for k, v in ck["action_map"].items()}   # id -> (verb_class, noun_class)
    print(f"[pred] loaded {CKPT} (verbs={nv} nouns={nn_} actions={na}, mode={POSMODE})", flush=True)

    @torch.no_grad()
    def predict(tok, idx):
        t = tok.unsqueeze(0).float().to(device)
        i = idx.unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = M.anticipate_from_ctx(model, t, 1.0, ctx_idx=i)
            vl, nl, al = probe(feats.float())
        return int(vl.argmax()), int(nl.argmax()), int(al.argmax())

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    random.seed(1)
    picks, tried = [], 0
    for f in random.sample(files, len(files)):
        tried += 1
        if tried > 60 or len(picks) >= N_CLIPS:
            break
        pid, vid, sf = parse(f)
        try:
            o = torch.load(f, map_location="cpu")
            tok, idxt = o["tok"], o["idx"].long()
            frames = decode_padded(pid, vid, sf)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {e}", flush=True); continue
        pv, pn, pa = predict(tok, idxt)
        picks.append((vid, sf, idxt.numpy().astype(np.int64), frames, (pv, pn, pa)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def vname(i): return verb_names.get(i, str(i))
    def nname(i): return noun_names.get(i, str(i))

    for vid, sf, idx, frames, (pv, pn, pa) in picks:
        row = gt.get((vid, sf), {})
        gv, gn = int(row.get("verb_class", -1)), int(row.get("noun_class", -1))
        narr = row.get("narration", "?")
        pav, pan = inv_action.get(pa, (-1, -1))
        vmark = "✓" if pv == gv else "✗"
        nmark = "✓" if pn == gn else "✗"
        amark = "✓" if (pav, pan) == (gv, gn) else "✗"

        slots, within = idx // GP, idx % GP
        keep = np.zeros((SLOTS, GRID, GRID), dtype=bool)
        keep[slots, within // GRID, within % GRID] = True
        show = np.linspace(0, 479, N_FRAMES_SHOW).round().astype(int)
        fig, axes = plt.subplots(2, N_FRAMES_SHOW, figsize=(2.7 * N_FRAMES_SHOW, 6.0))
        for col, p in enumerate(show):
            s = p // 2
            img = frames[p].astype(np.float32) / 255.0
            mask = np.kron(keep[s], np.ones((PATCH, PATCH)))[..., None]
            axes[0, col].imshow(img); axes[0, col].axis("off")
            axes[0, col].set_title(f"t={p/8:.1f}s", fontsize=9)
            axes[1, col].imshow(img * (0.2 + 0.8 * mask)); axes[1, col].axis("off")
            axes[1, col].set_title(f"kept {int(keep[s].sum())}/256", fontsize=9)
        axes[0, 0].set_ylabel("raw", fontsize=10); axes[1, 0].set_ylabel("pruned", fontsize=10)
        title = (f"{vid}  —  GT: {vname(gv)} / {nname(gn)}  (“{narr}”)\n"
                 f"Pred: verb={vname(pv)} {vmark}   noun={nname(pn)} {nmark}   "
                 f"action={vname(pav)}/{nname(pan)} {amark}")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        outp = os.path.join(OUT, f"overlay_{vid}__{sf}.png")
        fig.savefig(outp, dpi=120); plt.close(fig)
        print(f"[overlay] {outp} | GT {vname(gv)}/{nname(gn)} -> Pred {vname(pv)}/{nname(pn)} "
              f"(v{vmark} n{nmark} a{amark})", flush=True)


if __name__ == "__main__":
    main()
