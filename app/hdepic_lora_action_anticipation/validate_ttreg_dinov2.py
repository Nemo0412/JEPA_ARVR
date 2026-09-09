#!/usr/bin/env python3
# [ATTN-CORNER-SINK] Validate the no-training "test-time registers" method
# (nickjiang2378/test-time-registers) on OUR pipeline: baseline dinov2_vitl14 vs
# dinov2_vitl14_tt_reg (same weights + inference hooks + 1 appended register), measured
# with our sink metrics (per-head received-attention peak on PATCH tokens) + token norms.
# Runs via Slurm only. Loads the repo from a local clone (source='local').
from __future__ import annotations
import argparse, glob, json, os
import numpy as np, torch
from PIL import Image

os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMG, PATCH = 224, 14
GRID = IMG // PATCH


def load_img(p, device):
    im = Image.open(p).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255.)
    return ((x.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD).to(device)


class AttnRecv:
    """Hook last block attn to capture per-head received attention (col-sum of softmax)."""
    def __init__(self, model):
        self.blk = model.blocks[-1]
        self.attn = self.blk.attn
        self._orig = self.attn.forward
        self.recv = None
        a = self.attn
        def fwd(x, attn_bias=None):
            B, N, C = x.shape
            qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0] * a.scale, qkv[1]              # scale already folded into q (dinov2 convention)
            with torch.no_grad():
                logits = q.float() @ k.float().transpose(-2, -1)   # (B,H,N,N)
                self.recv = logits.softmax(-1).sum(dim=2)[0].cpu().numpy()  # (H,N) sum over queries
            return self._orig(x, attn_bias)
        self.attn.forward = fwd
    def remove(self):
        self.attn.forward = self._orig


def analyze(model, files, device):
    nreg = int(getattr(model, "num_register_tokens", 0))
    cap = AttnRecv(model)
    peak_patch, reg_mass, cls_mass = [], [], []
    max_patch_norm, reg_norm = [], []
    for f in files:
        x = load_img(f, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_features(x)
        recv = cap.recv                                  # (H, N), layout [CLS, patch*256, reg*nreg]
        H, N = recv.shape
        p0, p1 = 1, 1 + GRID * GRID
        patch = recv[:, p0:p1]                           # (H, 256)
        denom = recv.sum(axis=1, keepdims=True) + 1e-12
        uni = 1.0 / (GRID * GRID)
        pk = ((patch / denom) / uni).max(axis=1)         # per-head peak patch token / uniform
        peak_patch.append(pk.max())
        cls_mass.append(float((recv[:, 0] / denom.squeeze(1)).mean()))
        if nreg > 0:
            reg_mass.append(float((recv[:, -nreg:].sum(axis=1) / denom.squeeze(1)).mean()))
        # token norms (Darcet metric)
        pn = out["x_norm_patchtokens"][0].float()        # (256, D)
        max_patch_norm.append(float(pn.norm(dim=-1).max()))
        if nreg > 0 and "x_norm_regtokens" in out:
            rn = out["x_norm_regtokens"][0].float()
            reg_norm.append(float(rn.norm(dim=-1).max()))
    cap.remove()
    return {
        "n_register_tokens": nreg,
        "patch_sink_peak_over_uniform": round(float(np.mean(peak_patch)), 2),
        "cls_received_mass": round(float(np.mean(cls_mass)), 4),
        "register_received_mass": round(float(np.mean(reg_mass)), 4) if reg_mass else None,
        "max_patch_token_norm": round(float(np.mean(max_patch_norm)), 2),
        "register_token_norm": round(float(np.mean(reg_norm)), 2) if reg_norm else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    device = "cuda"
    files = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")) + glob.glob(os.path.join(args.img_dir, "*.png")))
    print(f"[data] {len(files)} images", flush=True)
    report = {}
    for name in ["dinov2_vitl14", "dinov2_vitl14_tt_reg"]:
        print(f"[load] {name}", flush=True)
        model = torch.hub.load(args.hub_dir, name, source="local", trust_repo=True).eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        report[name] = analyze(model, files, device)
        print(name, json.dumps(report[name]), flush=True)
        del model; torch.cuda.empty_cache()
    b, t = report["dinov2_vitl14"], report["dinov2_vitl14_tt_reg"]
    report["_summary"] = {
        "patch_sink_peak": f'{b["patch_sink_peak_over_uniform"]}x -> {t["patch_sink_peak_over_uniform"]}x',
        "max_patch_norm": f'{b["max_patch_token_norm"]} -> {t["max_patch_token_norm"]}',
        "register_absorbs_mass": t["register_received_mass"],
        "register_norm": t["register_token_norm"],
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
