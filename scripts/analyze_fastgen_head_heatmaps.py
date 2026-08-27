#!/usr/bin/env python3
"""Per-head attention key-importance heatmaps for V-JEPA (FastGen-style profiling).

For every encoder / predictor RoPE attention head, compute query-averaged
column-sum importance over keys (fine-grained token map), reshape video tokens
to [T_slot, H_patch, W_patch], and save:

  {out}/{model}/sample_{i}_{vid}/encoder/L{layer:02d}_h{head:02d}.png
  {out}/{model}/sample_{i}_{vid}/predictor/L{layer:02d}_h{head:02d}.png
  ... matching .npy vectors [N] and optional layer mosaics

Also writes ``{out}/{model}/mean/`` averaged over samples.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

# Reuse model/data helpers from FastGen profiler.
_spec = importlib.util.spec_from_file_location(
    "analyze_fastgen_heads",
    PROJECT_ROOT / "scripts" / "analyze_fastgen_heads.py",
)
_fastgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fastgen)

logger = logging.getLogger("fastgen_heatmaps")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_head_key_importance(q: torch.Tensor, k: torch.Tensor, scale: float, chunk: int = 128) -> np.ndarray:
    """Column-sum importance per head: imp[h,j] = sum_q softmax(q·k)[h,q,j]. Shape [H,N]."""
    _b, h, n, _d = q.shape
    imp = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        q_c = q[:, :, ci : ci + chunk, :].float()
        logits = torch.matmul(q_c, k_t) * scale
        imp += logits.softmax(dim=-1).sum(dim=2).mean(dim=0)
    arr = imp.cpu().numpy()
    # Per-head max-normalize so each head uses the full colormap (layer-global
    # norm makes weak heads look all-black in viridis).
    mx = arr.max(axis=1, keepdims=True)
    mx = np.maximum(mx, 1e-12)
    arr = arr / mx
    return arr.astype(np.float32)


def _reshape_video(imp: np.ndarray, n_aux: int, grid: int) -> tuple[np.ndarray, int]:
    n_vid = int(imp.size - n_aux)
    gp = grid * grid
    if n_vid <= 0 or n_vid % gp != 0:
        return np.zeros((0, grid, grid), dtype=np.float32), 0
    t_slots = n_vid // gp
    vid = imp[n_aux:].reshape(t_slots, grid, grid)
    return vid, t_slots


def save_head_figure(
    imp: np.ndarray,
    *,
    out_png: Path,
    title: str,
    n_aux: int,
    grid: int,
    policy: str | None = None,
) -> None:
    """Save one head: aux strip (if any) + temporal montage of 16×16 patches."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imp = imp.astype(np.float32, copy=False)
    mx = float(imp.max())
    if mx > 0:
        imp = imp / mx
    vid, t_slots = _reshape_video(imp, n_aux, grid)

    n_rows = 1 + (1 if n_aux > 0 else 0)
    fig_h = 2.2 * n_rows + max(0.5, t_slots * 0.15)
    fig, axes = plt.subplots(n_rows, max(t_slots, 1), figsize=(max(3.0, 2.2 * t_slots), fig_h), squeeze=False)
    if policy:
        title = f"{title}\npolicy={policy}"

    row = 0
    if n_aux > 0:
        ax0 = axes[row, 0]
        aux = imp[:n_aux]
        ax0.imshow(aux.reshape(1, -1), aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax0.set_ylabel("aux")
        ax0.set_xlabel(f"IMU/aux tokens (N={n_aux})")
        ax0.set_yticks([])
        for c in range(1, axes.shape[1]):
            axes[row, c].axis("off")
        row += 1

    if t_slots == 0:
        fig.suptitle(title, fontsize=9)
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return

    for t in range(t_slots):
        ax = axes[row, t] if axes.shape[1] > 1 else axes[row, 0]
        im = ax.imshow(vid[t], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"slot {t}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if t_slots == 1 and axes.shape[1] > 1:
        for c in range(1, axes.shape[1]):
            axes[row, c].axis("off")

    fig.suptitle(title, fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="per-head key importance")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_layer_mosaic(
    imp_heads: np.ndarray,
    *,
    out_png: Path,
    layer: int,
    module: str,
    n_aux: int,
    grid: int,
    policies: list[str] | None = None,
) -> None:
    """Grid of all heads in one layer (spatial avg over time for compact view)."""
    h, n = imp_heads.shape
    cols = 4 if h >= 12 else max(4, h)
    rows = int(np.ceil(h / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.atleast_2d(axes)
    for hi in range(rows * cols):
        r, c = divmod(hi, cols)
        ax = axes[r, c]
        if hi >= h:
            ax.axis("off")
            continue
        head_imp = imp_heads[hi].astype(np.float32, copy=False)
        mx = float(head_imp.max())
        if mx > 0:
            head_imp = head_imp / mx
        vid, _ = _reshape_video(head_imp, n_aux, grid)
        if vid.size == 0:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            ax.axis("off")
            continue
        spatial = vid.mean(axis=0)
        ax.imshow(spatial, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        pol = policies[hi] if policies and hi < len(policies) else ""
        short = pol.replace("recent+center+", "rc+").replace("recent+", "r+") if pol else ""
        ax.set_title(f"h{hi}" + (f"\n{short}" if short else ""), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{module} layer {layer} — spatial-mean key importance (all heads)", fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


class HeadCapture:
    """Install hooks; after forward collect {module, layer} -> [H,N] numpy."""

    def __init__(self, encoder, predictor, mask_fn_enc, mask_fn_pred, chunk: int):
        self.chunk = chunk
        self.mask_fn_enc = mask_fn_enc
        self.mask_fn_pred = mask_fn_pred
        self.encoder_maps: dict[tuple[str, int], np.ndarray] = {}
        self._orig = []

        pred_blocks = getattr(predictor, "predictor_blocks", None) or getattr(predictor, "blocks")
        for li, block in enumerate(encoder.blocks):
            self._patch(block.attn, "encoder", li)
        for li, block in enumerate(pred_blocks):
            self._patch(block.attn, "predictor", li)

    def _patch(self, attn_mod, module: str, layer: int):
        orig = attn_mod.forward

        def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            q, k, v = _fastgen._rope_qk(attn_mod, x, mask, T, H_patches, W_patches)
            imp = compute_head_key_importance(q, k, attn_mod.scale, chunk=self.chunk)
            self.encoder_maps[(module, layer)] = imp
            with torch.backends.cuda.sdp_kernel():
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=attn_mod.is_causal, attn_mask=attn_mask
                )
            y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2])
            y = attn_mod.proj(y)
            y = attn_mod.proj_drop(y)
            return y

        attn_mod.forward = wrapped
        self._orig.append((attn_mod, orig))

    def restore(self):
        for mod, orig in self._orig:
            mod.forward = orig


def run_heatmaps(tag, model, samples, device, args, aux_builders=None):
    if hasattr(model, "tri"):
        core = model.tri.base_model
        fusion_host = model.tri
    elif hasattr(model, "base_model") and hasattr(model.base_model, "encoder"):
        core = model.base_model
        fusion_host = model
    else:
        core = model
        fusion_host = model
    encoder = core.encoder
    predictor = core.predictor
    grid = int(getattr(core, "grid_size", 16))
    tubelet = int(getattr(core, "tubelet_size", 2))

    out_root = Path(args.out) / tag
    out_root.mkdir(parents=True, exist_ok=True)

    gaze_holder = {"mask": None}
    n_aux_holder = {"n": 0}

    def enc_mask_fn(n_tok, attn_mod):
        return _fastgen.build_key_masks(n_tok, grid, n_aux=0, gaze_flat=gaze_holder["mask"])

    def pred_mask_fn(n_tok, attn_mod):
        return _fastgen.build_key_masks(n_tok, grid, n_aux=n_aux_holder["n"], gaze_flat=None)

    # Load FastGen policies if available
    policies: dict[tuple[str, int, int], str] = {}
    summary_path = Path(args.fastgen_summary)
    if summary_path.is_file():
        rep = json.loads(summary_path.read_text())
        m = rep.get("models", {}).get(tag, {})
        for mod_key, bucket in (("encoder", "encoder"), ("predictor", "predictor")):
            for ly in m.get(mod_key, {}).get("layers", []):
                li = int(ly["layer"])
                for h, p in enumerate(ly.get("policy", [])):
                    policies[(bucket, li, h)] = str(p)

    mean_acc: dict[tuple[str, int, int], list[np.ndarray]] = {}
    n_ok = 0

    for si, (clip, meta) in enumerate(samples):
        vid = str(meta.get("video_id", f"sample{si}"))
        sample_dir = out_root / f"sample_{si:03d}_{vid}"
        gaze_holder["mask"] = None
        n_aux_holder["n"] = 0

        aux_map = None
        imu_batch = None
        clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
        ant = torch.full((1,), float(args.anticipation), device=device)
        if aux_builders is not None:
            builder, imu_loader = aux_builders
            try:
                aux_map = builder.build(clip_b, [meta])
                gaze_ch = aux_map[:, 0]
                gaze_holder["mask"] = _fastgen.gaze_token_mask(gaze_ch[0], tubelet, grid)
                imu_batch = imu_loader.load_batch([meta], device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("aux failed %s: %s", vid, exc)

        capture = HeadCapture(encoder, predictor, enc_mask_fn, pred_mask_fn, args.chunk)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if aux_builders is not None:
                out = model(clip_b, ant, binary_map=aux_map, imu_batch=imu_batch)
            else:
                out = model(clip_b, ant)
        capture.restore()
        n_aux_holder["n"] = int(getattr(fusion_host, "_n_aux_context_tokens", 0) or 0)
        if out is None:
            logger.warning("skip non-finite sample %s", vid)
            continue
        n_ok += 1

        # Note: capture.encoder_maps is overwritten each layer during forward; we need per-layer storage.
        # HeadCapture stores last layer only per forward call — BUG! Each hook overwrites same dict key
        # but different keys (module, layer) — actually each layer has unique key. Good.

        # Re-run with fixed capture that stores ALL layers in one forward — hooks fire once per layer
        # capture.encoder_maps should have all layers after one forward. Verify:
        enc_layers = max((k[1] for k in capture.encoder_maps if k[0] == "encoder"), default=-1) + 1
        pred_layers = max((k[1] for k in capture.encoder_maps if k[0] == "predictor"), default=-1) + 1
        logger.info("%s sample %d/%s maps enc=%d pred=%d n_aux=%d", tag, si + 1, vid, enc_layers, pred_layers, n_aux_holder["n"])

        for (module, layer), imp_hn in sorted(capture.encoder_maps.items()):
            n_aux = 0 if module == "encoder" else n_aux_holder["n"]
            sub = sample_dir / module
            mosaic_pol = []
            for h in range(imp_hn.shape[0]):
                pol = policies.get((module, layer, h), "")
                mosaic_pol.append(pol)
                png = sub / f"L{layer:02d}_h{h:02d}.png"
                npy = sub / f"L{layer:02d}_h{h:02d}.npy"
                save_head_figure(
                    imp_hn[h],
                    out_png=png,
                    title=f"{tag} {module} L{layer} head {h} | {vid}",
                    n_aux=n_aux,
                    grid=grid,
                    policy=pol or None,
                )
                np.save(npy, imp_hn[h])
                mean_acc.setdefault((module, layer, h), []).append(imp_hn[h].copy())

            save_layer_mosaic(
                imp_hn,
                out_png=sub / f"L{layer:02d}_mosaic.png",
                layer=layer,
                module=module,
                n_aux=n_aux,
                grid=grid,
                policies=mosaic_pol,
            )

    # Mean over samples
    if n_ok > 1:
        mean_dir = out_root / "mean"
        by_layer: dict[tuple[str, int], list[np.ndarray]] = {}
        for (module, layer, h), arrs in mean_acc.items():
            by_layer.setdefault((module, layer), [])
        for key, arrs in mean_acc.items():
            module, layer, h = key
            m = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)
            n_aux = 0 if module == "encoder" else int(getattr(fusion_host, "_n_aux_context_tokens", 0) or 0)
            sub = mean_dir / module
            pol = policies.get((module, layer, h), "")
            save_head_figure(
                m,
                out_png=sub / f"L{layer:02d}_h{h:02d}.png",
                title=f"{tag} MEAN({n_ok}) {module} L{layer} h{h}",
                n_aux=n_aux,
                grid=grid,
                policy=pol or None,
            )
            np.save(sub / f"L{layer:02d}_h{h:02d}.npy", m)

        for module, layer in sorted(set((k[0], k[1]) for k in mean_acc)):
            heads = sorted(h for mod, ly, h in mean_acc if mod == module and ly == layer)
            if not heads:
                continue
            stack = np.stack([np.mean(mean_acc[(module, layer, h)], axis=0) for h in heads], axis=0)
            n_aux = 0 if module == "encoder" else int(getattr(fusion_host, "_n_aux_context_tokens", 0) or 0)
            pols = [policies.get((module, layer, h), "") for h in heads]
            save_layer_mosaic(
                stack,
                out_png=mean_dir / module / f"L{layer:02d}_mosaic.png",
                layer=layer,
                module=f"{module} mean",
                n_aux=n_aux,
                grid=grid,
                policies=pols,
            )

    meta_out = {
        "tag": tag,
        "n_samples_ok": n_ok,
        "grid": grid,
        "out": str(out_root),
        "layout": "flat idx = aux[0:n_aux] + video[slot*256 + h*16 + w]",
    }
    (out_root / "meta.json").write_text(json.dumps(meta_out, indent=2))
    logger.info("wrote heatmaps under %s (%d samples)", out_root, n_ok)
    return meta_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=4)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--anticipation", type=float, default=1.0)
    p.add_argument("--val-csv", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv")
    p.add_argument("--video-root", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos")
    p.add_argument("--vjepa-ckpt", default="/scratch/ll5914/models/vjepa2/vitl.pt")
    p.add_argument(
        "--video-enc-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt",
    )
    p.add_argument(
        "--video-pred-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt",
    )
    ca = "/scratch/ll5914/experiments/concat_plus_cross_attn_v2/action_anticipation_frozen/concat-plus-ca-v2-L3-keepaux-softgate-vitl16-256-12ep-1xh100"
    p.add_argument("--ca-enc-lora", default=f"{ca}/encoder_lora_best.pt")
    p.add_argument("--ca-pred-lora", default=f"{ca}/predictor_lora_best.pt")
    p.add_argument("--ca-adapter", default=f"{ca}/binary_input_adapter_best.pt")
    p.add_argument("--ca-fusion", default=f"{ca}/tri_modal_fusion_best.pt")
    p.add_argument("--gaze-root", default="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze")
    p.add_argument("--gaze-extract", default="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract")
    p.add_argument("--gaze-sync", default="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos")
    p.add_argument("--pose-slam", default="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi")
    p.add_argument("--out", default="/scratch/ll5914/experiments/fastgen_head_heatmaps")
    p.add_argument(
        "--fastgen-summary",
        default="/scratch/ll5914/experiments/fastgen_head_profile/summary.json",
    )
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-ca", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("TRI_MODAL_FRAME_CACHE", "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = _fastgen.load_val_clips(args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    report = {"out": args.out, "models": {}}

    if not args.skip_video:
        logger.info("=== video-only heatmaps ===")
        video = _fastgen.load_video_model(args, device)
        report["models"]["video_only"] = run_heatmaps("video_only", video, samples, device, args)
        del video
        torch.cuda.empty_cache()

    if not args.skip_ca:
        logger.info("=== gaze+IMU heatmaps ===")
        ca_model = _fastgen.load_concat_ca_model(args, device)
        cfg = _fastgen.gaze_cfg(args)
        gate = _fastgen.GazeTokenGate({**cfg, "mode": "token_gate"})
        builder = _fastgen.GazePoseInputMapBuilder(cfg, gate=gate)
        imu_loader = _fastgen.ImuTrajectoryLoader(cfg, gate=gate)
        report["models"]["gaze_imu"] = run_heatmaps(
            "gaze_imu", ca_model, samples, device, args, aux_builders=(builder, imu_loader)
        )
        del ca_model
        torch.cuda.empty_cache()

    (Path(args.out) / "index.json").write_text(json.dumps(report, indent=2))
    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
