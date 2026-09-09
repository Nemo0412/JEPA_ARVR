#!/usr/bin/env python3
"""Render all64 slots from immutable RGB and pooled masks; CPU only, no model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROTOCOL = "b18-predictor-prune/egtea-q3-single-frame-source-v1"
CONDITIONS = ("continuous", "start_hdepic_20260907", "middle_hdepic_20260907")
EDITED = {"continuous": (), "start_hdepic_20260907": (0, 1), "middle_hdepic_20260907": (32, 33)}
EXPECTED = {"continuous": ([144, 139], [84, 46]),
            "start_hdepic_20260907": ([129, 236], [71, 41]),
            "middle_hdepic_20260907": ([130, 137], [255, 249])}
LEGACY_ANALYSIS_HASH = "89c0052b98ad051b906b4c279d6defcb5a46814c62e0d603448c443059f68850"
EDIT_COLOR = "#b2185b"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bytehash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-analysis", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    summary_path = args.source_analysis / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert manifest["protocol"] == summary["protocol"] == PROTOCOL
    assert digest(args.manifest) == summary["manifest_sha256"]
    assert summary["n_rows"] == 86 and not summary["partial"]
    old_script = Path(__file__).with_name("analyze_b18_q3_single_frame_source.py")
    assert digest(old_script) == summary["analysis_sha256"] == LEGACY_ANALYSIS_HASH
    for path, expected in manifest["code_sha256"].items():
        assert digest(path) == expected, path
    dependency = summary["metric_dependency"]
    assert digest(dependency["path"]) == dependency["sha256"]
    # Capture every existing analysis artifact before rendering, including the
    # original five-slot figures. No write operation targets these paths.
    preserved_paths = sorted(p for p in args.source_analysis.rglob("*") if p.is_file())
    preserved_paths += [args.manifest, old_script, Path(dependency["path"])]
    old_hashes = {str(path): digest(path) for path in preserved_paths}
    source_npz = args.source_analysis / "pooled_traces_scores_masks.npz"
    masks, mask_audits = {}, {}
    with np.load(source_npz) as source:
        for condition in CONDITIONS:
            score = source["score_actual__" + condition]
            mask = source["mask__" + condition]
            assert score.shape == mask.shape == (64, 256)
            assert mask.dtype == bool and np.isfinite(score).all() and int(mask.sum()) == 4096
            replay = np.zeros(64 * 256, dtype=bool)
            replay[np.argsort(-score.astype(np.float64).ravel(), kind="stable")[:4096]] = True
            assert np.array_equal(replay.reshape(64, 256), mask)
            count = mask.sum(1)
            assert count.tolist() == summary["conditions"][condition]["actual"]["count"]
            assert count[:2].tolist() == EXPECTED[condition][0]
            assert count[32:34].tolist() == EXPECTED[condition][1]
            masks[condition] = mask.copy()
            mask_audits[condition] = dict(slots=64, kept=int(count.sum()), counts=count.tolist(),
                mask_sha256=bytehash(mask), original_mask_exact=True, stable_topK_replay_exact=True,
                edited_slots=list(EDITED[condition]), oldest=count[:2].tolist(), middle=count[32:34].tolist())
    np.savez_compressed(args.out / "rendered_masks.npz", **masks)
    inputs, panels, outputs, video_audits = [], [], [], {}
    with PdfPages(args.out / "full64_all_receivers_conditions.pdf") as pdf:
        for index in (0, 1):
            sample, row = manifest["source"]["samples"][index], manifest["rows"][index]
            sid = sample["sample_id"]
            assert sid == row["sample_id"] == f"q3_{index:03d}"
            cache = Path(sample["cache"]["path"])
            assert digest(cache) == sample["cache"]["sha256"]
            receiver = np.load(cache)
            assert receiver.shape == (256, 256, 256, 3) and receiver.dtype == np.uint8
            original = np.array(receiver[-128:], copy=True)
            archive = Path(row["image_archive"]["path"])
            assert digest(archive) == row["image_archive"]["sha256"]
            draw = row["draws"]["hdepic_20260907"]
            image_path = Path(draw["image_path"])
            image = np.load(image_path)
            assert image.shape == (256, 256, 3) and image.dtype == np.uint8
            with np.load(archive) as images:
                assert np.array_equal(image, images["hdepic_20260907"])
            assert bytehash(image) == draw["image_sha256"] == row["image_sha256"]["hdepic_20260907"]
            tile = np.repeat(image[None], 4, axis=0)
            assert bytehash(tile) == draw["tile_sha256"]
            source_video = next(v for v in manifest["hdepic_videos"] if v["video_id"] == draw["video_id"])
            if draw["video_id"] not in video_audits:
                native_path = Path(source_video["path"])
                assert native_path.stat().st_size == source_video["size_bytes"]
                assert native_path.stat().st_mtime_ns == source_video["mtime_ns"]
                assert digest(native_path) == source_video["sha256"]
                video_audits[draw["video_id"]] = dict(path=str(native_path), sha256=source_video["sha256"],
                    frame_count=source_video["frame_count"], native_frame=draw["native_frame"],
                    source_fps=draw["source_fps"], split="p01_fixed_train", file_hash_exact=True)
            immutable_rgb_paths = [cache, archive, image_path]
            before_rgb_hashes = {str(path): digest(path) for path in immutable_rgb_paths}
            for condition in CONDITIONS:
                clip = original.copy()
                edited_slots = EDITED[condition]
                edited_frames = []
                if edited_slots:
                    first = edited_slots[0] * 2
                    edited_frames = list(range(first, first + 4))
                    clip[first:first + 4] = tile
                    assert np.array_equal(clip[first:first + 4], tile)
                unchanged = np.ones(128, dtype=bool)
                unchanged[edited_frames] = False
                assert np.array_equal(clip[unchanged], original[unchanged])
                assert bytehash(clip) == row["input_sha256"][condition]
                mask = masks[condition]
                counts = mask.sum(1)
                record = dict(receiver=sid, condition=condition, original_window_sha256=bytehash(original),
                    reconstructed_input_sha256=bytehash(clip), archived_input_sha256=row["input_sha256"][condition],
                    source_image_sha256=draw["image_sha256"], source_tile_sha256=draw["tile_sha256"],
                    edited_slots=list(edited_slots), edited_rgb_indices=edited_frames,
                    edited_rgb_sha256=bytehash(clip[edited_frames]) if edited_frames else None,
                    unchanged_rgb_indices=np.flatnonzero(unchanged).tolist(),
                    unchanged_rgb_sha256=bytehash(clip[unchanged]),
                    original_unchanged_rgb_sha256=bytehash(original[unchanged]),
                    source_draw=draw, source_cache_hashes=before_rgb_hashes,
                    panel_slots=list(range(64)), panel_count=64, total_kept=int(counts.sum()),
                    mask_sha256=bytehash(mask))
                inputs.append(record)
                fig, axes = plt.subplots(8, 8, figsize=(22, 24))
                for slot, ax in enumerate(axes.ravel()):
                    rgb = clip[slot * 2]
                    kept = mask[slot].reshape(16, 16)
                    ax.imshow(rgb)
                    rgba = np.zeros((16, 16, 4)); rgba[..., 1] = 1; rgba[..., 3] = kept * .42
                    ax.imshow(rgba, extent=(-.5, 255.5, 255.5, -.5), interpolation="nearest")
                    changed = slot in edited_slots
                    ax.set(xticks=[], yticks=[])
                    ax.set_title(f"slot {slot:02d} | t={slot / 4:.2f}s\n{int(kept.sum())} / 256 kept" + (" | EDITED" if changed else ""),
                                 fontsize=10, color=EDIT_COLOR if changed else "#222222", weight="bold" if changed else "normal")
                    for spine in ax.spines.values():
                        spine.set_color(EDIT_COLOR if changed else "#d9d9d9")
                        spine.set_linewidth(3 if changed else .7)
                    original_cache_index = 128 + slot * 2
                    panels.append(dict(receiver=sid, condition=condition, slot=slot, window_time_sec=slot / 4,
                        window_rgb_index=slot * 2, kept=int(kept.sum()), edited=changed,
                        rendered_rgb_sha256=bytehash(rgb), rendered_mask_sha256=bytehash(kept),
                        source_video_id=draw["video_id"] if changed else sample["video_id"],
                        native_frame=draw["native_frame"] if changed else sample["frame_indices"][original_cache_index]))
                title = f"Pooled mask calibrated on 86 videos | K=4096 | {condition}\nReceiver {sid}: {sample['video_id']} | all 64 slots (0–63)"
                if condition == "continuous":
                    edit_note = "CONTINUOUS: no RGB edits; edited slots: none."
                elif condition.startswith("start_"):
                    edit_note = "START intervention: slots 0–1 / RGB 0–3 (0.00–0.50s); ONE HD RGB repeated 4 times. All other RGB unchanged."
                else:
                    edit_note = "MIDDLE intervention: slots 32–33 / RGB 64–67 (8.00–8.50s); ONE HD RGB repeated 4 times. All other RGB unchanged."
                fig.suptitle(title, fontsize=17, y=.992)
                fig.text(.5, .953, edit_note, ha="center", fontsize=12, color=EDIT_COLOR if edited_slots else "#222222")
                fig.text(.5, .937, "t=0 at the 16s window start; each panel shows the FIRST RGB of its two-frame slot (8 FPS).\nGreen = kept pooled mask (opacity 0.42), identical for both receivers; magenta border = edited RGB. No slots omitted.",
                         ha="center", va="top", fontsize=11)
                fig.subplots_adjust(left=.025, right=.985, bottom=.015, top=.906, wspace=.10, hspace=.28)
                name = f"full64_{sid}_{condition}.png"
                fig.savefig(args.out / name, dpi=160)
                pdf.savefig(fig, dpi=160)
                plt.close(fig)
                outputs.append(name)
                print(f"RENDERED {name} panels=64 kept=4096", flush=True)
            assert all(digest(path) == value for path, value in before_rgb_hashes.items())
    fig, ax = plt.subplots(figsize=(14, 5.3))
    colors = ("#333333", "#2377b4", "#b2185b")
    labels = ("continuous: no RGB edits", "start HD07: edited slots 0–1", "middle HD07: edited slots 32–33")
    for condition, color, label in zip(CONDITIONS, colors, labels):
        ax.plot(np.arange(64), masks[condition].sum(1), "o-", color=color, label=label, markersize=3, lw=1.7)
    ax.axvspan(-.5, 1.5, color=colors[1], alpha=.08)
    ax.axvspan(31.5, 33.5, color=colors[2], alpha=.08)
    ax.axhline(64, color="gray", linestyle=":", lw=1, label="uniform allocation: 64 tokens/slot")
    ax.annotate("middle HD07 slots 32/33: 255/249", xy=(32.5, 252), xytext=(36, 225),
                fontsize=10, color=colors[2], arrowprops=dict(arrowstyle="->", color=colors[2]))
    ax.set(xlim=(-.5, 63.5), ylim=(0, 268), xticks=np.r_[np.arange(0, 64, 4), 63],
           xlabel="Temporal slot (0 = oldest; 4 slots per second)", ylabel="Kept tokens / 256")
    ax.grid(alpha=.2); ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("All64 counts | pooled on86 videos | K4096 | same mask for both illustrated receivers", fontsize=13)
    fig.text(.5, .015, "Start HD07 edits only slots 0–1; middle HD07 edits only slots 32–33. Each uses one HD image repeated across four RGB frames.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(args.out / "full64_count_profiles.png", dpi=170); plt.close(fig)
    outputs += ["full64_count_profiles.png", "full64_all_receivers_conditions.pdf", "rendered_masks.npz"]
    with (args.out / "panel_manifest.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(panels[0])); writer.writeheader(); writer.writerows(panels)
    assert len(inputs) == 6 and len(panels) == 384
    assert all(r["panel_slots"] == list(range(64)) and r["total_kept"] == 4096 for r in inputs)
    assert all(digest(path) == value for path, value in old_hashes.items()), "old artifact changed"
    audit = dict(protocol=PROTOCOL, run_tag="b18-q3-single-frame-full64-viz", job_id=os.environ.get("SLURM_JOB_ID"),
        metric_scope="visualization-only-existing-selection", source_manifest=str(args.manifest),
        source_analysis=str(args.source_analysis), renderer_sha256=digest(__file__),
        receivers=[r["sample_id"] for r in manifest["source"]["samples"][:2]], conditions=CONDITIONS,
        calibration_n=86, K=4096, figure_count=6, panel_count=384, mask_audits=mask_audits,
        inputs=inputs, native_source_audits=video_audits, legacy_hashes_before_and_after=old_hashes,
        gates=dict(complete_64_slots_each=True, K4096_each=True, exact_original_masks=True,
                   score_to_mask_replay_exact=True, original_RGB_cache_hashes_exact=True,
                   input_edit_and_unchanged_hashes_exact=True, native_source_file_hashes_exact=True,
                   old_artifacts_unchanged=True),
        time_definition="slot t shows local RGB index2*t at t/4 seconds after the16s window begins; second RGB remains part of the token slot but is not displayed",
        intervention_definition="start and middle are distinct input conditions; one selected HD RGB repeated4times in slots0/1 or32/33, respectively",
        seconds=time.time() - started,
        outputs={name: dict(path=str(args.out / name), sha256=digest(args.out / name), bytes=(args.out / name).stat().st_size) for name in outputs})
    write_json(args.out / "audit.json", audit)
    print("COMPLETE " + json.dumps(dict(out=str(args.out), seconds=audit["seconds"], gates=audit["gates"])), flush=True)


if __name__ == "__main__":
    main()
