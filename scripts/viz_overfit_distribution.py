"""Visualize B11 singleprobe overfitting vs HD-EPIC label distribution.

Reads the per-sample val prediction dump (PredictionDumper CSV) for a trained
checkpoint plus the train/val annotation CSVs, and produces:

  1. per-class accuracy vs *train* frequency scatter (verb / noun / action)
  2. correct-vs-wrong sample counts split by class frequency tier (head/torso/tail)
  3. "error collapse": for wrong samples, train-frequency of the predicted class
     vs the (true) gt class -- tests whether errors collapse toward frequent classes
  4. a handful of rendered example frames (gt vs pred, with class names)

All outputs land under --out-dir. No GPU needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- loading helpers -----------------------------

def _as_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def _load_json_list(value):
    if not value:
        return []
    try:
        out = json.loads(value)
    except json.JSONDecodeError:
        return []
    return out if isinstance(out, list) else []


def load_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not path or not path.exists():
        return names
    for row in csv.DictReader(path.open("r", encoding="utf-8")):
        try:
            names[int(row["id"])] = str(row.get("key", row.get("id")))
        except Exception:
            continue
    return names


def train_freq(path: Path, col: str) -> Counter:
    c: Counter = Counter()
    if not path.exists():
        return c
    for row in csv.DictReader(path.open("r", encoding="utf-8")):
        c[_as_int(row[col])] += 1
    return c


# ----------------------------- core analysis -----------------------------

TASKS = ("verb", "noun", "action")


def per_class_stats(rows, task):
    """Return dict[label] -> dict(total, top1, top3, top5)."""
    stats = defaultdict(lambda: Counter(total=0, top1=0, top3=0, top5=0))
    for row in rows:
        label = _as_int(row.get(f"{task}_label"))
        if label < 0:
            continue
        preds = _load_json_list(row.get(f"{task}_top10") or row.get(f"{task}_top5"))
        pred1 = _as_int(row.get(f"{task}_top1"), preds[0] if preds else -1)
        s = stats[label]
        s["total"] += 1
        s["top1"] += int(pred1 == label)
        s["top3"] += _as_int(row.get(f"{task}_top3_hit"), int(label in preds[:3]))
        s["top5"] += _as_int(row.get(f"{task}_top5_hit"), int(label in preds[:5]))
    return stats


def tier_of(count, q_lo, q_hi):
    if count >= q_hi:
        return "head"
    if count >= q_lo:
        return "torso"
    return "tail"


def plot_acc_vs_freq(stats, tfreq, names, task, out_dir):
    xs, y1, y3, sizes, labels = [], [], [], [], []
    for label, s in stats.items():
        if s["total"] == 0:
            continue
        xs.append(max(tfreq.get(label, 0), 0.5))  # avoid log(0)
        y1.append(100.0 * s["top1"] / s["total"])
        y3.append(100.0 * s["top3"] / s["total"])
        sizes.append(s["total"])
        labels.append(label)
    xs = np.array(xs); y1 = np.array(y1); y3 = np.array(y3); sizes = np.array(sizes)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(xs, y1, s=12 + 4 * sizes, c=y1, cmap="RdYlGn", vmin=0, vmax=100,
                    edgecolors="k", linewidths=0.4, alpha=0.85)
    # weighted trend (val-sample weighted) per log-freq bin
    if len(xs) > 3:
        order = np.argsort(xs)
        lx = np.log10(xs[order])
        bins = np.linspace(lx.min(), lx.max(), 7)
        bx, by = [], []
        for i in range(len(bins) - 1):
            m = (lx >= bins[i]) & (lx <= bins[i + 1])
            if m.sum() == 0:
                continue
            w = sizes[order][m]
            bx.append(10 ** ((bins[i] + bins[i + 1]) / 2))
            by.append(np.average(y1[order][m], weights=w))
        ax.plot(bx, by, "b--o", lw=1.6, ms=4, label="val-weighted Top-1 trend")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("train frequency of class (# train samples, log)")
    ax.set_ylabel("per-class Top-1 accuracy (%)")
    ax.set_title(f"{task}: per-class accuracy vs train frequency\n"
                 f"(point size ~ #val samples; color = Top-1)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, label="Top-1 acc (%)")
    fig.tight_layout()
    p = out_dir / f"acc_vs_freq_{task}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_correct_wrong_tiers(rows, tfreq, task, out_dir):
    counts = train_freq_quantiles(tfreq)
    q_lo, q_hi = counts
    tiers = ["head", "torso", "tail"]
    correct = Counter(); wrong = Counter(); tot = Counter()
    for row in rows:
        label = _as_int(row.get(f"{task}_label"))
        if label < 0:
            continue
        t = tier_of(tfreq.get(label, 0), q_lo, q_hi)
        hit = _as_int(row.get(f"{task}_top1"), -2) == label
        tot[t] += 1
        if hit:
            correct[t] += 1
        else:
            wrong[t] += 1
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(tiers))
    c = [correct[t] for t in tiers]; w = [wrong[t] for t in tiers]
    ax.bar(x, c, label="Top-1 correct", color="#2e7d32")
    ax.bar(x, w, bottom=c, label="Top-1 wrong", color="#c62828", alpha=0.85)
    for i, t in enumerate(tiers):
        acc = 100.0 * correct[t] / max(1, tot[t])
        ax.text(i, c[i] + w[i] + 2, f"{acc:.0f}% acc\nn={tot[t]}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(>= {q_hi} / >= {q_lo} / < {q_lo})".split("\n")[0] for t in tiers])
    ax.set_xlabel(f"{task} class frequency tier (by train count; head>={q_hi}, torso>={q_lo}, tail<{q_lo})")
    ax.set_ylabel("# val samples")
    ax.set_title(f"{task}: correct vs wrong val samples by class-frequency tier")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"correct_wrong_tiers_{task}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, {t: dict(total=tot[t], correct=correct[t], wrong=wrong[t]) for t in tiers}


def plot_error_collapse(rows, tfreq, task, out_dir):
    """For wrong samples: train-freq of predicted class vs gt class."""
    gt_f, pred_f = [], []
    for row in rows:
        label = _as_int(row.get(f"{task}_label"))
        pred1 = _as_int(row.get(f"{task}_top1"))
        if label < 0 or pred1 < 0 or pred1 == label:
            continue
        gt_f.append(max(tfreq.get(label, 0), 0.5))
        pred_f.append(max(tfreq.get(pred1, 0), 0.5))
    if not gt_f:
        return None
    gt_f = np.log10(np.array(gt_f)); pred_f = np.log10(np.array(pred_f))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bins = np.linspace(min(gt_f.min(), pred_f.min()), max(gt_f.max(), pred_f.max()), 18)
    ax.hist(gt_f, bins=bins, alpha=0.6, label="true class (of wrong samples)", color="#1565c0")
    ax.hist(pred_f, bins=bins, alpha=0.6, label="predicted class (Top-1)", color="#c62828")
    ax.axvline(np.median(gt_f), color="#1565c0", ls="--", lw=1)
    ax.axvline(np.median(pred_f), color="#c62828", ls="--", lw=1)
    ax.set_xlabel("log10(train frequency of class)")
    ax.set_ylabel("# wrong val samples")
    ax.set_title(f"{task}: on errors, predictions collapse toward frequent classes?\n"
                 f"median true={10**np.median(gt_f):.0f}  median pred={10**np.median(pred_f):.0f} train samples")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"error_collapse_{task}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def train_freq_quantiles(tfreq):
    vals = np.array([v for v in tfreq.values() if v > 0])
    if len(vals) == 0:
        return (1, 10)
    q_lo = int(np.quantile(vals, 0.50))
    q_hi = int(np.quantile(vals, 0.80))
    return (max(q_lo, 1), max(q_hi, q_lo + 1))


# ----------------------------- example frames -----------------------------

def render_examples(rows, tfreq_verb, tfreq_noun, vnames, nnames, video_root, out_dir, n_each=3):
    try:
        import decord
    except Exception as e:  # pragma: no cover
        (out_dir / "examples_SKIPPED.txt").write_text(f"decord unavailable: {e}\n")
        return None

    q_lo_v, q_hi_v = train_freq_quantiles(tfreq_verb)

    def vtier(lbl):
        return tier_of(tfreq_verb.get(lbl, 0), q_lo_v, q_hi_v)

    cats = {
        "head_correct": [],   # frequent verb, predicted right
        "tail_correct": [],   # rare verb, predicted right (rare win)
        "tail_wrong_collapse": [],  # rare verb, wrong, collapsed to a more frequent verb
        "head_wrong": [],     # frequent verb, wrong
    }
    for row in rows:
        vl = _as_int(row.get("verb_label")); vp = _as_int(row.get("verb_top1"))
        nl = _as_int(row.get("noun_label")); np_ = _as_int(row.get("noun_top1"))
        if vl < 0 or vp < 0:
            continue
        t = vtier(vl)
        correct = vp == vl
        if t == "head" and correct:
            cats["head_correct"].append(row)
        elif t == "tail" and correct:
            cats["tail_correct"].append(row)
        elif t == "tail" and not correct and tfreq_verb.get(vp, 0) > tfreq_verb.get(vl, 0):
            cats["tail_wrong_collapse"].append(row)
        elif t == "head" and not correct:
            cats["head_wrong"].append(row)

    picks = []
    for cat, lst in cats.items():
        for row in lst[:n_each]:
            picks.append((cat, row))

    manifest = []
    n = len(picks)
    if n == 0:
        return None
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (cat, row) in enumerate(picks):
        vid = str(row.get("video_id", ""))
        part = vid.split("_")[0]
        mp4 = Path(video_root) / part / f"{vid}.MP4"
        sf = _as_int(row.get("start_frame"), 0); ef = _as_int(row.get("stop_frame"), sf + 1)
        vl = _as_int(row.get("verb_label")); vp = _as_int(row.get("verb_top1"))
        nl = _as_int(row.get("noun_label")); np_ = _as_int(row.get("noun_top1"))
        frame = None
        try:
            vr = decord.VideoReader(str(mp4))
            mid = min(max((sf + ef) // 2, 0), len(vr) - 1)
            frame = vr[mid].asnumpy()
            del vr
        except Exception as e:
            frame = None
            err = str(e)
        ax = axes[i]
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, f"[no frame]\n{mp4.name}", ha="center", va="center", fontsize=7)
        gt = f"GT verb={vnames.get(vl, vl)} noun={nnames.get(nl, nl)}"
        pr = f"Pred verb={vnames.get(vp, vp)} noun={nnames.get(np_, np_)}"
        ok = "OK" if vp == vl else "X"
        ax.set_title(f"[{cat}] {ok}\n{gt}\n{pr}", fontsize=8,
                     color=("#2e7d32" if vp == vl else "#c62828"))
        ax.axis("off")
        manifest.append({
            "category": cat, "video_id": vid, "start_frame": sf, "stop_frame": ef,
            "gt_verb": vnames.get(vl, vl), "gt_noun": nnames.get(nl, nl),
            "pred_verb": vnames.get(vp, vp), "pred_noun": nnames.get(np_, np_),
            "verb_correct": vp == vl, "narration": row.get("narration", ""),
        })
    fig.suptitle("B11 singleprobe val examples (gt vs pred, by verb-frequency category)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "examples_contact_sheet.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    (out_dir / "examples_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--verb-names", type=Path, required=True)
    ap.add_argument("--noun-names", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.pred_csv.open("r", encoding="utf-8")))
    vnames = load_names(args.verb_names)
    nnames = load_names(args.noun_names)
    tfreq = {
        "verb": train_freq(args.train_csv, "verb_class"),
        "noun": train_freq(args.train_csv, "noun_class"),
    }
    # action freq: from train (verb,noun) pairs mapped via dump class_maps if present
    tfreq["action"] = Counter()
    for row in rows:
        tfreq["action"][_as_int(row.get("action_label"))] += 0  # ensure keys exist
    # approximate action train freq by val co-count is not ideal; use val sample count fallback
    report = {"pred_csv": str(args.pred_csv), "num_samples": len(rows), "tasks": {}}
    outputs = []
    for task in ("verb", "noun"):
        stats = per_class_stats(rows, task)
        outputs.append(str(plot_acc_vs_freq(stats, tfreq[task], vnames if task == "verb" else nnames, task, args.out_dir)))
        p, tier_summary = plot_correct_wrong_tiers(rows, tfreq[task], task, args.out_dir)
        outputs.append(str(p))
        ec = plot_error_collapse(rows, tfreq[task], task, args.out_dir)
        if ec:
            outputs.append(str(ec))
        # overall + correlation
        accs, freqs, wts = [], [], []
        for lbl, s in stats.items():
            if s["total"] == 0:
                continue
            accs.append(s["top1"] / s["total"]); freqs.append(tfreq[task].get(lbl, 0)); wts.append(s["total"])
        corr = float(np.corrcoef(np.log10(np.array(freqs) + 1), accs)[0, 1]) if len(accs) > 2 else float("nan")
        report["tasks"][task] = {
            "tiers": tier_summary,
            "spearman_proxy_logfreq_vs_top1acc": corr,
            "n_classes_seen": len(stats),
        }

    ex = render_examples(rows, tfreq["verb"], tfreq["noun"], vnames, nnames,
                         args.video_root, args.out_dir, n_each=args.n_examples)
    if ex:
        outputs.append(str(ex))

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for o in outputs:
        print(o)


if __name__ == "__main__":
    main()
