"""Table: released windows (R0, R5), honest 1s (A1) and our 4s/+2s window (W), same row subset."""
import json
from pathlib import Path
import numpy as np, pandas as pd
RW = Path("/scratch/yh6416/VJEPA2-EXP/outputs/inavit_official_ek100/released_window")
ROOT = Path("/scratch/yh6416/VJEPA2-EXP")
val = pd.read_csv(ROOT / "outputs/collab_inavit_ek100/prep/csv/EK100_val_stream_mtp.csv")
acts = pd.read_csv(ROOT / "third_party/rulstm/RULSTM/data/ek100/actions.csv")
done = {p.stem for p in (RW / "plans").glob("*.json")}
keep = val.video_id.isin(done).to_numpy()
obs = ((val.mtp_mask > 0) & (np.floor((val.action_start_frame / 30 - 2.0) * 30) >= 1)).to_numpy()
tg = {"verb": val.mtp_verbs.to_numpy(), "noun": val.mtp_nouns.to_numpy(), "action": val.action_id.to_numpy()}

def cmr(t, hit):
    p, inv = np.unique(t, return_inverse=True); return float((np.bincount(inv, weights=hit) / np.bincount(inv)).mean() * 100)

def score(top5, uid):
    assert (uid == val.uid.to_numpy()).all()
    res = {}
    for k in ("verb", "noun", "action"):
        hit = ((top5[k] == tg[k][:, None]).any(1) & obs)[keep]; res[k] = cmr(tg[k][keep], hit)
    av, an = acts.verb.to_numpy()[tg["action"]], acts.noun.to_numpy()[tg["action"]]
    hit = ((top5["verb"] == av[:, None]).any(1) & (top5["noun"] == an[:, None]).any(1) & obs)[keep]
    res["mt5r"] = cmr(tg["action"][keep], hit); return res

rows = []
z = ROOT / "outputs/collab_inavit_ek100_review/zeroshot_mf"
rows.append(("W (ours: 4s ending 2s before start)", "mf", score({k: np.load(z / f"native_bgr_{k}_top5.npy") for k in ("verb", "noun", "action")}, np.load(z / "uid.npy"))))
b = np.load(ROOT / "outputs/inavit_official_ek100/train_e100/val_epoch027.npz", allow_pickle=True)
rows.append(("W (ours: 4s ending 2s before start)", "inavit", score({k: b[f"{k}_top5"] for k in ("verb", "noun", "action")}, b["uid"])))
names = {"A1": "A1 (honest: 2.1s ending 1s before start)", "R0": "R0 (released, view 0)", "R5": "R5 (released, view 5)"}
for v in ("A1", "R0", "R5"):
    for m in ("mf", "inavit"):
        f = RW / f"results_{v}_{m}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows.append((names[v], m, {"verb": d["verb"]["cmr5"], "noun": d["noun"]["cmr5"], "action": d["action"]["cmr5"],
                                       "mt5r": d["action_released_mt5r_rule"]["cmr5"], "n": d["n_rows"]}))
print(f"subset rows: {int(keep.sum())} / {len(val)}; excluded videos: {sorted(set(val.video_id) - done)}")
print("| window | model | action CMR@5 | verb | noun | action (released MT5R rule) |")
print("|---|---|---:|---:|---:|---:|")
for w, m, r in rows:
    print(f"| {w} | {m} | {r['action']:.2f} | {r['verb']:.2f} | {r['noun']:.2f} | {r['mt5r']:.2f} |")
Path(RW / "summary.md").write_text("\n".join(f"{w}\t{m}\t{r}" for w, m, r in rows))
