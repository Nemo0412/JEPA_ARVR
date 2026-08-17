"""Build V-JEPA2 annotation CSVs for EGTEA Gaze+ — V1 protocol (2026-07-01).

V1 fix: use SESSION-LEVEL start/stop frames (sf, ef from clip name F<sf>-F<ef>)
with session_videos/<session>.mp4 as the video source.  This enables the standard
HD-EPIC anticipation sampling (observe before action start) instead of observing
during the action clip (which was V0's mistake).

CSV schema (same 10-col HD-EPIC schema + action_class):
  participant_id, video_id, start_frame, stop_frame, verb_class, noun_class,
  all_noun_classes, start_timestamp, stop_timestamp, narration_id, action_class

video_id = "<session>_<rest>" (underscore-joined) so the existing _clean_video_id()
in gaze.py reconstructs the clip name for gaze lookup.  The actual video file is
vjepa_videos/<session>/<video_id>.MP4 → symlink → session_videos/<session>.mp4.

Usage:
    python3 build_egtea_csvs_v2.py [--data-root DIR] [--split 1|2|3|all]
                                    [--no-symlinks] [--val-frac 0.1]
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("numpy/pandas not found — run: module load anaconda3/2025.06")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_clip_name(name):
    """'<session>-...-F<sf>-F<ef>' -> (session, sf, ef)."""
    m = re.search(r"-F(\d+)-F(\d+)$", name)
    if not m:
        raise ValueError(f"no F<sf>-F<ef> suffix in: {name}")
    sf, ef = int(m.group(1)), int(m.group(2))
    # session = everything before the first timestamp segment (-<digits>-<digits>-F...)
    session = re.sub(r"-\d+-\d+-F\d+-F\d+$", "", name)
    return session, sf, ef


def read_split_file(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        name = parts[0]
        action_id = int(parts[1])  # 1-based
        verb_id = int(parts[2])    # 1-based
        nouns = [int(x) for x in parts[3:]]
        rows.append((name, action_id, verb_id, nouns))
    return rows


def build_df(rows, action_idx, verb_idx, noun_idx):
    """Convert raw split rows to a flat DataFrame with 0-based class indices."""
    records = []
    for (name, action_id, verb_id, nouns) in rows:
        try:
            session, sf, ef = parse_clip_name(name)
        except ValueError as e:
            print(f"  [warn] {e}", file=sys.stderr)
            continue
        # video_id: replace first '-' between session and rest with '_'
        # session = e.g. "OP01-R01-PastaSalad", name = "OP01-R01-PastaSalad-...-F...-F..."
        # video_id replaces only the separator between session and start-ms with '_'
        rest = name[len(session)+1:]  # everything after "<session>-"
        video_id = session + "_" + rest

        # 0-based class indices (input IDs are 1-based)
        verb_cls = verb_id - 1
        noun_cls = nouns[0] - 1 if nouns else 0
        all_nouns = ";".join(str(n - 1) for n in nouns)
        action_cls = action_id - 1

        records.append({
            "participant_id": session,
            "video_id": video_id,
            "start_frame": sf,         # session-level 24fps frame index
            "stop_frame": ef,           # session-level 24fps frame index
            "verb_class": verb_cls,
            "noun_class": noun_cls,
            "all_noun_classes": all_nouns,
            "start_timestamp": "",
            "stop_timestamp": "",
            "narration_id": name,       # original clip name for traceability
            "action_class": action_cls,
        })
    return pd.DataFrame(records)


def stratified_val_split(df, val_frac, seed=42):
    """Carve val_frac of each action class from df; remainder is train."""
    rng = np.random.default_rng(seed)
    val_idx = []
    for cls, grp in df.groupby("action_class"):
        n = max(1, round(len(grp) * val_frac))
        chosen = rng.choice(grp.index.tolist(), size=n, replace=False)
        val_idx.extend(chosen.tolist())
    val_mask = df.index.isin(val_idx)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def make_symlinks(df, vjepa_videos_dir, session_videos_dir):
    """For each row, symlink vjepa_videos/<session>/<video_id>.MP4 → session video."""
    n_ok = n_miss = 0
    for _, row in df.iterrows():
        session = row["participant_id"]
        video_id = row["video_id"]
        src = session_videos_dir / f"{session}.mp4"
        dst_dir = vjepa_videos_dir / session
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{video_id}.MP4"
        if not src.exists():
            n_miss += 1
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        n_ok += 1
    return n_ok, n_miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data/egtea"))
    ap.add_argument("--split", default="1", choices=["1", "2", "3", "all"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--no-symlinks", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    ann_dir = data_root / "action_annotation"
    session_videos_dir = data_root / "session_videos"
    vjepa_videos_dir = data_root / "vjepa_videos"

    # Load class index files (1-based)
    def load_idx(fname):
        # Format: "<label_words> <id>" — id is the LAST token, label is everything before
        idx = {}
        for line in (ann_dir / fname).read_text().splitlines():
            parts = line.strip().rsplit(maxsplit=1)
            if len(parts) == 2 and parts[1].isdigit():
                idx[int(parts[1])] = parts[0]
        return idx

    action_idx = load_idx("action_idx.txt")
    verb_idx = load_idx("verb_idx.txt")
    noun_idx = load_idx("noun_idx.txt")
    print(f"Classes: {len(action_idx)} actions, {len(verb_idx)} verbs, {len(noun_idx)} nouns")

    splits = ["1", "2", "3"] if args.split == "all" else [args.split]

    for split in splits:
        out_dir = data_root / "vjepa_annotations" / "v1" / f"split{split}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_file = ann_dir / f"train_split{split}.txt"
        test_file = ann_dir / f"test_split{split}.txt"

        print(f"\n=== Split {split} ===")
        train_rows = read_split_file(train_file)
        test_rows = read_split_file(test_file)

        train_df_full = build_df(train_rows, action_idx, verb_idx, noun_idx)
        test_df = build_df(test_rows, action_idx, verb_idx, noun_idx)

        train_df, val_df = stratified_val_split(train_df_full, args.val_frac)

        print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        print(f"  action classes in train: {train_df['action_class'].nunique()}")

        cols = ["participant_id", "video_id", "start_frame", "stop_frame",
                "verb_class", "noun_class", "all_noun_classes",
                "start_timestamp", "stop_timestamp", "narration_id", "action_class"]

        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            p = out_dir / f"EGTEA_{name}_vjepa.csv"
            if p.exists() and not args.overwrite:
                print(f"  [skip] {p} exists (--overwrite to replace)")
                continue
            df[cols].to_csv(p, index=False)
            print(f"  wrote {p}")

        if not args.no_symlinks:
            if not session_videos_dir.exists():
                print(f"  [warn] session_videos/ not found at {session_videos_dir}; skipping symlinks")
            else:
                all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
                unique_df = all_df.drop_duplicates(subset=["video_id"])
                n_ok, n_miss = make_symlinks(unique_df, vjepa_videos_dir, session_videos_dir)
                print(f"  symlinks: {n_ok} ok, {n_miss} missing session videos")

    print("\nDone.")


if __name__ == "__main__":
    main()
