#!/usr/bin/env python3
"""List reproducible B18 windows or merge complete, disjoint accuracy shards."""
import argparse
import csv
import json
from pathlib import Path
from app.hdepic_lora_action_anticipation.share_reproduction import merge_reports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    samples = sub.add_parser('samples')
    samples.add_argument('--val-csv', type=Path, required=True)
    samples.add_argument('--context-sec', type=float, default=16)
    samples.add_argument('--video-id')
    samples.add_argument('--all-windows', action='store_true')
    samples.add_argument('--limit', type=int, default=20)
    merge = sub.add_parser('merge')
    merge.add_argument('--reports', type=Path, nargs='+', required=True)
    merge.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'samples':
        seen, count = set(), 0
        if args.limit < 1:
            ap.error('--limit must be positive')
        with args.val_csv.open(newline='') as f:
            for i, row in enumerate(csv.DictReader(f)):
                vid = row['video_id']
                if abs(float(row['context_sec'])-args.context_sec) > 1e-6:
                    continue
                if args.video_id and vid != args.video_id:
                    continue
                if vid in seen and not args.all_windows:
                    continue
                print(json.dumps(dict(row_index=i, video_id=vid, context_sec=row['context_sec'],
                                      frame_indices=row['frame_indices'])))
                seen.add(vid)
                count += 1
                if count >= args.limit:
                    break
        if not count:
            ap.error('No matching windows')
    else:
        result = merge_reports([json.loads(p.read_text()) for p in args.reports])
        result['shard_files'] = [str(p) for p in args.reports]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            ap.error('Output already exists; choose a new path')
        args.out.write_text(json.dumps(result, indent=2)+'\n')
        print('strategy\tTop5@2s (%)\tTop5@4s (%)\tTop5@6s (%)')
        print(result['prune_strategy']+'\t'+'\t'.join(f"{100*result['action_top5'][h]:.4f}" for h in ('2s','4s','6s')))


if __name__ == '__main__':
    main()
