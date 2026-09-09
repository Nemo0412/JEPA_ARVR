"""Deterministic sample selection and disjoint-shard validation for shared B18 runs."""
import hashlib
import json
from pathlib import Path


def select_sample(rows, context_sec, video_id=None, row_index=None):
    candidates = [(i, r) for i, r in enumerate(rows)
                  if abs(float(r['context_sec']) - context_sec) < 1e-6
                  and (video_id is None or str(r['video_id']) == video_id)]
    if row_index is not None:
        candidates = [(i, r) for i, r in candidates if i == row_index]
    if not candidates:
        raise ValueError(f'No matching sample: context={context_sec}, video={video_id}, row={row_index}')
    return candidates[0]


def select_shard(rows, start, stop):
    stop = len(rows) if stop is None else stop
    if not 0 <= start < stop <= len(rows):
        raise ValueError(f'Invalid half-open row range [{start}, {stop}) for {len(rows)} eligible rows')
    return rows[start:stop], [start, stop], len(rows)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def merge_reports(reports):
    if not reports:
        raise ValueError('No shard reports supplied')
    expected = reports[0]
    keys = ('dataset', 'split', 'metric_scope', 'prune_strategy', 'max_frames', 'keep_count',
            'only_context_sec', 'population_size', 'input_sha256', 'evaluation_contract')
    for report in reports:
        if any(report.get(k) != expected.get(k) for k in keys):
            raise ValueError('Shard input, protocol or strategy mismatch')
        if report.get('code',{}).get('source_sha256') != expected.get('code',{}).get('source_sha256'):
            raise ValueError('Shard evaluator source mismatch')
        if report.get('partial_batches'):
            raise ValueError('Smoke or truncated batches cannot enter a full merge')
    ordered = sorted(reports, key=lambda r: r['row_range'][0])
    end = 0
    for report in ordered:
        start, stop = report['row_range']
        if start != end or stop <= start:
            raise ValueError('Overlapping, duplicate, empty or missing shard range')
        end = stop
    if end != expected['population_size']:
        raise ValueError('Incomplete population coverage')
    horizons = set(expected['n'])
    if any(set(r['n']) != horizons or set(r['action_top5']) != horizons for r in reports):
        raise ValueError('Horizon mismatch')
    counts = {h: sum(r['n'][h] for r in reports) for h in horizons}
    result = {k: expected[k] for k in keys}
    result.update(n=counts, action_top5={h: sum(r['action_top5'][h]*r['n'][h] for r in reports)/max(1,counts[h])
                                       for h in horizons}, row_range=[0,end],
                  shard_ranges=[r['row_range'] for r in ordered])
    return result


def sample_frame_indices(indices, src_fps, fps, frames, mode='csv'):
    if not indices or fps <= 0 or src_fps % fps or frames <= 0:
        raise ValueError('Invalid source indices, FPS or frame count')
    if mode == 'csv':
        selected = indices[::-1][::src_fps//fps][::-1]
        if len(selected) != frames:
            raise ValueError(f'CSV gives {len(selected)} frames, requested {frames}')
        return selected
    if mode == 'legacy_stride':
        step = indices[1]-indices[0] if len(indices)>1 else src_fps//fps
        return [indices[-1]-(frames-1-i)*step for i in range(frames)]
    raise ValueError(f'Unknown frame mode {mode}')
