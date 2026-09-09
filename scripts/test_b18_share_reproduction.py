"""Regression checks for reproducible windows and complete shard aggregation."""
from copy import deepcopy
from app.hdepic_lora_action_anticipation.share_reproduction import select_sample, select_shard, merge_reports


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected a rejected invalid reproduction input')


rows = [dict(video_id='A', context_sec=4), dict(video_id='A', context_sec=16),
        dict(video_id='B', context_sec=16), dict(video_id='A', context_sec=16)]
assert select_sample(rows,16,'A') == (1,rows[1])
assert select_sample(rows,16,'A',3) == (3,rows[3])
reject(lambda: select_sample(rows,16,'missing'))
reject(lambda: select_sample(rows,16,'A',2))
reject(lambda: select_sample(rows,16,row_index=0))
assert select_shard(rows,1,3) == (rows[1:3],[1,3],4)
for a,b in [(0,0),(-1,2),(1,5),(4,None)]:
    reject(lambda: select_shard(rows,a,b))
base = dict(dataset='EGTEA',split='temporal-half',metric_scope='native',prune_strategy='recent',
            max_frames=128,keep_count=4096,only_context_sec=16,population_size=4,
            input_sha256={'train_csv':'frozen'},evaluation_contract={'horizons_sec':'2,4,6'},
            code={'source_sha256':{'evaluator':'same'}},partial_batches=False)
a = dict(base,row_range=[0,2],n={'2s':2,'4s':1,'6s':1},action_top5={'2s':.5,'4s':1.,'6s':0.})
b = dict(base,row_range=[2,4],n={'2s':2,'4s':2,'6s':0},action_top5={'2s':1.,'4s':.5,'6s':0.})
m = merge_reports([b,a])
assert m['n'] == {'2s':4,'4s':3,'6s':1}
assert m['action_top5'] == {'2s':.75,'4s':2/3,'6s':0.}
reject(lambda: merge_reports([a]))
reject(lambda: merge_reports([a,a,b]))
reject(lambda: merge_reports([dict(a,partial_batches=True),b]))
reject(lambda: merge_reports([a,dict(b,input_sha256={'train_csv':'other'})]))
reject(lambda: merge_reports([a,dict(b,row_range=[3,4])]))
print('PASS sample identity, no fallback, shard coverage, and denominator-weighted merge')

from app.hdepic_lora_action_anticipation.share_reproduction import sample_frame_indices
assert sample_frame_indices([10,13,17,20],8,8,4) == [10,13,17,20]
assert sample_frame_indices([10,13,17,20],8,4,2) == [13,20]
assert sample_frame_indices([10,13,17,20],8,8,4,'legacy_stride') == [11,14,17,20]
reject(lambda: sample_frame_indices([10,13,17,20],8,8,3))
reject(lambda: sample_frame_indices([10,13,17,20],8,3,2))
print('PASS irregular CSV sampling, newest-anchored FPS, and explicit historical frame mode')
