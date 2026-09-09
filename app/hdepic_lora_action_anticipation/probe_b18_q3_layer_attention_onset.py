#!/usr/bin/env python3
"""B18 all-encoder/all-predictor attention readout; real forwards unchanged."""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, os, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='b18-predictor-prune/egtea-q3-layer-attention-onset-v1'
GROUP='B18-predictor-prune-keep-pattern-causes'
CONDITIONS=['continuous']
ENCODER_STAGES=[f'enc{i:02d}' for i in range(24)]
PREDICTOR_STAGES=[f'pred{i:02d}' for i in range(12)]
STAGES=ENCODER_STAGES+PREDICTOR_STAGES
GP,NS,K=256,64,4096

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def bytehash(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write_json(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def inputs(sample,frozen):
    base=np.array(np.load(sample['cache']['path'],mmap_mode='r')[-128:],copy=True)
    clips={'continuous':base}
    for c,rgb in clips.items():
        assert rgb.shape==(128,256,256,3) and rgb.dtype==np.uint8
        assert bytehash(rgb)==frozen['input_sha256'][c],(sample['sample_id'],c)
    return clips
def comparison(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    ia,ib=[set(np.argsort(-v.ravel(),kind='stable')[:K].tolist()) for v in (a,b)]
    return dict(relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),
                top4096_jaccard=len(ia&ib)/len(ia|ib),max_abs=float(np.max(np.abs(a-b))))
def prepare(a):
    old=json.loads(a.source_manifest.read_text()); source=old['source']
    assert source['n_videos']==86 and source['n_participants']==32
    assert len(old['rows'])==len(source['samples'])==86
    a.out.mkdir(parents=True,exist_ok=False); (a.out/'legacy').mkdir()
    dependencies=dict(old['code_sha256'])
    for p,h in dependencies.items(): assert digest(p)==h,('old code',p)
    for v in source['model'].values(): assert digest(v['path'])==v['sha256']
    for rel in ['app/hdepic_lora_action_anticipation/probe_b18_q3_layer_attention_onset.py',
                'scripts/egtea/analyze_b18_q3_layer_attention_onset.py',
                'scripts/egtea/run_b18_q3_layer_onset_gpu.slurm','scripts/egtea/run_b18_q3_layer_onset_cpu.slurm']:
        dependencies[str(ROOT/rel)]=digest(ROOT/rel)
    rows=[]
    for i,(sample,row) in enumerate(zip(source['samples'],old['rows'])):
        assert sample['sample_id']==row['sample_id']
        assert digest(sample['cache']['path'])==sample['cache']['sha256']; inputs(sample,row)
        shard='b18-q3-mechanism-full-a-17092450' if i<43 else 'b18-q3-mechanism-full-b-17092454'
        archive=a.source_manifest.parent.parent/shard/'capture'/f"{sample['sample_id']}.npz"
        assert archive.is_file(),archive
        with np.load(archive) as z: scores={c:z['score_actual__'+c] for c in CONDITIONS}
        assert all(v.shape==(64,256) and np.isfinite(v).all() for v in scores.values())
        target=a.out/'legacy'/f"{sample['sample_id']}.npz"; np.savez_compressed(target,**scores)
        rows.append(dict(row,index=i,legacy_path=str(target),legacy_sha256=digest(target),
                         legacy_original_path=str(archive),legacy_original_sha256=digest(archive)))
    write_json(a.out/'manifest.json',dict(protocol=PROTOCOL,group=GROUP,conditions=CONDITIONS,stages=STAGES,
               source_manifest=str(a.source_manifest),source_manifest_sha256=digest(a.source_manifest),
               source=source,rows=rows,code_sha256=dependencies,query_chunk=256,anchor_indices=[0,1],
               hardware_gate=dict(relative_l2_max=.005,top4096_jaccard_min=.98),
               selection='stable descending FP32 score; ascending original index resolves exact ties; K4096 context keys',
               onset_rule='first execution-order stage where full and both halves have oldest-pair mass/uniform >1 and pooled topK mean >64, with receiver-paired oldest-minus-middle mass and keep-count bootstrap95% lower bounds >0',
               dtype='actual BF16 autocast QK, native FP32 softmax/reductions; no probability cast to BF16'))
    print('PREPARED',len(rows),digest(a.out/'manifest.json'),flush=True)
class CaptureComplete(Exception): pass
class LayerCapture:
    def __init__(self,m,name):
        self.m,self.name,self.original=m,name,m.forward; self.data,self.runtime={},{}; m.forward=self.forward
    def remove(self): self.m.forward=self.original
    def forward(self,x,mask=None,attn_mask=None,T=None,H_patches=None,W_patches=None):
        import torch
        from src.models.utils.modules import rotate_queries_or_keys
        m=self.m
        out=self.original(x,mask=mask,attn_mask=attn_mask,T=T,H_patches=H_patches,W_patches=W_patches)
        B,N,C=x.shape; slots=65 if self.name.startswith('pred') else 64
        assert B==1 and N==slots*GP and attn_mask is None and not m.is_causal
        assert not m.training and m.proj_drop_prob==0
        qkv=m.qkv(x).unflatten(-1,(3,m.num_heads,-1)).permute(2,0,3,1,4); q,k=qkv[0],qkv[1]
        if mask is not None: mp=mask.unsqueeze(1).repeat(1,m.num_heads,1)
        else:
            depth=N//(m.grid_size*m.grid_size)
            count=depth*m.grid_size*m.grid_size if any(z is None for z in (T,H_patches,W_patches)) else T*H_patches*W_patches
            mp=torch.arange(count,device=x.device)
        positions=m.separate_positions(mp,H_patches,W_patches)
        expected=torch.arange(N,device=x.device)
        if slots==65: expected[-256:]+=(72-64)*256
        actual=mp.reshape(-1,N)[0] if mp.ndim>1 else mp
        assert torch.equal(actual,expected),(self.name,'positions')
        qp,kp,offset=[],[],0
        for width,pos in zip((m.d_dim,m.h_dim,m.w_dim),positions):
            qp.append(rotate_queries_or_keys(q[...,offset:offset+width],pos=pos))
            kp.append(rotate_queries_or_keys(k[...,offset:offset+width],pos=pos)); offset+=width
        if offset<m.head_dim: qp.append(q[...,offset:]); kp.append(k[...,offset:])
        q,k=torch.cat(qp,-1),torch.cat(kp,-1)
        imp=torch.zeros((m.num_heads,N),device=x.device,dtype=torch.float32)
        ctx=torch.zeros_like(imp) if slots==65 else None
        flow=torch.zeros((m.num_heads,slots,slots),device=x.device,dtype=torch.float32)
        diagonal=torch.zeros((m.num_heads,slots),device=x.device,dtype=torch.float32)
        same,row_mass=torch.zeros_like(diagonal),torch.zeros_like(diagonal)
        for qs in range(slots):
            lo=qs*GP; logits=(q[:,:,lo:lo+GP]@k.transpose(-2,-1))*m.scale
            p=logits.softmax(dim=-1); assert p.dtype==torch.float32
            received=p.sum(dim=2)[0]; imp+=received
            if ctx is not None and qs<NS: ctx+=received
            flow[:,qs]=received.reshape(m.num_heads,slots,GP).sum(-1)
            within=p[0,:,:,lo:lo+GP]
            diagonal[:,qs]=within.diagonal(dim1=-2,dim2=-1).sum(-1)
            same[:,qs]=within.sum((-1,-2)); row_mass[:,qs]=p.sum((-1,-2))[0]
            del p,received,within
        assert torch.allclose(flow.sum(1),imp.reshape(m.num_heads,slots,GP).sum(-1),rtol=1e-5,atol=.03)
        assert torch.allclose(flow.sum(-1),row_mass,rtol=1e-5,atol=.003)
        assert torch.allclose(same,flow.diagonal(dim1=-2,dim2=-1),rtol=1e-5,atol=.003)
        assert torch.allclose(row_mass,torch.full_like(row_mass,GP),rtol=1e-5,atol=.003)
        assert bool((diagonal>=0).all() and (diagonal<=same+.003).all())
        self.data=dict(received=imp,flow=flow,exact_self=diagonal,same_slot=same,other_slot=row_mass-same,row_mass=row_mass)
        self.data['score']=imp[:,:16384].sum(0).reshape(64,256)
        if ctx is not None:
            self.data['received_context_queries']=ctx
            self.data['score_context_queries']=ctx[:,:16384].sum(0).reshape(64,256)
        self.runtime=dict(N=N,heads=m.num_heads,head_dim=m.head_dim,q_dtype=str(q.dtype),logits_dtype=str(logits.dtype),
                          softmax_dtype='torch.float32',reduction_dtype=str(row_mass.dtype),causal=m.is_causal,attn_mask=None,
                          mask_shape=list(mask.shape) if mask is not None else None,
                          temporal_positions=list(range(64))+([72] if slots==65 else []),T=T,H_patches=H_patches,W_patches=W_patches)
        return out
class NormCapture:
    def __init__(self,base):
        import torch
        self.data,self.handles={},[]; modules={'patch':base.encoder.patch_embed}
        for i,b in enumerate(base.encoder.blocks):
            modules[f'enc{i:02d}_norm1']=b.norm1; modules[f'enc{i:02d}_residual']=b
        modules.update(encoder_final_norm=base.encoder.norm,predictor_embed=base.predictor.predictor_embed,
                       predictor_l0_norm1=base.predictor.predictor_blocks[0].norm1)
        for name,m in modules.items():
            def hook(m,a,out,name=name):
                assert out.ndim==3 and out.shape[0]==1 and out.shape[1]%GP==0
                z=out[0].float().reshape(-1,GP,out.shape[-1]); norm=z.norm(dim=-1)
                self.data[name]=torch.stack((norm.mean(-1),z.mean(1).norm(dim=-1)),-1)
            self.handles.append(m.register_forward_hook(hook))
    def remove(self):
        for h in self.handles: h.remove()
def gpu(a):
    import torch
    from app.hdepic_lora_action_anticipation import train_stream_mtp as T
    from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20,build_finetuned_20
    init=time.time(); torch.set_num_threads(1); torch.manual_seed(20260907)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    man=json.loads(a.manifest.read_text()); assert man['protocol']==PROTOCOL and man['conditions']==CONDITIONS
    for p,h in man['code_sha256'].items(): assert digest(p)==h,('frozen code',p)
    assert digest(man['source_manifest'])==man['source_manifest_sha256']
    source=man['source']
    for v in source['model'].values(): assert digest(v['path'])==v['sha256']
    a.out.mkdir(parents=True,exist_ok=False); loadlog=io.StringIO(); paths={k:v['path'] for k,v in source['model'].items()}
    with contextlib.redirect_stdout(loadlog):
        base,_=build_finetuned_20(torch.device('cuda'),max_frames=256,fps=8,img_size=256,
                                 checkpoint=paths['checkpoint'],enc_lora=paths['encoder_lora'],pred_lora=paths['predictor_lora'],parent_ckpt=paths['parent'])
    (a.out/'model_load.txt').write_text(loadlog.getvalue())
    assert '[finetuned] parent load: 0 missing, 0 unexpected' in loadlog.getvalue()
    assert len(base.encoder.blocks)==24 and len(base.predictor.predictor_blocks)==12
    assert base.num_steps==1 and base.tubelet_size==2
    base.requires_grad_(False); model=T.PrunedAnticipativeModel(base,None,prune_threshold=10**9).eval()
    modules={s:b.attn for s,b in zip(ENCODER_STAGES,base.encoder.blocks)}
    modules.update({s:b.attn for s,b in zip(PREDICTOR_STAGES,base.predictor.predictor_blocks)})
    mean,std=T.IMAGENET_MEAN.cuda(),T.IMAGENET_STD.cuda(); ant=torch.full((1,),2.,device='cuda')
    def run(rgb,early):
        x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(3,0,1,2).unsqueeze(0).cuda()
        x=x.float().div_(255).sub_(mean).div_(std)
        def abort(*unused): raise CaptureComplete()
        hook=modules['pred11'].register_forward_hook(abort) if early else None; result=None
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): result=model(x,ant)
            assert not early
        except CaptureComplete: assert early
        finally:
            if hook: hook.remove()
        torch.cuda.synchronize(); return result
    def collect(caps,norms):
        arrays={}
        for s,cap in caps.items():
            for k,v in cap.data.items():
                assert bool(torch.isfinite(v).all()),(s,k); arrays[f'{s}__{k}']=v.float().cpu().numpy()
        for k,v in norms.data.items(): arrays['norm__'+k]=v.float().cpu().numpy()
        return arrays
    def instrument(): return {s:LayerCapture(m,s) for s,m in modules.items()},NormCapture(base)
    def remove(caps,norms):
        for c in caps.values(): c.remove()
        norms.remove()
    def exact_output(a,b):
        if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(exact_output(x,y) for x,y in zip(a,b))
        if isinstance(a,dict): return a.keys()==b.keys() and all(exact_output(a[k],b[k]) for k in a)
        return a==b
    gates,anchors={},{}; (a.out/'anchors').mkdir()
    for i in [0,1]:
        sample,row=source['samples'][i],man['rows'][i]
        for cond,rgb in inputs(sample,row).items():
            key=f"{sample['sample_id']}__{cond}"
            if a.smoke:
                plain=run(rgb,False); caps,norms=instrument(); full=run(rgb,False)
                assert exact_output(plain,full),(key,'model output parity')
                fullstats=collect(caps,norms); run(rgb,True); arrays=collect(caps,norms)
                assert fullstats.keys()==arrays.keys() and all(np.array_equal(v,arrays[k]) for k,v in fullstats.items()),(key,'full early')
                runtime={s:c.runtime for s,c in caps.items()}; remove(caps,norms)
                refs={s:HeadAttnCapture20(m,chunk_size=GP) for s,m in modules.items()}; run(rgb,True)
                for s,r in refs.items():
                    assert np.array_equal(arrays[f'{s}__received'],r.importance[0].float().cpu().numpy()),(key,s,'reference')
                    r.remove()
                gates[key]=dict(plain_instrumented_output_exact=True,full_early_all_arrays_exact=True,independent_reference_all36_exact=True)
                del plain,full,fullstats,refs
            else:
                assert a.anchors is not None
                caps,norms=instrument(); run(rgb,True); arrays=collect(caps,norms)
                runtime={s:c.runtime for s,c in caps.items()}; remove(caps,norms)
                with np.load(a.anchors/'anchors'/(key+'.npz')) as ref:
                    checks={}
                    for s in STAGES:
                        x,y=arrays[f'{s}__received'],ref[f'{s}__received']
                        chk=comparison(x[:,:16384].sum(0),y[:,:16384].sum(0))
                        chk['head_relative_l2']=float(np.linalg.norm(x.astype(np.float64)-y)/np.linalg.norm(y.astype(np.float64)))
                        assert chk['relative_l2']<=.005 and chk['head_relative_l2']<=.005 and chk['top4096_jaccard']>=.98,(key,s,chk)
                        checks[s]=chk
                    gates[key]=checks
            np.savez_compressed(a.out/'anchors'/(key+'.npz'),**arrays); anchors[key]=digest(a.out/'anchors'/(key+'.npz'))
            print('ONSET_ANCHOR',key,'all36_pass',flush=True)
    caps,norms=instrument()
    meta=dict(protocol=PROTOCOL,group=GROUP,tag=a.tag,job_id=os.environ.get('SLURM_JOB_ID'),
              manifest=str(a.manifest),manifest_sha256=digest(a.manifest),gpu=torch.cuda.get_device_name(),
              torch_version=torch.__version__,cuda=torch.version.cuda,runtime=runtime,start=a.start,stop=a.stop,smoke=a.smoke,
              gates=gates,anchors=anchors,source=source['model'],code_sha256=man['code_sha256'],
              initialization_and_gates_seconds=time.time()-init,compute_start=time.time())
    write_json(a.out/'metadata.json',meta); print('ONSET_COMPUTE_START',meta['compute_start'],flush=True)
    times,legacy=[],[]
    for i in range(a.start,a.stop):
        start=time.time(); sample,row=source['samples'][i],man['rows'][i]; arrays={'sample_index':np.asarray(i)}
        assert digest(sample['cache']['path'])==sample['cache']['sha256'] and digest(row['legacy_path'])==row['legacy_sha256']
        with np.load(row['legacy_path']) as old:
            for cond,rgb in inputs(sample,row).items():
                run(rgb,True); values=collect(caps,norms)
                arrays.update({cond+'__'+k:v for k,v in values.items()})
                chk=comparison(values['pred00__score'],old[cond]); chk.update(index=i,condition=cond)
                assert chk['relative_l2']<=.005 and chk['top4096_jaccard']>=.98,chk
                legacy.append(chk)
        target=a.out/f"{sample['sample_id']}.npz"; temporary=target.with_suffix('.partial.npz')
        np.savez_compressed(temporary,**arrays); temporary.rename(target)
        times.append(time.time()-start); print('ONSET_ROW',i,times[-1],flush=True)
    benchstart=time.time(); repeats=0
    if a.smoke:
        rgb=inputs(source['samples'][0],man['rows'][0])['continuous']
        while time.time()-benchstart<30: run(rgb,True); repeats+=1
    benchend=time.time(); remove(caps,norms)
    summary=dict(protocol=PROTOCOL,run_status='completed',start=a.start,stop=a.stop,n_rows=a.stop-a.start,n_conditions=1,n_stages=36,
                 sample_seconds=times,compute_start=meta['compute_start'],compute_end=time.time(),
                 initialization_and_gates_seconds=meta['initialization_and_gates_seconds'],benchmark_start=benchstart,benchmark_end=benchend,
                 benchmark_repeats=repeats,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,legacy_checks=legacy,gates=gates,runtime=runtime)
    write_json(a.out/'summary.json',summary)
    print('ONSET_COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ['gates','runtime','legacy_checks']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--source-manifest',type=Path,required=True); q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('gpu'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--out',type=Path,required=True)
    q.add_argument('--start',type=int,default=0); q.add_argument('--stop',type=int,default=2); q.add_argument('--tag',required=True)
    q.add_argument('--smoke',action='store_true'); q.add_argument('--anchors',type=Path)
    a=p.parse_args(); prepare(a) if a.mode=='prepare' else gpu(a)
