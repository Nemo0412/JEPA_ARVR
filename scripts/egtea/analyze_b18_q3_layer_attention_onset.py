#!/usr/bin/env python3
"""Independent CPU reconstruction and full-stage figures for B18 Q3 onset."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PROTOCOL='b18-predictor-prune/egtea-q3-layer-attention-onset-v1'
CONDS=['continuous']
ENCODER_STAGES=[f'enc{i:02d}' for i in range(24)]
PREDICTOR_STAGES=[f'pred{i:02d}' for i in range(12)]
STAGES=ENCODER_STAGES+PREDICTOR_STAGES
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def js(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mask(x):
    flat=x.ravel(); ix=np.argsort(-flat,kind='stable')[:4096]
    m=np.zeros(flat.size,bool); m[ix]=True; m=m.reshape(64,256)
    cut=float(flat[ix[-1]]); strict=(x>cut).sum(-1); ties=(x==cut).sum(-1); nt=int(ties.sum()); quota=4096-int(strict.sum())
    lo=strict+np.maximum(0,quota-(nt-ties)); hi=strict+np.minimum(quota,ties)
    assert m.sum()==4096
    return m,dict(cutoff=cut,strict_count=int(strict.sum()),tie_count=nt,ties_selected=quota,
                  count_lower=lo.tolist(),count_upper=hi.tolist(),ambiguous_slots=int((lo!=hi).sum()))
def ratios(x):
    v=x.mean(-1)/x.mean()
    return dict(slot0=float(v[0]),slot1=float(v[1]),oldest_pair=float(v[:2].mean()),
                remaining2_64=float(v[2:].mean()),middle16_48=float(v[16:48].mean()))
def compare(x,y):
    r=float(np.linalg.norm(x.astype('float64')-y)/np.linalg.norm(y.astype('float64')))
    a,b=mask(x)[0],mask(y)[0]; j=float((a&b).sum()/(a|b).sum())
    assert r<=.005 and j>=.98,(r,j)
    return dict(relative_l2=r,top4096_jaccard=j)
def paired_bootstrap(values,seed,reps=10000):
    values=np.asarray(values,dtype=np.float64); rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),size=(reps,len(values)))].mean(1)
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                positive=int((values>0).sum()),zero=int((values==0).sum()),negative=int((values<0).sum()),
                bootstrap_seed=seed,bootstrap_reps=reps,bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
                quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist())
def telemetry(run,summary):
    rows=[]
    for line in (run.parent/'gpu.csv').read_text().splitlines():
        p=[v.strip() for v in line.split(',')]
        try: rows.append((datetime.strptime(p[0],'%Y/%m/%d %H:%M:%S.%f').timestamp(),float(p[2]),float(p[4])))
        except (ValueError,IndexError): pass
    assert rows
    t=np.asarray(rows)
    compute=t[(t[:,0]>=summary['compute_start'])&(t[:,0]<=summary['compute_end'])]
    bench=t[(t[:,0]>=summary['benchmark_start'])&(t[:,0]<=summary['benchmark_end'])]
    return dict(samples=len(t),interval_seconds=5,first_epoch=float(t[0,0]),last_epoch=float(t[-1,0]),
                whole_gpu_util=float(t[:,1].mean()),compute_gpu_util=float(compute[:,1].mean()) if len(compute) else None,
                benchmark_gpu_util=float(bench[:,1].mean()) if len(bench) else None,benchmark_samples=len(bench),
                peak_memory_mib=float(t[:,2].max()))
def main(a):
    man=json.loads(a.manifest.read_text()); assert man['protocol']==PROTOCOL and man['conditions']==CONDS and man['stages']==STAGES
    assert digest(man['source_manifest'])==man['source_manifest_sha256']
    for p,h in man['code_sha256'].items(): assert digest(p)==h
    a.out.mkdir(parents=True,exist_ok=False)
    indices=[]; rows=[]; ctxscores=[]; norms=[]; flow_sums={}; diag_sums={}; artifacts=[]; jobs=[]; legacy=[]
    audit=dict(flow_column=True,flow_row=True,diagonal_same_other=True,finite=True,position_runtime=True)
    for run in a.runs:
        meta=json.loads((run/'metadata.json').read_text()); sm=json.loads((run/'summary.json').read_text())
        assert meta['protocol']==PROTOCOL and meta['manifest_sha256']==digest(a.manifest) and sm['run_status']=='completed' and sm['n_stages']==36
        assert set(meta['runtime'])==set(STAGES)
        for s,r in meta['runtime'].items():
            n=65 if s.startswith('pred') else 64
            assert r['N']==n*256 and not r['causal'] and r['attn_mask'] is None
            assert r['q_dtype']==r['logits_dtype']=='torch.bfloat16'
            assert r['softmax_dtype']==r['reduction_dtype']=='torch.float32'
            assert r['temporal_positions']==list(range(64))+([72] if n==65 else [])
        jobs.append(dict(job_id=meta['job_id'],gpu=meta['gpu'],run=str(run),summary=sm,telemetry=telemetry(run,sm)))
        for i in range(meta['start'],meta['stop']):
            assert i not in indices
            sample=man['source']['samples'][i]; path=run/f"{sample['sample_id']}.npz"
            with np.load(path) as z:
                assert int(z['sample_index'])==i
                score=np.empty((1,36,64,256),np.float32); ctx=np.empty((1,12,64,256),np.float32)
                nrow={}
                for ci,c in enumerate(CONDS):
                    for si,s in enumerate(STAGES):
                        pre=c+'__'+s+'__'; imp=z[pre+'received']; f=z[pre+'flow']
                        d=z[pre+'exact_self']; same=z[pre+'same_slot']; other=z[pre+'other_slot']; rm=z[pre+'row_mass']
                        h,n=imp.shape[0],65 if s.startswith('pred') else 64
                        assert imp.shape==(h,n*256) and f.shape==(h,n,n) and d.shape==(h,n)
                        assert all(np.isfinite(v).all() for v in [imp,f,d,same,other,rm])
                        np.testing.assert_allclose(f.sum(1,dtype=np.float64),imp.reshape(h,n,256).sum(-1,dtype=np.float64),rtol=1e-5,atol=.03)
                        np.testing.assert_allclose(f.sum(-1,dtype=np.float64),rm,rtol=1e-5,atol=.003)
                        np.testing.assert_allclose(np.diagonal(f,axis1=-2,axis2=-1),same,rtol=1e-5,atol=.003)
                        np.testing.assert_allclose(same+other,rm,rtol=1e-5,atol=.003)
                        np.testing.assert_allclose(rm,256,rtol=1e-5,atol=.003)
                        assert (d>=0).all() and (d<=same+.003).all()
                        np.testing.assert_allclose(z[pre+'score'],imp[:,:16384].sum(0).reshape(64,256),rtol=1e-6,atol=1e-5)
                        score[ci,si]=z[pre+'score']
                        if s.startswith('pred'):
                            pi=PREDICTOR_STAGES.index(s); ctximp=z[pre+'received_context_queries']; ctx[ci,pi]=z[pre+'score_context_queries']
                            np.testing.assert_allclose(ctx[ci,pi],ctximp[:,:16384].sum(0).reshape(64,256),rtol=1e-6,atol=1e-5)
                            np.testing.assert_allclose(f[:,:64].sum(1,dtype=np.float64),
                                ctximp.reshape(h,n,256).sum(-1,dtype=np.float64),rtol=1e-5,atol=.03)
                        key=c+'__'+s
                        if key not in flow_sums:
                            flow_sums[key]=np.zeros_like(f,dtype=np.float64); diag_sums[key]=np.zeros_like(d,dtype=np.float64)
                        flow_sums[key]+=f; diag_sums[key]+=d
                    for k in z.files:
                        if k.startswith(c+'__norm__'):
                            arr=z[k]; assert np.isfinite(arr).all(); nrow[k]=arr[:64]
                    with np.load(man['rows'][i]['legacy_path']) as old:
                        chk=compare(score[ci,24],old[c]); chk.update(index=i,condition=c); legacy.append(chk)
                rows.append(score); ctxscores.append(ctx); norms.append(nrow)
            indices.append(i); artifacts.append(dict(index=i,path=str(path),sha256=digest(path)))
    order=np.argsort(indices); indices=np.asarray(indices)[order]; scores=np.stack(rows)[order]; ctxscores=np.stack(ctxscores)[order]
    norms=[norms[i] for i in order]; n=len(indices)
    if not a.allow_partial: assert np.array_equal(indices,np.arange(86))
    else: assert n>=2
    halves=np.asarray([man['source']['samples'][int(i)]['calibration_half'] for i in indices])
    half_values=sorted(set(halves.tolist()))
    if not a.allow_partial: assert len(half_values)==2 and all((halves==v).sum()==43 for v in half_values)
    pooled=scores.mean(0); pc=ctxscores.mean(0); pack={}; tables=[]; individual=[]; summaries={}
    for ci,c in enumerate(CONDS):
        summaries[c]={}
        for si,s in enumerate(STAGES):
            score=pooled[ci,si]; m,ties=mask(score); rat=ratios(score); counts=m.sum(-1)
            halfstats={}
            for hv in half_values:
                hs=scores[halves==hv,ci,si].mean(0); hm,ht=mask(hs)
                hc=hm.sum(-1); hr=ratios(hs)
                halfstats[str(hv)]=dict(n=int((halves==hv).sum()),ratios=hr,counts=hc.tolist(),
                    oldest_pair_keep_mean=float(hc[:2].mean()),middle16_48_keep_mean=float(hc[16:48].mean()),ties=ht)
                pack[f'{c}__{s}__half{hv}_score']=hs; pack[f'{c}__{s}__half{hv}_mask']=hm
            indivrat=[]; indivcounts=[]; massdiff=[]; keepdiff=[]
            for ii,idx in enumerate(indices):
                sample_score=scores[ii,ci,si]; rr=ratios(sample_score); mm,_=mask(sample_score); mc=mm.sum(-1)
                indivrat.append(rr['oldest_pair']); indivcounts.append(float(mc[:2].mean()))
                massdiff.append(rr['oldest_pair']-rr['middle16_48']); keepdiff.append(float(mc[:2].mean()-mc[16:48].mean()))
                individual.append(dict(index=int(idx),sample_id=man['rows'][int(idx)]['sample_id'],condition=c,stage=s,
                                       oldest_pair_ratio=rr['oldest_pair'],middle_ratio=rr['middle16_48'],oldest_minus_middle_mass=massdiff[-1],
                                       oldest_keep_mean=indivcounts[-1],middle_keep_mean=float(mc[16:48].mean()),oldest_minus_middle_keep=keepdiff[-1]))
            flow=flow_sums[c+'__'+s].sum(0)/n; diag=diag_sums[c+'__'+s].sum(0)/n
            denom=flow[:,:64].sum(0); incoming=flow[:,:64]/denom[None,:]
            same=np.diag(flow)[:64]/denom; selfshare=diag[:64]/denom
            src=dict(exact_self=selfshare.tolist(),same_slot_other=(same-selfshare).tolist(),same_slot=same.tolist(),
                     other_slot=(1-same).tolist(),largest_source_slot=incoming.argmax(0).tolist(),
                     largest_source_share=incoming.max(0).tolist(),
                     same_slot_largest_count=int((incoming.argmax(0)==np.arange(64)).sum()),
                     any_source_majority_count=int((incoming.max(0)>.5).sum()),same_slot_majority_count=int((same>.5).sum()))
            np.testing.assert_allclose(incoming.sum(0),1,rtol=1e-12,atol=1e-12)
            summary=dict(ratios=rat,counts=counts.tolist(),oldest_pair_keep_mean=float(counts[:2].mean()),
                         middle16_48_keep_mean=float(counts[16:48].mean()),ties=ties,halves=halfstats,
                         individual=dict(n=n,above1=int((np.asarray(indivrat)>1).sum()),above1_05=int((np.asarray(indivrat)>1.05).sum()),
                                         oldest_pair_ratio_quantiles=np.quantile(indivrat,[0,.1,.5,.9,1]).tolist(),
                                         oldest_keep_quantiles=np.quantile(indivcounts,[0,.1,.5,.9,1]).tolist(),
                                         paired_oldest_minus_middle_mass=paired_bootstrap(massdiff,20260907+si),
                                         paired_oldest_minus_middle_keep=paired_bootstrap(keepdiff,20261907+si)),incoming=src)
            summaries[c][s]=summary
            pack[c+'__'+s+'__score']=score; pack[c+'__'+s+'__mask']=m
            pack[c+'__'+s+'__flow']=flow; pack[c+'__'+s+'__incoming_column_normalized']=incoming
            for slot in range(64):
                tables.append(dict(condition=c,stage=s,slot=slot,received_mass_uniform=float(score[slot].mean()/score.mean()),
                                   kept=int(counts[slot]),kept_lower=ties['count_lower'][slot],kept_upper=ties['count_upper'][slot],
                                   exact_self=src['exact_self'][slot],same_slot=src['same_slot'][slot],
                                   largest_source=src['largest_source_slot'][slot],largest_source_share=src['largest_source_share'][slot]))
        for pi,s in enumerate(PREDICTOR_STAGES):
            cm,ct=mask(pc[ci,pi]); summaries[c][s+'_context_queries_only']=dict(ratios=ratios(pc[ci,pi]),counts=cm.sum(-1).tolist(),ties=ct)
            pack[c+'__'+s+'__context_queries_only__score']=pc[ci,pi]; pack[c+'__'+s+'__context_queries_only__mask']=cm
    if not a.allow_partial:
        assert summaries['continuous']['pred00']['counts'][:2]==[144,139]
    def qualifies(z):
        if len(z['halves'])!=2: return False
        halves_pass=all(h['ratios']['oldest_pair']>1 and h['oldest_pair_keep_mean']>64 for h in z['halves'].values())
        return (z['ratios']['oldest_pair']>1 and z['oldest_pair_keep_mean']>64 and halves_pass and
                z['individual']['paired_oldest_minus_middle_mass']['bootstrap95'][0]>0 and
                z['individual']['paired_oldest_minus_middle_keep']['bootstrap95'][0]>0)
    eligible=[s for s in STAGES if qualifies(summaries['continuous'][s])]
    onset=dict(criterion=man['onset_rule'],first_observed_stage=eligible[0] if eligible else None,
               qualifying_stages=eligible,all_stage_pass={s:s in eligible for s in STAGES},
               caveat='Observed attention-readout onset in execution order; not the causal layer that created its input representation.')
    norm_keys=sorted(norms[0])
    norm_avg={k:np.stack([z[k] for z in norms]).mean(0) for k in norm_keys}
    pack.update({'norm__'+k:v for k,v in norm_avg.items()})
    np.savez_compressed(a.out/'pooled_scores_masks_flows.npz',**pack)
    np.savez_compressed(a.out/'individual_scores.npz',indices=indices,scores=scores,predictor_context_query_scores=ctxscores)
    for name,records in [('stage_slot_metrics.csv',tables),('individual_stage_metrics.csv',individual)]:
        with (a.out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    plt.rcParams.update({'font.size':10}); c='continuous'; x=np.arange(36)
    mass=pooled[0].mean(-1)/pooled[0].mean((-1,-2))[:,None]
    counts=np.asarray([summaries[c][s]['counts'] for s in STAGES])
    fig,ax=plt.subplots(figsize=(20,11),constrained_layout=True)
    im=ax.imshow(mass,aspect='auto',origin='upper',cmap='coolwarm',vmin=.75,vmax=1.25); fig.colorbar(im,ax=ax,label='Received score / stage context-key mean')
    ax.axhline(23.5,color='black',lw=2); ax.set(yticks=x,yticklabels=STAGES,xticks=np.arange(0,64,4),xlabel='Context key temporal slot0–63',title=f'Original continuous16s | pooled{n} | layerwise received attention')
    fig.savefig(a.out/'layer_slot_received_mass_heatmap.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(20,11),constrained_layout=True)
    im=ax.imshow(counts,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=256); fig.colorbar(im,ax=ax,label='Tokens retained in independently selected stage top4096')
    ax.axhline(23.5,color='white',lw=2); ax.set(yticks=x,yticklabels=STAGES,xticks=np.arange(0,64,4),xlabel='Context key temporal slot0–63',title=f'Original continuous16s | pooled{n} | diagnostic K4096 per stage; uniform64')
    fig.savefig(a.out/'layer_slot_top4096_heatmap.png',dpi=170); plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(17,11),sharex=True,constrained_layout=True)
    for field,style in [('slot0','-'),('slot1','--'),('middle16_48','-.'),('remaining2_64',':')]: axs[0].plot(x,[summaries[c][s]['ratios'][field] for s in STAGES],style,label=field)
    axs[0].axhline(1,color='gray'); axs[0].set(ylabel='Received score / stage mean',title='Layerwise regions; encoder/predictor raw masses are not directly compared'); axs[0].legend(ncol=4)
    axs[1].plot(x,counts[:,0],label='slot0'); axs[1].plot(x,counts[:,1],label='slot1'); axs[1].plot(x,counts[:,16:48].mean(1),label='middle16:48'); axs[1].plot(x,counts[:,2:].mean(1),label='remaining2:64')
    axs[1].axhline(64,color='gray'); axs[1].set(ylabel='Mean retained tokens/slot'); axs[1].legend(ncol=4)
    first=onset['first_observed_stage']
    for ax in axs:
        ax.axvline(23.5,color='black',lw=2)
        if first is not None: ax.axvline(STAGES.index(first),color='red',ls='--',label='first observed '+first)
    axs[1].set(xticks=x,xticklabels=STAGES); axs[1].tick_params(axis='x',rotation=60)
    fig.savefig(a.out/'oldest_middle_remaining_and_onset.png',dpi=170); plt.close(fig)
    selected=['enc00','enc11','enc23','pred00','pred05','pred11']
    fig,axs=plt.subplots(2,1,figsize=(17,10),sharex=True,constrained_layout=True)
    for s in selected:
        si=STAGES.index(s); axs[0].plot(np.arange(64),mass[si],label=s); axs[1].plot(np.arange(64),counts[si],label=s)
    axs[0].axhline(1,color='gray'); axs[0].set(ylabel='Received score / stage mean',title='Prespecified full64 profiles: early/late encoder and predictor stages')
    axs[1].axhline(64,color='gray'); axs[1].set(xlabel='Context key temporal slot0–63',ylabel='Retained tokens',xticks=np.arange(0,64,4))
    for ax in axs: ax.legend(ncol=6)
    fig.savefig(a.out/'selected_full64_profiles.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(17,7),constrained_layout=True)
    for hv in half_values: ax.plot(x,[summaries[c][s]['halves'][str(hv)]['ratios']['oldest_pair'] for s in STAGES],label=f'half{hv}')
    med=np.asarray([summaries[c][s]['individual']['oldest_pair_ratio_quantiles'] for s in STAGES]); ax.fill_between(x,med[:,1],med[:,3],alpha=.15,label='individual10–90%'); ax.plot(x,med[:,2],':',label='individual median')
    ax.axhline(1,color='gray'); ax.axvline(23.5,color='black',lw=2); ax.set(ylabel='Oldest-pair mass / uniform',xticks=x,xticklabels=STAGES,title='Frozen halves and receiver distribution'); ax.tick_params(axis='x',rotation=60); ax.legend()
    fig.savefig(a.out/'half_and_individual_stability.png',dpi=170); plt.close(fig)
    audit.update(exact_coverage=n==86,measured_stage_condition_rows=n*36,distinct_indices=len(set(indices.tolist())),
                 legacy_max_relative_l2=max(z['relative_l2'] for z in legacy),legacy_min_jaccard=min(z['top4096_jaccard'] for z in legacy),
                 pooled_legacy_counts_checked=not a.allow_partial,mask_all4096=True,column_normalized_flow=True)
    summary=dict(protocol=PROTOCOL,run_status='completed',n=n,partial=a.allow_partial,stages=STAGES,conditions=CONDS,
                 indices=indices.tolist(),half_counts={str(v):int((halves==v).sum()) for v in half_values},onset=onset,
                 results=summaries,audit=audit,jobs=jobs,manifest_sha256=digest(a.manifest),analyzer_sha256=digest(__file__),
                 artifacts=artifacts,legacy_checks=legacy,
                 onset_caveat='Predefined descriptive criterion combines full pool, both halves, and paired receiver bootstrap intervals. Observed attention-readout onset is not a causal change point.',
                 cross_module_caveat='Encoder stages use64 context query slots; predictor stages use64 context plus one target-mask query slot. Compare normalized shape and criterion, not absolute raw mass across the boundary.',
                 score_convention='Native GPU FP32 head-sum independently reconstructed on CPU, FP32 pooled mean, stable descending top4096 with ascending original index ties; context keys only.')
    js(a.out/'summary.json',summary)
    print('ONSET_CPU_AUDIT',json.dumps(dict(n=n,audit=audit,onset=onset)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--allow-partial',action='store_true'); main(p.parse_args())
