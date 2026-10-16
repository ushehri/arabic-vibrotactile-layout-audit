"""S5.5 full run: layout optimisation 'mirror the tongue'. Seeded (55), deterministic."""
import json, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata
B='data/'
rng=np.random.default_rng(55); NBOOT=1000
T=['ت','ر','ض','ي','ك','ق','خ','غ']; I={c:i for i,c in enumerate(T)}
FRONT={'ت','ر','ض'}; UVUL={'ق','خ','غ'}
SLOTS={'main':[1.5,1.5,11.0,11.0,15.0,18.0,20.5,23.0],
       'dad_tip':[1.5,1.5,1.5,11.0,15.0,18.0,20.5,23.0]}   # one 11.0 slot moved to 1.5
CURRENT={'main':['ت','ر','ض','ي','ك','ق','خ','غ'],'dad_tip':['ت','ر','ض','ي','ك','ق','خ','غ']}
iu=np.triu_indices(8,1)
# ---- targets ----
P=pd.read_csv(B+'s54/s54_pairs.csv'); FEAT=np.zeros((8,8))
for _,r in P.iterrows():
    a,b=r.pair.split('–'); FEAT[I[a],I[b]]=FEAT[I[b],I[a]]=r.feature_dist
PL={'ت':0,'ر':0,'ض':0,'ي':1,'ك':2,'ق':3,'خ':3,'غ':3}
PLACE=np.array([[abs(PL[a]-PL[b]) for b in T] for a in T],float)
v2=json.load(open(B+'full52-v2/results_v2.json')); MF=np.array(v2['ALL']['mfcc_euclid']['mean'],float)
v2L=v2.get('letters') or v2['ALL'].get('letters')
if v2L is not None and list(v2L)!=T: MF=MF[np.ix_([list(v2L).index(c) for c in T],[list(v2L).index(c) for c in T])]
assert np.allclose(MF[iu],P.mfcc.values,atol=1e-4), 'MFCC mismatch vs s54_pairs'
tok=pd.read_csv(B+'s54/tokens.csv'); x=tok[(tok.group=='non-native_speakers')&tok.canon.isin(T)]
SPK=sorted(x.speaker.unique())
def mats(df):
    N=np.array([(df.canon==l).sum() for l in T],float); C=np.zeros((8,8))
    for i,a in enumerate(T):
        s=df[df.canon==a].real.value_counts()
        for j,b in enumerate(T): C[i,j]=s.get(b,0)
    return N,C
def sym(N,C):
    M=C.copy(); np.fill_diagonal(M,0); M=M/np.where(N[:,None]>0,N[:,None],np.nan); return (M+M.T)/2
pre={s:mats(x[x.speaker==s]) for s in SPK}
N0=sum(v[0] for v in pre.values()); C0=sum(v[1] for v in pre.values()); CONF=sym(N0,C0)
assert np.allclose(CONF[iu],P.sym_rate.values), 'confusion mismatch vs s54_pairs'
rn=lambda v:(rankdata(v)-1)/(len(v)-1)   # rank-normalise to [0,1]
BLEND=(rn(FEAT[iu])+rn(PLACE[iu]))/2
TG={'blend':BLEND,'panphon':FEAT[iu],'place':PLACE[iu],'confusion':-CONF[iu],'mfcc':MF[iu]}
def z(v): v=rankdata(v); v=v-v.mean(); s=np.sqrt((v**2).sum()); return v/s if s>0 else v*0
# ---- enumerate ----
def enum(slots):
    seen={}; 
    for perm in itertools.permutations(T):
        pos=np.zeros(8)
        for s,c in zip(slots,perm): pos[I[c]]=s
        key=tuple(pos)
        if key not in seen: seen[key]=perm
    POS=np.array(list(seen.keys())); D=np.abs(POS[:,:,None]-POS[:,None,:])[:,iu[0],iu[1]]
    Z=np.array([z(d) for d in D]); return POS,list(seen.values()),Z
def order_str(pos):  # letters sorted by slot, ties grouped by '/'
    out=[];
    for s in sorted(set(pos)): out.append('/'.join(sorted([T[i] for i in range(8) if pos[i]==s],key=T.index)))
    return ' | '.join(out)
def flags(pos):
    f=[]
    if any(pos[I[c]]>=18 for c in FRONT): f.append('FRONT_AT_ELBOW')
    sens=all(pos[I[c]]<=11 for c in FRONT) and all(pos[I[c]]>=15 for c in UVUL)
    return f,sens
W_GRID=[(a,c,round(1-a-c,2)) for a in (0.4,0.5,0.6,0.7,0.8,0.9) for c in (0.0,0.1,0.2,0.3,0.4,0.5) if 1-a-c>=-1e-9 and 1-a-c<=0.3+1e-9]
OUT={'meta':dict(slots=SLOTS,blend='mean of rank-normalised (0..1) PanPhon and place pair vectors',
     weights_primary=[0.6,0.3,0.1],weight_grid=W_GRID,nboot=NBOOT,seed=55,n_learners=len(SPK))}
tables={}
for var,slots in SLOTS.items():
    POS,PERMS,Z=enum(slots); n=len(POS)
    R={k:Z@z(t) for k,t in TG.items()}
    comb=lambda w:w[0]*R['blend']+w[1]*R['confusion']+w[2]*R['mfcc']
    R['combined']=comb((.6,.3,.1))
    curpos=np.zeros(8)
    for s,c in zip(slots,CURRENT[var]): curpos[I[c]]=s
    ci=int(np.where((POS==curpos).all(1))[0][0])
    FL=[flags(p) for p in POS]; sens_mask=np.array([f[1] for f in FL]); elbow=np.array(['FRONT_AT_ELBOW' in f[0] for f in FL])
    res={'n_distinct':n,'current':order_str(curpos),'n_sensible':int(sens_mask.sum()),'criteria':{}}
    top_rows=[]
    for k,v in R.items():
        cur=v[ci]; ge=int((v>=cur-1e-12).sum()); gt=int((v>cur+1e-12).sum())
        res['criteria'][k]=dict(score=float(cur),rank_best=gt+1,rank_worst=ge,p_ge=ge/n,percentile=float((v<=cur+1e-12).mean()),
            best=float(v.max()),n_tied_best=int((v>=v.max()-1e-12).sum()),
            best_sensible=float(v[sens_mask].max()),frac_top100_front_at_elbow=float(elbow[np.argsort(-v)[:100]].mean()))
        for r_,j in enumerate(np.argsort(-v,kind='stable')[:10]):
            top_rows.append(dict(variant=var,criterion=k,rank=r_+1,score=float(v[j]),layout=order_str(POS[j]),
                                 front_at_elbow=bool(elbow[j]),sensible=bool(sens_mask[j])))
    # weight stability
    stab=[];tops={}
    for w in W_GRID:
        c=comb(w); o=np.argsort(-c,kind='stable'); tops[w]=set(o[:10])
        so=[j for j in o if sens_mask[j]]
        stab.append(dict(variant=var,w_artic=w[0],w_conf=w[1],w_acou=round(w[2],2),current_rank=int((c>c[ci]+1e-12).sum()+1),
            top1=order_str(POS[o[0]]),top1_front_at_elbow=bool(elbow[o[0]]),top1_sensible=order_str(POS[so[0]]),current_p_ge=float((c>=c[ci]-1e-12).mean())))
    base=tops[(0.6,0.3,0.1)] if (0.6,0.3,0.1) in tops else None
    for s_,w in zip(stab,W_GRID): s_['top10_overlap_with_primary']=len(tops[w]&base)/10 if base else None
    # single swaps from current
    sw=[]
    key={tuple(p):j for j,p in enumerate(POS)}
    for a,b in itertools.combinations(T,2):
        p=curpos.copy(); p[I[a]],p[I[b]]=p[I[b]],p[I[a]]
        if (p==curpos).all(): continue
        j=key[tuple(p)]; d={k:float(R[k][j]-R[k][ci]) for k in R}
        sw.append(dict(variant=var,move=f'swap {a}<->{b}',layout=order_str(p),front_at_elbow=bool(elbow[j]),sensible=bool(sens_mask[j]),
                       **{'d_'+k:v for k,v in d.items()},combined=float(R['combined'][j])))
    sw=sorted(sw,key=lambda r:-r['d_combined'])
    # recommended: best combined among sensible
    so=np.argsort(-R['combined'],kind='stable'); rec=[j for j in so if sens_mask[j]][0]
    res['recommended']=dict(layout=order_str(POS[rec]),positions={T[i]:float(POS[rec][i]) for i in range(8)},
        **{k:float(R[k][rec]) for k in R},rank_combined_all=int((R['combined']>R['combined'][rec]+1e-12).sum()+1))
    # bootstrap over learners: recompute confusion, combined; track ranks
    pos_all=[]; bc=[];br=[]; top1=[]
    zb=z(BLEND); zm=z(MF[iu])
    for _ in range(NBOOT):
        smp=rng.choice(len(SPK),len(SPK)); N=sum(pre[SPK[s]][0] for s in smp); C=sum(pre[SPK[s]][1] for s in smp)
        cv=-sym(N,C)[iu]; cv=np.nan_to_num(cv)
        c=0.6*R['blend']+0.3*(Z@z(cv))+0.1*R['mfcc']
        bc.append(int((c>c[ci]+1e-12).sum()+1)); br.append(int((c>c[rec]+1e-12).sum()+1)); top1.append(int(np.argmax(c)))
    vc=pd.Series(top1).value_counts()
    res['bootstrap']=dict(current_rank_median=float(np.median(bc)),current_rank_ci95=[float(np.percentile(bc,2.5)),float(np.percentile(bc,97.5))],
        recommended_rank_median=float(np.median(br)),recommended_rank_ci95=[float(np.percentile(br,2.5)),float(np.percentile(br,97.5))],
        recommended_top10_frac=float(np.mean(np.array(br)<=10)),
        modal_top1=order_str(POS[vc.index[0]]),modal_top1_frac=float(vc.iloc[0]/NBOOT))
    OUT[var]=res
    tables.setdefault('top10',[]).extend(top_rows); tables.setdefault('weights',[]).extend(stab); tables.setdefault('swaps',[]).extend(sw)
    tables.setdefault('current',[]).extend([dict(variant=var,criterion=k,**v) for k,v in res['criteria'].items()])
    np.save(f'scores_{var}.npy',np.vstack([R[k] for k in R])); json.dump([order_str(p) for p in POS],open(f'layouts_{var}.json','w'),ensure_ascii=False)
json.dump(OUT,open('s55_results.json','w'),ensure_ascii=False,indent=1)
for k,v in tables.items(): pd.DataFrame(v).to_csv(f's55_{k}.csv',index=False)
print(json.dumps(OUT,ensure_ascii=False,indent=1)[:6000])
