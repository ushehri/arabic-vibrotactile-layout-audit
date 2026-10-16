"""5.5 revision: spatial correspondence score (SCS) over all motor assignments, cue-distinguishability filter,
anatomical plausibility, baselines, leave-one-term-out, discriminability + Pareto, learner bootstrap.
ض primary representation = time-weighted centroid of the Push cue (M2 400 ms + overlap 450 ms; M1 overlap 450 + 400 ms) = midpoint 6.25 cm;
for every other letter, |x-6.25| equals the mean of its distances to the two endpoints. Sensitivity: start (11.0) / end (1.5)."""
import json, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
exec(open('targets.py').read())
OUT='results/s5.5-rev/'
MOT=np.array([1.5,11.0,15.0,18.0,20.5,23.0])            # M1..M6
PAT={'ت':'Tap','ر':'Trill','ض':'Hold*','ي':'Tap','ك':'Tap','ق':'Tap','خ':'Hold','غ':'Hold'}   # Hold* = redesign pattern for single-location ض (Hold)
PATG={k:('Hold' if v.startswith('Hold') else v) for k,v in PAT.items()}
PLACE=np.array([PL[c] for c in T])
CLASS=np.array([0,0,1,2,3,4,5,5])
# B2 uses the final full-precision EPG extraction (Table 5 / Table S2): pooled mean contact position per letter from s51_rev_epg_table.py.
# (Earlier versions hard-coded 3-decimal means from a previous extraction; replaced 16 Oct 2026.)
EPG=pd.read_csv('results/s5.1-rev/s51r_epg_table.csv').set_index('letter').mean_pos.reindex(T).to_numpy(float)
assert not np.isnan(EPG).any()
zt={k:z(v) for k,v in TG.items()}
W={'primary':(0.6,0.3,0.1),'drop_artic':(0,0.75,0.25),'drop_conf':(0.6/0.7,0,0.1/0.7),'drop_acou':(0.6/0.9,0.3/0.9,0)}
CP=[(I[a],I[b]) for a,b in itertools.combinations(T,2)]
Sconf=CONF[iu]; cpairs=np.where(Sconf>0)[0]; wS=Sconf[cpairs]
patdiff=np.array([PATG[T[i]]!=PATG[T[j]] for i,j in CP],float)
def score_pos(P):  # P: (n,8) cm -> dict of rhos
    D=qz(np.abs(P[:,iu[0]]-P[:,iu[1]])); R=rankdata(D,axis=1); R=R-R.mean(1,keepdims=True); s=np.sqrt((R**2).sum(1)); s[s==0]=np.nan; Zd=R/s[:,None]
    out={k:Zd@zt[k] for k in zt}; out['D']=D; return out
def scs(r,w): return w[0]*r['blend']+w[1]*r['confusion']+w[2]*r['mfcc']
def ss(D): return (D[:,cpairs]*wS).sum(1)/wS.sum()
# ---- enumerate all 6^8 assignments in chunks ----
N=6**8; idx=np.arange(N); A=np.stack([(idx//6**k)%6 for k in range(8)][::-1],1).astype(np.int8)  # col order = T
pats=np.array([['Tap','Trill','Hold','Tap','Tap','Tap','Hold','Hold']])
# distinguishable: no two letters share motor AND pattern
dist_ok=np.ones(N,bool)
for i,j in itertools.combinations(range(8),2):
    if PATG[T[i]]==PATG[T[j]]: dist_ok&=A[:,i]!=A[:,j]
P=MOT[A]
def plausible(P,code=PLACE):
    ok=np.ones(len(P),bool)
    for i,j in itertools.combinations(range(8),2):
        if code[i]<code[j]: ok&=P[:,i]<=P[:,j]
        elif code[i]>code[j]: ok&=P[:,i]>=P[:,j]
    return ok
plaus=plausible(P)
res={k:np.empty(N) for k in ['blend','panphon','place','confusion','mfcc']}; SSv=np.empty(N)
for s in range(0,N,200000):
    r=score_pos(P[s:s+200000]); 
    for k in res: res[k][s:s+200000]=r[k]
    SSv[s:s+200000]=ss(r['D'])
valid=~np.isnan(res['blend'])
SC={w:scs(res,W[w]) for w in W}
# ---- named layouts ----
def lay(d): return np.array([d[c] for c in T],float)
NAMED={
 'current (ض midpoint, primary)':lay({'ت':1.5,'ر':1.5,'ض':6.25,'ي':11,'ك':15,'ق':18,'خ':20.5,'غ':23}),
 'current (ض start, M2)':lay({'ت':1.5,'ر':1.5,'ض':11,'ي':11,'ك':15,'ق':18,'خ':20.5,'غ':23}),
 'current (ض end, M1)':lay({'ت':1.5,'ر':1.5,'ض':1.5,'ي':11,'ك':15,'ق':18,'خ':20.5,'غ':23}),
 'redesign RD1: ض at tip (M1, Hold)':lay({'ت':1.5,'ر':1.5,'ض':1.5,'ي':11,'ك':15,'ق':18,'خ':20.5,'غ':23}),
 'baseline B1: KAPD place code, linear':1.5+PLACE/3*21.5,
 'baseline B2: KAPD EPG mean contact, linear':1.5+(EPG-EPG.min())/(EPG.max()-EPG.min())*21.5,
 'baseline B3: classical code, linear':1.5+CLASS/5*21.5,
}
rn=score_pos(np.vstack(list(NAMED.values())))
sets={'all assignments':valid,'collision-free':valid&dist_ok,'collision-free + place-order-preserving':valid&dist_ok&plaus}
rows=[]
for ni,(nm,pos) in enumerate(NAMED.items()):
    row=dict(layout=nm,positions=' '.join(f'{c}{p:g}' for c,p in zip(T,pos)),**{k:round(float(rn[k][ni]),3) for k in res},
             SS_cm=round(float(ss(rn['D'][ni:ni+1])[0]),2),SS_cm_full=round(float(ss(rn['D'][ni:ni+1])[0]),4),PS=round(float((patdiff[cpairs]*wS).sum()/wS.sum()),2))
    for w in W:
        v=scs({k:rn[k][ni:ni+1] for k in res},W[w])[0]; row['SCS_'+w]=round(float(v),3)
        for sn,m in sets.items():
            row[f'rank_{w}|{sn}']=int((SC[w][m]>v+1e-9).sum()+1)
    rows.append(row)
named=pd.DataFrame(rows); named.to_csv(OUT+'s55r_named_layouts.csv',index=False)
def ostr(pos):
    return ' '.join(f'M{int(np.where(MOT==s)[0][0])+1}:'+'/'.join(T[i] for i in range(8) if pos[i]==s) for s in sorted(set(pos)))
summary={'n':{k:int(m.sum()) for k,m in sets.items()},'random_mean_SCS':{sn:float(np.nanmean(SC['primary'][m])) for sn,m in sets.items()}}
top=[]
for w in W:
    for sn,m in sets.items():
        o=np.where(m)[0][np.argsort(-SC[w][m],kind='stable')[:10]]
        for r_,j in enumerate(o): top.append(dict(weights=w,set=sn,rank=r_+1,SCS=round(float(SC[w][j]),3),layout=ostr(P[j]),SS_cm=round(float(SSv[j]),2),
                                                   place_order_preserving=bool(plaus[j]),collision_free=bool(dist_ok[j])))
pd.DataFrame(top).to_csv(OUT+'s55r_top10.csv',index=False)
# ---- Pareto (SCS primary vs SS) within distinguishable+plausible ----
m=sets['collision-free + place-order-preserving']; ids=np.where(m)[0]
x=SC['primary'][ids]; y=SSv[ids]; o=np.lexsort((-y,-x)); front=[]; best=-1
for k in o:
    if y[k]>best+1e-9: front.append(ids[k]); best=y[k]
par=pd.DataFrame([dict(SCS=round(float(SC['primary'][j]),3),SS_cm=round(float(SSv[j]),2),layout=ostr(P[j])) for j in front]); par.to_csv(OUT+'s55r_pareto.csv',index=False)
np.savez_compressed(OUT+'s55r_scatter.npz',SCS=SC['primary'][ids],SS=SSv[ids])
# ---- learner bootstrap (conditional on learner resampling only) ----
cur=NAMED['current (ض midpoint, primary)']; rd=NAMED['redesign RD1: ض at tip (M1, Hold)']
Zm=None; bc=[];br=[]
Dm=qz(np.abs(P[ids][:,iu[0]]-P[ids][:,iu[1]])); Rm=rankdata(Dm,axis=1); Rm=Rm-Rm.mean(1,keepdims=True); Zm=Rm/np.sqrt((Rm**2).sum(1))[:,None]
def zrow(pos):
    d=qz(np.abs(pos[iu[0]]-pos[iu[1]])); r=rankdata(d); r=r-r.mean(); return r/np.sqrt((r**2).sum())
zc,zr=zrow(cur),zrow(rd)
for _ in range(NBOOT):
    smp=rng.choice(len(SPK),len(SPK)); Nn=sum(pre[SPK[s]][0] for s in smp); C=sum(pre[SPK[s]][1] for s in smp)
    cv=z(np.nan_to_num(-sym(Nn,C)[iu])); sb=0.6*(Zm@zt['blend'])+0.3*(Zm@cv)+0.1*(Zm@zt['mfcc'])
    f=lambda zz:0.6*zz@zt['blend']+0.3*zz@cv+0.1*zz@zt['mfcc']
    bc.append(int((sb>f(zc)+1e-9).sum()+1)); br.append(int((sb>f(zr)+1e-9).sum()+1))
summary['bootstrap_dist_plaus']=dict(n_set=int(len(ids)),current_rank_median=float(np.median(bc)),current_ci95=[float(np.percentile(bc,2.5)),float(np.percentile(bc,97.5))],
    RD1_rank_median=float(np.median(br)),RD1_ci95=[float(np.percentile(br,2.5)),float(np.percentile(br,97.5))],note='conditional on learner resampling only; blend/MFCC fixed; words not resampled')
summary['same_pattern_confused_SS']={nm:float(np.average([abs(pos[I[a]]-pos[I[b]]) for a,b in [('ك','ق'),('خ','غ')]],weights=[Sconf[CP.index((I['ك'],I['ق']))],Sconf[CP.index((I['خ'],I['غ']))]])) for nm,pos in NAMED.items()}
summary['confused_pairs']=[dict(pair=f'{T[CP[k][0]]}–{T[CP[k][1]]}',S=round(float(Sconf[k]),4),pattern_differs=bool(patdiff[k])) for k in cpairs]
json.dump(summary,open(OUT+'s55r_summary.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(summary,ensure_ascii=False,indent=1)); print(named.T.to_string())
print(pd.DataFrame(top).query("rank<=3").to_string())
print(par.to_string())
