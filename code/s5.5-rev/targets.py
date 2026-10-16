"""S5.5 full run: layout optimisation 'mirror the tongue'. Seeded (55), deterministic."""
import json, itertools, numpy as np, pandas as pd
from scipy.stats import rankdata
# Tie handling: distances/targets are quantised to 1e-9 before ranking so that floating-point noise
# (e.g. 1.5+k/3*21.5 or averaged rank-normalised codes) cannot break mathematical ties.
# 1e-9 is ~7 orders of magnitude below the smallest real difference in any input (cm, rates, MFCC units).
QTOL=9
def qz(x): return np.round(np.asarray(x,float),QTOL)
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
# Exclude the erroneous خ token (canon خ in الرحيم; realised ح), as in Section 4.3 / s54_rev.py -> 205 خ tokens, 9766 total (fixed 16 Oct 2026; previously 206 خ, 9767 total).
x=x[~((x.canon=='خ')&(x.word=='الرحيم'))].copy()
assert (x.canon=='خ').sum()==205 and len(x)==9766, ((x.canon=='خ').sum(),len(x))
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
import os
if os.path.exists('results/s5.4-rev/s54r_pairs.csv'):   # cross-check against the corrected Section 4.3 pair table (from s54_rev.py)
    _R=pd.read_csv('results/s5.4-rev/s54r_pairs.csv'); _S={tuple(r.pair.split('–')):r.S_pct for _,r in _R.iterrows()}
    for (a,b),v in _S.items(): assert abs(CONF[I[a],I[b]]*100-v)<1e-6, ('confusion mismatch vs s54r_pairs',a,b)
    assert sum(1 for i in CONF[iu] if i>0)==len(_S)
rn=lambda v:(rankdata(qz(v))-1)/(len(v)-1)   # rank-normalise to [0,1]
BLEND=(rn(FEAT[iu])+rn(PLACE[iu]))/2
TG={'blend':BLEND,'panphon':FEAT[iu],'place':PLACE[iu],'confusion':-CONF[iu],'mfcc':MF[iu]}
def z(v): v=rankdata(qz(v)); v=v-v.mean(); s=np.sqrt((v**2).sum()); return v/s if s>0 else v*0
