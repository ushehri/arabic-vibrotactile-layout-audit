"""ALL recomputed with Iqra utterances that duplicate a used Common Voice clip (same normalised text, |Δdur|<0.05 s) removed.
Iqra matrices from stored features (MFCC: per-utterance CMVN, so unaffected by removal; SSL L23: exact mean pairwise cosine). Other sources: stored P0."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, results_dir
import numpy as np, pandas as pd, pickle, re, json
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
V=data('full52-v2')+'/'; LET=['t','r','dd','j','k','q','x','gh']; IU=np.triu_indices(8,1)
POS={'t':1.5,'r':1.5,'dd':11.0,'j':11.0,'k':15.0,'q':18.0,'x':20.5,'gh':23.0}
def sleeve(dd): p=np.array([dict(POS,dd=dd)[l] for l in LET]); return np.abs(p[:,None]-p[None,:])
exec(open(Path(__file__).resolve().parent/'s52_rev_overlap.py',encoding='utf-8').read().split("json.dump")[0].replace('print','0 and print'))
dup=set(j[(j.dur-j.dur_o).abs()<0.05].utt) if False else None
o=m[m.source=='cv']; jj=iq.merge(o[['n','dur']],on='n',suffixes=('','_o')); dup=set(jj[(jj.dur-jj.dur_o).abs()<0.05].utt)
d=pd.read_csv(V+'tokens_iqra.csv',keep_default_na=False); mf=pickle.load(open(V+'feats_mfcc_iqra.pkl','rb')); ssl=np.load(V+'feats_ssl_iqra.npy',mmap_mode='r')
ix=np.where((d.n_mfcc_frames.astype(int)>=3)&(~d.utt.isin(dup)))[0]; li=d.letter.map({l:k for k,l in enumerate(LET)}).values[ix]
X=np.array([mf[i].mean(0) for i in ix]); rng=np.random.default_rng(0); M=np.full((8,8),np.nan)
G=[np.where(li==k)[0] for k in range(8)]; G=[g if len(g)<=300 else rng.choice(g,300,replace=False) for g in G]
for i in range(8):
    for k in range(i,8):
        D=cdist(X[G[i]],X[G[k]]); M[i,k]=M[k,i]=D[np.triu_indices(len(G[i]),1)].mean() if i==k else D.mean()
E=np.asarray(ssl[ix,23],np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)
Ssum=np.stack([E[li==k].sum(0) for k in range(8)]); n=np.array([(li==k).sum() for k in range(8)],float)
C=1-(Ssum@Ssum.T)/np.outer(n,n)
B=pickle.load(open(V+'pooled_boots.pkl','rb'))
out={'n_dup_utts_removed':len(dup),'iqra_tokens_kept':int(len(ix))}
for meas,Mi in [('mfcc_euclid',M),('ssl_L23',C)]:
    ALL=np.mean([B['l2ksu'][0][meas],Mi,B['everyayah'][0][meas],B['cv'][0][meas]],0)
    NOL2=np.mean([Mi,B['everyayah'][0][meas],B['cv'][0][meas]],0)
    for dd in (6.25,11.0,1.5):
        out[f'{meas}|dd{dd}|ALL_dedup']=float(spearmanr(ALL[IU],sleeve(dd)[IU])[0]); out[f'{meas}|dd{dd}|noL2_dedup']=float(spearmanr(NOL2[IU],sleeve(dd)[IU])[0])
    out[f'{meas}|iqra_only_dedup_vs_full_rho']=float(spearmanr(Mi[IU],B['iqra'][0][meas][IU])[0])
json.dump(out,open(results_dir('s5.2-rev')+'s52r_dedup.json','w'),indent=1); print(json.dumps(out,indent=1))
