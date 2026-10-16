"""5.2 revision: consonant-window MFCC sensitivity. Recomputes 39-d MFCC (same settings as full52 features.mfcc_utt; per-unit CMVN)
for token windows: 'orig' = [start,end] as stored (reproduction check), 'c60'/'c100' = [start, min(end, seg_start+cap+PAD)] i.e. first 60/100 ms from the
aligned onset (less of the following vowel). Pooled P matrix (mean pairwise Euclidean, <=300 tokens/letter, seed 0), ALL = mean over sources."""
import numpy as np, pandas as pd, librosa, collections, json, pickle
from scipy.spatial.distance import cdist
SR=16000; HOP=0.010; PAD=0.010
LET=['t','r','dd','j','k','q','x','gh']
def mfcc_utt(y):
    m=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13,n_fft=400,win_length=400,hop_length=160,center=True)
    return np.vstack([m,librosa.feature.delta(m,order=1),librosa.feature.delta(m,order=2)]).T
def fs(a,b,n): return slice(max(0,int(np.ceil(a/HOP))),min(n,int(np.floor(b/HOP))+1))
V='data/full52-v2/'
t1=pd.read_csv('data/full52/tokens.csv',keep_default_na=False); t1=t1[(t1.corpus=='l2ksu')&(t1.n_mfcc_frames.astype(int)>=3)].copy(); t1['unit']=t1.speaker; t1['source']='l2ksu'
src={'l2ksu':t1}
for c in ['iqra','everyayah','cv']:
    d=pd.read_csv(V+f'tokens_{c}.csv',keep_default_na=False,dtype={'unit':str}); src[c]=d[d.n_mfcc_frames.astype(int)>=3].copy()
WIN={'orig':None,'c60':0.060,'c100':0.100}
res={}
for c,d in src.items():
    d=d.reset_index(drop=True); raw={w:[None]*len(d) for w in WIN}; unitF=collections.defaultdict(list)
    for wav,idx in d.groupby('wav').groups.items():
        y,_=librosa.load(wav,sr=SR); M=mfcc_utt(y); u=d.loc[idx[0],'unit']; unitF[u].append(M)
        for i in idx:
            st,en,ss=float(d.at[i,'start']),float(d.at[i,'end']),float(d.at[i,'seg_start'])
            for w,cap in WIN.items():
                e=en if cap is None else min(en,ss+cap+PAD); raw[w][i]=M[fs(st,e,len(M))]
    stt={u:(np.vstack(F).mean(0),np.vstack(F).std(0)+1e-8) for u,F in unitF.items()}
    li=d.letter.map({l:k for k,l in enumerate(LET)}).values
    res[c]={}
    for w in WIN:
        X=np.array([((raw[w][i]-stt[d.at[i,'unit']][0])/stt[d.at[i,'unit']][1]).mean(0) if len(raw[w][i]) else np.full(39,np.nan) for i in range(len(d))])
        ok=~np.isnan(X).any(1); rng=np.random.default_rng(0); M=np.full((8,8),np.nan)
        G=[np.where((li==k)&ok)[0] for k in range(8)]; G=[g if len(g)<=300 else rng.choice(g,300,replace=False) for g in G]
        for i in range(8):
            for j in range(i,8):
                D=cdist(X[G[i]],X[G[j]]); M[i,j]=M[j,i]=D[np.triu_indices(len(G[i]),1)].mean() if i==j else D.mean()
        res[c][w]=M
    print(c,len(d),flush=True)
pickle.dump(res,open('results/s5.2-rev/s52r_window_mats.pkl','wb')); print('DONE')
