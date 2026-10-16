"""Diagnose the Table S7 reproduction check (recomputed vs stored pooled MFCC matrices).
Step A: rebuild each source's pooled MFCC P matrix from the STORED token features (feats_mfcc*.pkl, float32 token means, as analysis_v2.py)
        with a freshly seeded generator per source (rng(0), as s52_rev_window.py) -> compare with the recomputed 'orig' matrix of s52r_window_mats.pkl.
Step B: replay the exact sequence of calls that analysis_v2.main() makes on its single shared generator rng(0)
        (source order l2ksu, halabi, cv, everyayah, iqra; P point estimate, 500 cluster-bootstrap replicates, 30-token DTW draw, W bootstrap)
        WITHOUT computing distances, and rebuild each source's P point estimate from the generator state it actually had -> compare with stored
        pooled_boots.pkl.
Step C: recompute token MFCCs from audio exactly as s52_rev_window.py ('orig' window), once with pandas' default CSV float parser (as in
        s52_rev_window.py) and once with float_precision='round_trip'; count tokens whose frame count differs from the stored n_mfcc_frames,
        and compare the fresh-seed matrices with Step A. Pooled (ALL) Spearman r with sleeve distance (ض 6.25 cm) is reported for each variant.
No new analysis; diagnostic only."""
import pickle, glob, os, json, numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
LET=['t','r','dd','j','k','q','x','gh']; IU=np.triu_indices(8,1); F1='data/full52'; V='data/full52-v2'
NB_P,NB_W=500,1000
def euc(X,li,G):
    M=np.full((8,8),np.nan)
    for i in range(8):
        for j in range(i,8):
            D=cdist(X[G[i]],X[G[j]]); M[i,j]=M[j,i]=D[np.triu_indices(len(G[i]),1)].mean() if i==j else D.mean()
    return M
def groups(li,cap,r):
    G=[np.where(li==i)[0] for i in range(8)]; return [g if len(g)<=cap else r.choice(g,cap,replace=False) for g in G]
# load sources exactly as analysis_v2.load() (MFCC only)
src={}
t=pd.read_csv(f'{F1}/tokens.csv',keep_default_na=False); mf=pickle.load(open(f'{F1}/feats_mfcc.pkl','rb'))
for c in ['l2ksu','halabi']:
    ix=np.where((t.corpus==c)&(t.n_mfcc_frames.astype(int)>=3))[0]; d=t.iloc[ix].copy(); d['unit']=d.speaker
    src[c]=dict(tok=d.reset_index(drop=True),mm=np.array([mf[i].mean(0) for i in ix],dtype=np.float32))
os.chdir(V)
for f in sorted(glob.glob('tokens_*.csv')):
    c=f[7:-4]; d=pd.read_csv(f,keep_default_na=False,dtype={'unit':str}); m=pickle.load(open(f'feats_mfcc_{c}.pkl','rb'))
    ix=np.where(d.n_mfcc_frames.astype(int)>=3)[0]; src[c]=dict(tok=d.iloc[ix].reset_index(drop=True),mm=np.array([m[i].mean(0) for i in ix],dtype=np.float32))
for c,s in src.items():
    s['li']=s['tok'].letter.map({l:i for i,l in enumerate(LET)}).values; u=sorted(s['tok'].unit.unique()); um={x:k for k,x in enumerate(u)}
    s['ui']=s['tok'].unit.map(um).values; s['U']=len(u); N=np.zeros((len(u),8)); np.add.at(N,(s['ui'],s['li']),1); s['N']=N
order=list(src); print('analysis_v2 source order:',order)
stored=pickle.load(open(f'{V}/pooled_boots.pkl','rb')); W=pickle.load(open('results/s5.2-rev/s52r_window_mats.pkl','rb'))
out={'source_order_analysis_v2':order,'letters_over_300_tokens':{c:[LET[i] for i in range(8) if (s['li']==i).sum()>300] for c,s in src.items()},'n_tokens':{c:{LET[i]:int((s['li']==i).sum()) for i in range(8)} for c,s in src.items()}}
cmp=lambda A,B:dict(spearman_r=float(spearmanr(A[IU],B[IU])[0]),max_abs_diff_all_cells=float(np.nanmax(np.abs(A-B))))
# Step A
out['A_fresh_seed_stored_features_vs_recomputed_orig']={c:cmp(euc(s['mm'],s['li'],groups(s['li'],300,np.random.default_rng(0))),W[c]['orig']) for c,s in src.items() if c in W}
# Step B: replay shared generator
rng=np.random.default_rng(0); rep={}
for c in order:
    s=src[c]; li=s['li']; U=s['U']
    G0=groups(li,300,rng); rep[c]=euc(s['mm'],li,G0)                       # P0 = mats_P(s, ones)
    if U>1:
        for b in range(NB_P):
            w=np.bincount(rng.integers(0,U,U),minlength=U).astype(float)
            ix=np.where(w[s['ui']]>0)[0]; ix=np.repeat(ix,w[s['ui'][ix]].astype(int)); groups(li[ix],300,rng)
    for i in range(8): rng.permutation(np.where(li==i)[0])                     # DTW 30-token draw
    if c not in ('iqra','halabi'):
        full=np.where((s['N']>=1).all(1))[0]
        if len(full)>=2:
            for k in range(26):                                                # mfcc_euclid + 25 SSL layers
                for _ in range(NB_W): rng.integers(0,len(full),len(full))
    print(c,'replayed',flush=True)
out['B_replayed_shared_generator_vs_stored']={c:cmp(rep[c],stored[c][0]['mfcc_euclid']) for c in order}
out['stored_vs_recomputed_orig']={c:cmp(stored[c][0]['mfcc_euclid'],W[c]['orig']) for c in W}
# Step C
import librosa, collections
SR=16000; HOP=0.010
def mfcc_utt(y):
    m=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13,n_fft=400,win_length=400,hop_length=160,center=True)
    return np.vstack([m,librosa.feature.delta(m,order=1),librosa.feature.delta(m,order=2)]).T
def fsl(a,b,n): return slice(max(0,int(np.ceil(a/HOP))),min(n,int(np.floor(b/HOP))+1))
def recompute(c,fp):
    if c=='l2ksu':
        d=pd.read_csv(f'{F1}/tokens.csv',keep_default_na=False,float_precision=fp); d=d[(d.corpus=='l2ksu')&(d.n_mfcc_frames.astype(int)>=3)].copy(); d['unit']=d.speaker
    else:
        d=pd.read_csv(f'{V}/tokens_{c}.csv',keep_default_na=False,dtype={'unit':str},float_precision=fp); d=d[d.n_mfcc_frames.astype(int)>=3].copy()
    d=d.reset_index(drop=True); raw=[None]*len(d); uF=collections.defaultdict(list)
    for wav,idx in d.groupby('wav').groups.items():
        y,_=librosa.load(wav,sr=SR); M=mfcc_utt(y); uF[d.loc[idx[0],'unit']].append(M)
        for i in idx: raw[i]=M[fsl(float(d.at[i,'start']),float(d.at[i,'end']),len(M))]
    stt={u:(np.vstack(F).mean(0),np.vstack(F).std(0)+1e-8) for u,F in uF.items()}
    X=np.array([((raw[i]-stt[d.at[i,'unit']][0])/stt[d.at[i,'unit']][1]).mean(0) for i in range(len(d))])
    li=d.letter.map({l:i for i,l in enumerate(LET)}).values
    return euc(X,li,groups(li,300,np.random.default_rng(0))), int((np.array([len(r) for r in raw])!=d.n_mfcc_frames.astype(int).values).sum()), len(d)
POS=np.array([1.5,1.5,6.25,11,15,18,20.5,23]); SL=np.abs(POS[:,None]-POS[None,:]); four=['l2ksu','iqra','everyayah','cv']
rS=lambda Ms:float(spearmanr(np.mean(Ms,0)[IU],SL[IU])[0])
A={c:euc(src[c]['mm'],src[c]['li'],groups(src[c]['li'],300,np.random.default_rng(0))) for c in four}
out['C']={}
Cm={}
for fp in [None,'round_trip']:
    k='default_parser' if fp is None else 'round_trip_parser'; out['C'][k]={}; Cm[k]={}
    for c in four:
        M,nmis,n=recompute(c,fp); Cm[k][c]=M; out['C'][k][c]=dict(tokens=n,tokens_frame_count_differs_from_stored=nmis,vs_stepA=cmp(M,A[c]))
        print(k,c,out['C'][k][c],flush=True)
out['ALL_MFCC_r_sleeve_6.25']={'stored (primary, pooled_boots.pkl)':rS([stored[c][0]['mfcc_euclid'] for c in four]),
    'recomputed orig (s52r_window_mats.pkl)':rS([W[c]['orig'] for c in four]),
    'recomputed now, default parser, fresh seed':rS([Cm['default_parser'][c] for c in four]),
    'recomputed now, round-trip parser, fresh seed':rS([Cm['round_trip_parser'][c] for c in four]),
    'stored features, fresh seed (Step A)':rS([A[c] for c in four]),
    'stored features, replayed shared generator (Step B)':rS([rep[c] for c in four])}
json.dump(out,open('results/s5.2-rev/s52r_repro_check.json','w'),indent=1); print(json.dumps(out,indent=1))
