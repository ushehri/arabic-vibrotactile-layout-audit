"""5.2 revision: Mantel (Spearman, 9,999 perms, seed 0) of pooled acoustic matrices vs sleeve for 3 ض codings
(midpoint 6.25 primary; start 11.0; end 1.5), for ALL (4 sources) and ALL without L2-KSU (layer 23 was pre-chosen on L2-KSU).
CIs from stored per-source cluster bootstrap matrices (pooled_boots.pkl, 500 reps). Two-sided p added."""
import sys,pickle,json,numpy as np
from scipy.stats import spearmanr
LET=['t','r','dd','j','k','q','x','gh']; POS={'t':1.5,'r':1.5,'dd':11.0,'j':11.0,'k':15.0,'q':18.0,'x':20.5,'gh':23.0}  # as full52/analysis.py
IU=np.triu_indices(8,1)
def sleeve(pos): p=np.array([pos[l] for l in LET]); return np.abs(p[:,None]-p[None,:])
B=pickle.load(open('data/full52-v2/pooled_boots.pkl','rb'))
rng0=lambda:np.random.default_rng(0)
def mantel2(A,S,n=9999):
    r0=spearmanr(A[IU],S[IU])[0]; g=rng0(); pp=[]
    for _ in range(n):
        p=g.permutation(8); pp.append(spearmanr(A[IU],S[np.ix_(p,p)][IU])[0])
    pp=np.array(pp); return float(r0),float((1+(pp>=r0).sum())/(n+1)),float((1+(np.abs(pp)>=abs(r0)).sum())/(n+1))
COD={'midpoint 6.25 (primary)':6.25,'start 11.0':11.0,'end 1.5':1.5}
POOLS={'ALL (4 sources)':['l2ksu','iqra','everyayah','cv'],'ALL without L2-KSU':['iqra','everyayah','cv']}
meas=['mfcc_euclid','ssl_L23']+[f'ssl_L{L:02d}' for L in range(25) if L!=23]
out=[]
for pn,src in POOLS.items():
    for m in meas:
        M=np.mean([B[c][0][m] for c in src],0); Bb=np.mean([B[c][1][m] for c in src],0)
        for cn,dd in COD.items():
            S=sleeve(dict(POS,dd=dd))
            if m.startswith('ssl') and m!='ssl_L23':
                r=spearmanr(M[IU],S[IU])[0]; out.append(dict(pool=pn,measure=m,dad=cn,r=float(r))); continue
            r,p1,p2=mantel2(M,S); rb=[spearmanr(b[IU],S[IU])[0] for b in Bb]
            out.append(dict(pool=pn,measure=m,dad=cn,r=r,p_one_sided_pos=p1,p_two_sided=p2,ci=[float(np.nanpercentile(rb,2.5)),float(np.nanpercentile(rb,97.5))]))
import pandas as pd
df=pd.DataFrame(out); df.to_csv('results/s5.2-rev/s52r_mantel.csv',index=False)
mx=df[df.measure.str.startswith('ssl')].groupby(['pool','dad']).r.agg(['max','idxmax']); mx['layer']=df.loc[mx['idxmax'],'measure'].values
print(df[~df.p_two_sided.isna()].to_string()); print(mx.to_string())
