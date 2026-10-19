# 5.2 main table at ض midpoint (6.25 cm), two-sided Mantel (9,999 perms, seed 0); per source/estimator; P-matrix CIs from stored cluster bootstraps
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, results_dir
import numpy as np, pandas as pd, pickle
from scipy.stats import rankdata
V=data('full52-v2')+'/'; LET=['t','r','dd','j','k','q','x','gh']; iu=np.triu_indices(8,1)
q=lambda v:np.round(np.asarray(v,float),9)
p=np.array([1.5,1.5,6.25,11,15,18,20.5,23]); S=np.abs(p[:,None]-p[None,:])
M=pd.read_csv(V+'matrices_v2.csv'); B=pickle.load(open(V+'pooled_boots.pkl','rb'))
def mat(g):
    X=np.full((8,8),np.nan); I={l:i for i,l in enumerate(LET)}
    for r in g.itertuples(): X[I[r.a],I[r.b]]=r.d
    return X
def rr(A): return np.corrcoef(rankdata(q(A[iu])),rankdata(q(S[iu])))[0,1]
rng=np.random.default_rng(0); PERMS=[rng.permutation(8) for _ in range(9999)]
def mantel(A):
    r0=rr(A); a=rankdata(q(A[iu])); pp=np.array([np.corrcoef(a,rankdata(q(S[np.ix_(pm,pm)][iu])))[0,1] for pm in PERMS])
    return r0,(1+(np.abs(pp)>=abs(r0)-1e-12).sum())/10000
rows=[]; mats={}
for (src,est),g in M.groupby(['source','estimator'],sort=False):
    for meas in ['mfcc_euclid','mfcc_dtw_30tok','ssl_L23']:
        h=g[g.measure==meas]
        if not len(h): continue
        A=mat(h); mats[(src,est,meas)]=A; r,pv=mantel(A); ci=None
        if est=="P" and meas!="mfcc_dtw_30tok" and src in B and meas in B[src][1] and np.asarray(B[src][1][meas]).ndim==3:
            bs=[rr(x) for x in np.asarray(B[src][1][meas])]; ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
        rows.append(dict(source=src,est=est,measure=meas,r=r,p_two=pv,ci=ci))
    best=max(((m,rr(mat(g[g.measure==m]))) for m in g.measure.unique() if m.startswith('ssl_')),key=lambda t:t[1])
    r,pv=mantel(mat(g[g.measure==best[0]])); rows.append(dict(source=src,est=est,measure='best layer '+best[0],r=r,p_two=pv,ci=None))
for meas in ['mfcc_euclid','ssl_L23']:
    for nm,srcs in [('ALL',['l2ksu','iqra','everyayah','cv']),('ALL without L2-KSU',['iqra','everyayah','cv'])]:
        A=np.mean([mats[(s,'P',meas)] for s in srcs],0); r,pv=mantel(A); rows.append(dict(source=nm,est='P-avg',measure=meas,r=r,p_two=pv,ci=None))
d=pd.DataFrame(rows); d.to_csv(results_dir('s5.2-rev')+'s52r_main_midpoint.csv',index=False); print(d.round(3).to_string())
