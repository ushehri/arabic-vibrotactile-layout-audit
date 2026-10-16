import pickle,numpy as np,pandas as pd
from scipy.stats import spearmanr
R=pickle.load(open('results/s5.2-rev/s52r_window_mats.pkl','rb'))
B=pickle.load(open('data/full52-v2/pooled_boots.pkl','rb'))
LET=['t','r','dd','j','k','q','x','gh']; IU=np.triu_indices(8,1); POS={'t':1.5,'r':1.5,'dd':11.0,'j':11.0,'k':15.0,'q':18.0,'x':20.5,'gh':23.0}
def sl(dd): p=np.array([dict(POS,dd=dd)[l] for l in LET]); return np.abs(p[:,None]-p[None,:])
def mant(A,S,n=9999):
    r0=spearmanr(A[IU],S[IU])[0]; g=np.random.default_rng(0); pp=np.array([spearmanr(A[IU],S[np.ix_(p,p)][IU])[0] for p in (g.permutation(8) for _ in range(n))])
    return r0,(1+(pp>=r0).sum())/(n+1),(1+(np.abs(pp)>=abs(r0)).sum())/(n+1)
out=[]
for c in R: out.append(dict(pool=c,window='orig vs stored',r=spearmanr(R[c]['orig'][IU],B[c][0]['mfcc_euclid'][IU])[0]))
for pool,src in {'ALL':['l2ksu','iqra','everyayah','cv'],'ALL without L2-KSU':['iqra','everyayah','cv']}.items():
    for w in ['orig','c100','c60']:
        M=np.mean([R[c][w] for c in src],0)
        for dd in (6.25,11.0,1.5):
            r,p1,p2=mant(M,sl(dd)); out.append(dict(pool=pool,window=w,dad=dd,r=r,p_one_sided_pos=p1,p_two_sided=p2))
df=pd.DataFrame(out); df.to_csv('results/s5.2-rev/s52r_window_mantel.csv',index=False); print(df.round(4).to_string())
