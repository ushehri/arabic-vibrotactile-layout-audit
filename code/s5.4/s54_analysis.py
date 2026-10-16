"""S5.4 L2 confusion check. Inputs: tokens.csv (s54_parse.py), full52-v2/results_v2.json. Seeded, deterministic."""
import json,itertools,numpy as np,pandas as pd,panphon
from scipy.stats import rankdata
rng=np.random.default_rng(54)
NPERM=9999; NBOOT=10000
T=['ت','ر','ض','ي','ك','ق','خ','غ']; IPA=['t','r','dˤ','j','k','q','x','ɣ']
SLEEVE_MAIN={'ت':1.5,'ر':1.5,'ض':11.0,'ي':11.0,'ك':15.0,'ق':18.0,'خ':20.5,'غ':23.0}
d=pd.read_csv('tokens.csv'); d=d[d.group=='non-native_speakers']
x=d[d.canon.isin(T)].copy(); x['sentence']=x.sentence.astype(str)
SPK=sorted(x.speaker.unique())
def sleeve(pos): p=np.array([pos[l] for l in T]); return np.abs(p[:,None]-p[None,:])
iu=np.triu_indices(8,1)
# ---------- reference distances ----------
ft=panphon.FeatureTable(); V=np.array([ft.word_to_vector_list(s,numeric=True)[0] for s in IPA])
FEAT=(V[:,None,:]!=V[None,:,:]).sum(2).astype(float)
v2=json.load(open('data/full52-v2/results_v2.json'))
ACOU={k:np.array(v2['ALL'][k]['mean']) for k in ('mfcc_euclid','ssl_L23')}
# ---------- matrices ----------
def mats(df):
    N=np.array([(df.canon==l).sum() for l in T],float)
    C=np.zeros((8,8))
    for i,a in enumerate(T):
        s=df[df.canon==a].real.value_counts()
        for j,b in enumerate(T): C[i,j]=s.get(b,0)
    return N,C
def sim(N,C,kind='rate'):
    M=C.copy(); np.fill_diagonal(M,0)
    if kind=='rate': M=M/np.where(N[:,None]>0,N[:,None],np.nan)
    return (M+M.T)/2
def spear_vec(a): return rankdata(a)
def mantel(S,D,nperm=NPERM,rng=rng):
    s=spear_vec(S[iu]); r0=np.corrcoef(s,spear_vec(D[iu]))[0,1]; cnt=0; perms=[]
    for _ in range(nperm):
        p=rng.permutation(8); r=np.corrcoef(s,spear_vec(D[np.ix_(p,p)][iu]))[0,1]; perms.append(r)
    perms=np.array(perms)
    return dict(r=r0,p_one_sided_neg=(1+(perms<=r0).sum())/(nperm+1),p_two_sided=(1+(np.abs(perms)>=abs(r0)).sum())/(nperm+1))
def pcor(a,b,c):
    a,b,c=[spear_vec(v) for v in (a,b,c)]
    ra=a-np.polyval(np.polyfit(c,a,1),c); rb=b-np.polyval(np.polyfit(c,b,1),c)
    return np.corrcoef(ra,rb)[0,1]
def pmantel(S,D,Z,nperm=NPERM,rng=rng):
    r0=pcor(S[iu],D[iu],Z[iu]); perms=np.array([pcor(S[iu],D[np.ix_(p,p)][iu],Z[iu]) for p in (rng.permutation(8) for _ in range(nperm))])
    return dict(r=r0,p_one_sided_neg=(1+(perms<=r0).sum())/(nperm+1),p_two_sided=(1+(np.abs(perms)>=abs(r0)).sum())/(nperm+1))
def boot_r(df,D,kind='rate',nb=NBOOT):
    g={s:df[df.speaker==s] for s in SPK}; pre={s:mats(g[s]) for s in SPK}; out=[]
    for _ in range(nb):
        smp=rng.choice(SPK,len(SPK)); N=sum(pre[s][0] for s in smp); C=sum(pre[s][1] for s in smp)
        S=sim(N,C,kind)[iu]
        if np.nanstd(S)==0: out.append(np.nan); continue
        out.append(np.corrcoef(spear_vec(S),spear_vec(D[iu]))[0,1])
    out=np.array(out); return dict(ci95=[float(np.nanpercentile(out,2.5)),float(np.nanpercentile(out,97.5))],n_nan=int(np.isnan(out).sum()),p_boot_r_lt0=float(np.nanmean(out<0)))
R={}
def run(label,df,pos,kind='rate',full=True):
    N,C=mats(df); S=sim(N,C,kind); D=sleeve(pos); res=dict(N=N.tolist(),mantel_sleeve=mantel(S,D))
    if full:
        res['mantel_feature']=mantel(S,FEAT)
        for k,A in ACOU.items(): res['mantel_'+k]=mantel(S,A)
        res['partial_sleeve|feature']=pmantel(S,D,FEAT)
        for k,A in ACOU.items(): res['partial_sleeve|'+k]=pmantel(S,D,A)
        res['boot_sleeve']=boot_r(df,D,kind)
    R[label]=res; return N,C,S
N,C,S=run('main_rate_dd11',x,SLEEVE_MAIN)
pos15=dict(SLEEVE_MAIN,**{'ض':1.5})
run('sens_dd1.5',x,pos15)
run('sens_counts',x,SLEEVE_MAIN,'count')
run('sens_noS5',x[x.sentence!='5'],SLEEVE_MAIN)
run('sens_noS5_dd1.5',x[x.sentence!='5'],pos15,full=False)
run('sens_noYa',x,SLEEVE_MAIN,full=False)  # placeholder overwritten below
# exclude ي: 7x7
T7=[l for l in T if l!='ي']; k7=[T.index(l) for l in T7]; iu7=np.triu_indices(7,1)
S7=S[np.ix_(k7,k7)]; D7=sleeve(SLEEVE_MAIN)[np.ix_(k7,k7)]
s7=spear_vec(S7[iu7]); r7=np.corrcoef(s7,spear_vec(D7[iu7]))[0,1]
pp=np.array([np.corrcoef(s7,spear_vec(D7[np.ix_(p,p)][iu7]))[0,1] for p in (rng.permutation(7) for _ in range(NPERM))])
R['sens_noYa']=dict(mantel_sleeve=dict(r=r7,p_one_sided_neg=(1+(pp<=r7).sum())/(NPERM+1),p_two_sided=(1+(np.abs(pp)>=abs(r7)).sum())/(NPERM+1)))
# leave-one-word-out
lowo=[]
for w in sorted(x.word.unique()):
    sub=x[x.word!=w]; Nw,Cw=mats(sub); Sw=sim(Nw,Cw); s=Sw[iu]
    if np.any(Nw==0): continue
    lowo.append(dict(word_dropped=w,r=float(np.corrcoef(spear_vec(s),spear_vec(sleeve(SLEEVE_MAIN)[iu]))[0,1])))
R['leave_one_word_out']=dict(min=min(l['r'] for l in lowo),max=max(l['r'] for l in lowo),n=len(lowo),items=lowo)
# ---------- tables ----------
full=pd.crosstab(x.canon,x.real).reindex(T); full.to_csv('s54_full_substitution_matrix.csv')
rows=[]
for i,a in enumerate(T):
    sa=x[x.canon==a]; n=len(sa); dele=(sa.real=='<del>').sum(); within=C[i].sum()-C[i,i]
    off=n-C[i,i]-dele-within
    # speaker bootstrap of error rate
    per=sa.groupby('speaker').apply(lambda g:pd.Series(dict(n=len(g),e=(g.real!=a).sum(),s=((g.real!=a)&(g.real!='<del>')).sum()))).reindex(SPK).fillna(0)
    bs=[];bss=[]
    for _ in range(NBOOT):
        smp=per.iloc[rng.integers(0,len(SPK),len(SPK))]; bs.append(smp.e.sum()/smp.n.sum()); bss.append(smp.s.sum()/smp.n.sum())
    rows.append(dict(letter=a,ipa=IPA[i],N=n,correct=int(C[i,i]),within8_sub=int(within),offtarget_sub=int(off),deletions=int(dele),
        error_rate=(n-C[i,i])/n,err_ci_lo=np.percentile(bs,2.5),err_ci_hi=np.percentile(bs,97.5),
        sub_rate=(within+off)/n,sub_ci_lo=np.percentile(bss,2.5),sub_ci_hi=np.percentile(bss,97.5),
        speakers_with_error=int((per.e>0).sum()),n_words=sa.word.nunique(),sentences=','.join(sorted(sa.sentence.unique()))))
er=pd.DataFrame(rows); er.to_csv('s54_per_letter_error_rates.csv',index=False)
pd.DataFrame(C,index=T,columns=T).to_csv('s54_within8_counts.csv')
pd.DataFrame(C/N[:,None],index=T,columns=T).to_csv('s54_within8_rates.csv')
pd.DataFrame(S,index=T,columns=T).to_csv('s54_within8_symmetrised_rate.csv')
# word-level confound table for nonzero within-8 confusions and off-target/deletions
wl=x[x.real!=x.canon].groupby(['canon','real','word']).size().rename('n').reset_index()
opp=x.groupby(['canon','word']).size().rename('opportunities').reset_index()
wl=wl.merge(opp,on=['canon','word']); wl['rate']=wl.n/wl.opportunities
wl['within8']=wl.real.isin(T); wl.sort_values(['canon','real','n'],ascending=[True,True,False]).to_csv('s54_word_level.csv',index=False)
spk=x[x.real!=x.canon].groupby(['canon','real']).speaker.nunique().rename('n_speakers')
pairs=[]
for i,j in zip(*iu):
    a,b=T[i],T[j]
    pairs.append(dict(pair=f'{a}–{b}',sleeve_cm=sleeve(SLEEVE_MAIN)[i,j],sleeve_dd15=sleeve(pos15)[i,j],feature_dist=FEAT[i,j],
        mfcc=ACOU['mfcc_euclid'][i,j],ssl_L23=ACOU['ssl_L23'][i,j],count_ab=int(C[i,j]),count_ba=int(C[j,i]),
        rate_ab=C[i,j]/N[i],rate_ba=C[j,i]/N[j],sym_rate=S[i,j],
        spk_ab=int(spk.get((a,b),0)),spk_ba=int(spk.get((b,a),0)),
        words_ab=int(wl[(wl.canon==a)&(wl.real==b)].word.nunique()),words_ba=int(wl[(wl.canon==b)&(wl.real==a)].word.nunique())))
pd.DataFrame(pairs).to_csv('s54_pairs.csv',index=False)
mt=[]
for lab,res in R.items():
    for k,v in res.items():
        if isinstance(v,dict) and 'r' in v and k!='leave_one_word_out': mt.append(dict(analysis=lab,test=k,**{kk:vv for kk,vv in v.items()}))
        pass
pd.DataFrame(mt).to_csv('s54_mantel.csv',index=False)
R['bootstrap']={k:v['boot_sleeve'] for k,v in R.items() if isinstance(v,dict) and 'boot_sleeve' in v}
R['meta']=dict(letters=T,ipa=IPA,sleeve_main=SLEEVE_MAIN,n_speakers=len(SPK),n_tokens=int(len(x)),nperm=NPERM,nboot=NBOOT,seed=54,
  feature_dist='PanPhon 24-feature Hamming count',acoustic='full52-v2 results_v2.json ALL (pooled) mfcc_euclid and ssl_L23 mean matrices, off-diagonal')
json.dump(R,open('s54_results.json','w'),ensure_ascii=False,indent=1,default=float)
print(er.to_string()); print(pd.DataFrame(mt).to_string()); print(R['leave_one_word_out']['min'],R['leave_one_word_out']['max'])
print(pd.DataFrame(pairs)[['pair','sleeve_cm','feature_dist','count_ab','count_ba','sym_rate','spk_ab','spk_ba','words_ab','words_ba']].to_string())
m=pd.read_csv('s54_mantel.csv')
for k,v in R['bootstrap'].items():
    i=(m.analysis==k)&(m.test=='mantel_sleeve'); m.loc[i,'boot_ci_lo']=v['ci95'][0]; m.loc[i,'boot_ci_hi']=v['ci95'][1]; m.loc[i,'boot_P_r_lt0']=v['p_boot_r_lt0']
m.to_csv('s54_mantel.csv',index=False)
