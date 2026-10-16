"""5.4 revision: ض primary = midpoint 6.25 (Push centroid); start/end sensitivity; erroneous خ token (canon خ in الرحيم, should be ح) excluded;
7-consonant analysis (no ي) co-primary; leave-one-word-out with 'not estimable' when a letter loses all tokens; two-sided p reported."""
import os,json,itertools,numpy as np,pandas as pd
os.chdir('results/s5.4')
src=open('code/s5.4/s54_analysis.py').read().split('\nR={}')[0]
exec(src)
NPERM=9999; NBOOT=10000
# tie handling: quantise to 1e-9 before ranking (floating-point noise must not break mathematical ties)
from scipy.stats import rankdata as _rk
def rankdata(x): return _rk(np.round(np.asarray(x,float),9))
x=x[~((x.canon=='خ')&(x.word=='الرحيم'))].copy()
CODINGS={'midpoint (primary)':6.25,'start M2 (11.0)':11.0,'end M1 (1.5)':1.5}
def pos(dd): return dict(SLEEVE_MAIN,**{'ض':dd})
def sub(M,keep): return M[np.ix_(keep,keep)]
def mantel_k(S,D,k,nperm=NPERM):
    iuk=np.triu_indices(k,1); s=rankdata(S[iuk]); r0=np.corrcoef(s,rankdata(D[iuk]))[0,1]
    pp=np.array([np.corrcoef(s,rankdata(D[np.ix_(p,p)][iuk]))[0,1] for p in (rng.permutation(k) for _ in range(nperm))])
    return dict(r=float(r0),p_one_sided_neg=float((1+(pp<=r0).sum())/(nperm+1)),p_two_sided=float((1+(np.abs(pp)>=abs(r0)).sum())/(nperm+1)))
def pmantel_k(S,D,Z,k,nperm=NPERM):
    iuk=np.triu_indices(k,1); r0=pcor(S[iuk],D[iuk],Z[iuk])
    pp=np.array([pcor(S[iuk],D[np.ix_(p,p)][iuk],Z[iuk]) for p in (rng.permutation(k) for _ in range(nperm))])
    return dict(r=float(r0),p_one_sided_neg=float((1+(pp<=r0).sum())/(nperm+1)),p_two_sided=float((1+(np.abs(pp)>=abs(r0)).sum())/(nperm+1)))
def boot_k(df,D,keep,nb=NBOOT):
    k=len(keep); iuk=np.triu_indices(k,1); pre={s:mats(df[df.speaker==s]) for s in SPK}; out=[]
    for _ in range(nb):
        smp=rng.choice(SPK,len(SPK)); N=sum(pre[s][0] for s in smp); C=sum(pre[s][1] for s in smp); S=sub(sim(N,C),keep)[iuk]
        out.append(np.nan if np.nanstd(S)==0 else np.corrcoef(rankdata(S),rankdata(D[iuk]))[0,1])
    out=np.array(out); return [float(np.nanpercentile(out,2.5)),float(np.nanpercentile(out,97.5))]
N,C=mats(x); S=sim(N,C)
SETS={'8 letters':list(range(8)),'7 consonants (no ي)':[i for i in range(8) if T[i]!='ي']}
rows=[]
for sn,keep in SETS.items():
    k=len(keep); Sk=sub(S,keep); Fk=sub(FEAT,keep)
    for cn,dd in CODINGS.items():
        Dk=sub(sleeve(pos(dd)),keep)
        m=mantel_k(Sk,Dk,k); rows.append(dict(set=sn,dad=cn,test='sleeve',**m,boot_ci=boot_k(x,Dk,keep)))
        rows.append(dict(set=sn,dad=cn,test='sleeve | feature (partial)',**pmantel_k(Sk,Dk,Fk,k)))
        iuk=np.triu_indices(k,1); rows.append(dict(set=sn,dad=cn,test='rho(sleeve, feature) over pairs',r=float(np.corrcoef(rankdata(Dk[iuk]),rankdata(Fk[iuk]))[0,1])))
    rows.append(dict(set=sn,dad='-',test='feature',**mantel_k(Sk,Fk,k)))
    for a,A in ACOU.items(): rows.append(dict(set=sn,dad='-',test=a,**mantel_k(Sk,sub(A,keep),k)))
res=pd.DataFrame(rows); res.to_csv('results/s5.4-rev/s54r_mantel.csv',index=False)
# leave-one-word-out, primary coding, both sets
lw=[]
for sn,keep in SETS.items():
    k=len(keep); iuk=np.triu_indices(k,1); Dk=sub(sleeve(pos(6.25)),keep)
    for w in sorted(x.word.unique()):
        s2=x[x.word!=w]; Nw,Cw=mats(s2); lost=[T[i] for i in keep if Nw[i]==0]
        if lost: lw.append(dict(set=sn,word_dropped=w,r=None,status='not estimable: no tokens of '+'/'.join(lost))); continue
        Sw=sub(sim(Nw,Cw),keep)[iuk]; lw.append(dict(set=sn,word_dropped=w,r=float(np.corrcoef(rankdata(Sw),rankdata(Dk[iuk]))[0,1]),status='ok'))
lw=pd.DataFrame(lw); lw.to_csv('results/s5.4-rev/s54r_lowo.csv',index=False)
summ={sn:dict(min=float(g.r.min()),max=float(g.r.max()),n_ok=int((g.status=='ok').sum()),not_estimable=g[g.status!='ok'][['word_dropped','status']].values.tolist(),
      min_word=g.loc[g.r.idxmin(),'word_dropped'],max_word=g.loc[g.r.idxmax(),'word_dropped']) for sn,g in lw.groupby('set')}
json.dump(dict(n_tokens=int(len(x)),excluded='1 token canon خ in الرحيم (annotation error; realised ح)',N=dict(zip(T,N.tolist())),lowo=summ),open('results/s5.4-rev/s54r_summary.json','w'),ensure_ascii=False,indent=1)
pd.set_option('display.width',250); print(res.to_string()); print(json.dumps(summ,ensure_ascii=False,indent=1))
# ---- per-letter table, totals, pair table and figures from the corrected token set (erroneous خ excluded) ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUTR='results/s5.4-rev/'; FIG='figures/'
rows=[]
for i,a in enumerate(T):
    sa=x[x.canon==a]; n=len(sa); dele=int((sa.real=='<del>').sum()); within=int(C[i].sum()-C[i,i]); off=int(n-C[i,i]-dele-within)
    per=sa.groupby('speaker').apply(lambda g:pd.Series(dict(n=len(g),e=(g.real!=a).sum(),s=((g.real!=a)&(g.real!='<del>')).sum()))).reindex(SPK).fillna(0)
    bs=[];bss=[]
    for _ in range(NBOOT):
        smp=per.iloc[rng.integers(0,len(SPK),len(SPK))]; bs.append(smp.e.sum()/smp.n.sum()); bss.append(smp.s.sum()/smp.n.sum())
    rows.append(dict(letter=a,ipa=IPA[i],N=n,correct=int(C[i,i]),within8_sub=within,offtarget_sub=off,deletions=dele,
        error_pct=100*(n-C[i,i])/n,err_lo=100*np.percentile(bs,2.5),err_hi=100*np.percentile(bs,97.5),
        sub_pct=100*(within+off)/n,sub_lo=100*np.percentile(bss,2.5),sub_hi=100*np.percentile(bss,97.5),
        learners_with_error=int((per.e>0).sum()),n_words=sa.word.nunique()))
er=pd.DataFrame(rows); er.to_csv(OUTR+'s54r_per_letter.csv',index=False)
tot=dict(tokens=int(er.N.sum()),correct=int(er.correct.sum()),errors=int(er.N.sum()-er.correct.sum()),within8=int(er.within8_sub.sum()),
         offtarget=int(er.offtarget_sub.sum()),deletions=int(er.deletions.sum()))
pr=[]
for i,j in itertools.combinations(range(8),2):
    if C[i,j]+C[j,i]>0: pr.append(dict(pair=f'{T[i]}–{T[j]}',a_as_b=int(C[i,j]),b_as_a=int(C[j,i]),S_pct=100*S[i,j]))
pd.DataFrame(pr).to_csv(OUTR+'s54r_pairs.csv',index=False)
J=json.load(open(OUTR+'s54r_summary.json')); J['totals']=tot; J['NBOOT']=NBOOT; json.dump(J,open(OUTR+'s54r_summary.json','w'),ensure_ascii=False,indent=1)
print(er.round(1).to_string()); print(tot); print(pd.DataFrame(pr).round(2).to_string())
fig,a=plt.subplots(figsize=(7,3.5)); xs=np.arange(8)
a.bar(xs-0.2,er.error_pct,0.4,yerr=[er.error_pct-er.err_lo,er.err_hi-er.error_pct],label='all errors',capsize=2)
a.bar(xs+0.2,er.sub_pct,0.4,yerr=[er.sub_pct-er.sub_lo,er.sub_hi-er.sub_pct],label='substitutions',capsize=2)
a.set_xticks(xs); a.set_xticklabels([f'/{p}/\nN={n}' for p,n in zip(er.ipa,er.N)]); a.set_ylabel('% of target tokens'); a.legend(); fig.tight_layout(); fig.savefig(FIG+'s54r_fig3_error_rates.png',dpi=200)
fig,a=plt.subplots(figsize=(5.5,5)); R_=C/N[:,None]; M=np.where(np.eye(8,dtype=bool),np.nan,R_); a.imshow(M,cmap='Reds')
for i in range(8):
    for j in range(8):
        if i!=j and C[i,j]>0: a.text(j,i,f'{int(C[i,j])}\n{100*R_[i,j]:.1f}%',ha='center',va='center',fontsize=7)
a.set_xticks(range(8)); a.set_yticks(range(8)); a.set_xticklabels([f'/{p}/' for p in IPA]); a.set_yticklabels([f'/{p}/' for p in IPA]); a.set_xlabel('realised'); a.set_ylabel('target')
fig.tight_layout(); fig.savefig(FIG+'s54r_fig1_confusion_heatmap.png',dpi=200); print('DONE')
