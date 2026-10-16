"""5.1 revision. (a) layout agreement with ض primary = Push centroid 6.25 cm (start 11.0 / end 1.5 sensitivity);
(b) EPG alternative place measures: anterior-most contacted row and row of maximal contact (peak frame), same decoding as P_sample/analyse.py;
(c) UTI emphasis at speaker level: per-speaker means over vowels, exact sign test, random-intercept mixed model over 25 cells."""
import os,struct,itertools,json,numpy as np,pandas as pd
from scipy.stats import spearmanr,kendalltau,binomtest
import statsmodels.formula.api as smf
O='results/s5.1-rev/'
T=['ت','ر','ض','ي','ك','ق','خ','غ']; CODE={'ت':'TS','ر':'RS','ض':'DB','ي':'YS','ك':'KS','ق':'QS','خ':'XS','غ':'GS'}
# ---- (b) EPG
os.chdir('data/kapd/P_sample')
def chunks(f):
    d=open(f,'rb').read();i=12;out={}
    while i+8<=len(d):
        cid=d[i:i+4].decode('latin1');ln=struct.unpack('<I',d[i+4:i+8])[0];out.setdefault(cid,[]).append(d[i+8:i+8+ln]);i+=8+ln+(ln&1)
    return out
rows=[]
for s in 'AFGM':
  for l,c in CODE.items():
    for v in 'AIU':
      f=f'{s}/P{s}{c}SM{v}.NSP'
      if not os.path.exists(f): continue
      ch=chunks(f);P=np.array(struct.unpack('<196h',ch['PREF'][0])[4:]).reshape(96,2)
      fr=np.unpackbits(np.frombuffer(ch['PALA'][0],np.uint8),bitorder='little').reshape(-1,96)
      ns=len(ch['SDA_'][0])//2; mk=[struct.unpack('<I',m[:4])[0] for m in ch['MKA_']]
      k=fr.shape[0]/ns; a,b=int(mk[1]*k),int(mk[3]*k)+1; seg=fr[a:b]; on=fr[a+seg.sum(1).argmax()].astype(bool)
      y=P[:,1]; rel=(y-y.min())/(y.max()-y.min()); ry=np.round(rel,3); rowsY=np.unique(ry)
      if on.sum()==0: continue
      prop=np.array([on[ry==r].mean() for r in rowsY]); best=rowsY[prop==prop.max()].mean()
      rows.append(dict(subj=s,letter=l,vowel=v,mean_pos=rel[on].mean(),anterior_most=rel[on].min(),max_contact_row=best))
E=pd.DataFrame(rows); E.to_csv(O+'s51r_epg_measures_cells.csv',index=False)
Em=E.groupby('letter')[['mean_pos','anterior_most','max_contact_row']].mean().reindex(T)
# ---- (a) agreement
L={'midpoint 6.25 (primary)':[1.5,1.5,6.25,11,15,18,20.5,23],'start 11.0':[1.5,1.5,11,11,15,18,20.5,23],'end 1.5':[1.5,1.5,1.5,11,15,18,20.5,23]}
EV={'KAPD_label':[0,0,0,1,2,3,3,3],'Modern_lit':[0,0,0,1,2,3,2.5,2.5],'Classical':[0,0,1,2,3,4,5,5],
    'EPG_mean_pos':list(Em.mean_pos),'EPG_anterior_most':list(Em.anterior_most),'EPG_max_contact_row':list(Em.max_contact_row)}
def st(x,y):
    tau=kendalltau(x,y)[0]; perm=np.nan_to_num([kendalltau(x,[y[i] for i in p])[0] for p in itertools.permutations(range(8))])
    c=d=0
    for i,j in itertools.combinations(range(8),2):
        a=np.sign(x[j]-x[i]); b=np.sign(y[j]-y[i])
        if a and b: c+=a==b; d+=a!=b
    return dict(spearman=round(spearmanr(x,y)[0],3),kendall_tau_b=round(tau,3),p_exact=round(float((perm>=tau-1e-12).mean()),4),concordant=int(c),discordant=int(d))
ag=pd.DataFrame([dict(layout=ln,evidence=en,**st(lx,ey)) for ln,lx in L.items() for en,ey in EV.items()]); ag.to_csv(O+'s51r_agreement.csv',index=False)
# ---- (c) UTI speaker level
c=pd.read_csv('results/s5.1/s51_uti_emphasis_cells.csv'); c=c[(c.pair=='d')&(c.region!='Vallecula')]
U=[]
for reg,g in c.groupby('region'):
    sp=g.groupby('participant').dX.mean(); nb=int((sp<0).sum())
    mm=smf.mixedlm('dX ~ 1',g,groups=g['participant']).fit(reml=True)
    U.append(dict(region=reg,speaker_means=' '.join(f'{k}:{v:.3f}' for k,v in sp.items()),n_speakers_backed=nb,
        sign_test_p_two_sided=binomtest(nb,len(sp),0.5).pvalue,sign_test_p_one_sided=binomtest(nb,len(sp),0.5,alternative='greater').pvalue,
        mixed_intercept=mm.params['Intercept'],mixed_se=mm.bse['Intercept'],mixed_p=mm.pvalues['Intercept'],speaker_var=float(mm.cov_re.iloc[0,0])))
U=pd.DataFrame(U); U.to_csv(O+'s51r_uti_speaker_level.csv',index=False)
pd.set_option('display.width',250); print(Em.round(3)); print(ag.to_string()); print(U.round(4).to_string())
