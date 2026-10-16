# Authoritative EPG extraction for 5.1 (same decoding as s51_rev.py; unrounded cell values): Table 5.1-3 and Figure 5.1-2
import os,struct,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O='results/s5.1-rev/'
T=['ت','ر','ض','ي','ك','ق','خ','غ']; IPA=['t','r','dˤ','j','k','q','x','ɣ']; CODE={'ت':'TS','ر':'RS','ض':'DB','ي':'YS','ك':'KS','ق':'QS','خ':'XS','غ':'GS'}
POS={'ت':'1.5','ر':'1.5','ض':'6.25 (Push midpoint)','ي':'11.0','ك':'15.0','ق':'18.0','خ':'20.5','غ':'23.0'}
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
      k=fr.shape[0]/ns; a,b=int(mk[1]*k),int(mk[3]*k)+1; on=fr[a+fr[a:b].sum(1).argmax()].astype(bool)
      y=P[:,1]; x=P[:,0]; rel=(y-y.min())/(y.max()-y.min()); cen=np.abs(x-156)<=24
      if on.sum()==0: continue
      rows.append(dict(subj=s,letter=l,vowel=v,n_on=int(on.sum()),mean_pos=rel[on].mean(),closure=bool((on&cen&(rel<0.25)).any())))
E=pd.DataFrame(rows)
tab=E.groupby('letter').agg(n_on=('n_on','mean'),mean_pos=('mean_pos','mean'),sd=('mean_pos','std'),closure=('closure','sum'),cells=('closure','size')).reindex(T)
tab.to_csv(O+'s51r_epg_table.csv'); print(tab.round(3))
sp=E.groupby(['subj','letter']).mean_pos.mean().unstack().reindex(columns=T)
fig,a=plt.subplots(figsize=(7,3.6)); xs=np.arange(8)
for s in sp.index: a.plot(xs,sp.loc[s].values,'o-',alpha=0.6,label=f'speaker {s}')
a.plot(xs,tab.mean_pos.values,'k-s',lw=2,label='pooled')
a.set_xticks(xs); a.set_xticklabels([f'/{p}/\n{POS[t].split(" ")[0]} cm' for p,t in zip(IPA,T)],fontsize=8)
a.set_ylabel('EPG mean contact position\n(0 = front, 1 = back)'); a.set_xlabel('letter in sleeve order (ض at the 6.25 cm Push midpoint)'); a.legend(fontsize=7); fig.tight_layout()
fig.savefig('figures/s51r_fig2_kapd_epg_by_speaker.png',dpi=200)
# per-speaker / per-vowel tau_b (ض midpoint) and back-region cell comparisons on unrounded values
from scipy.stats import kendalltau
pos=[1.5,1.5,6.25,11,15,18,20.5,23]
ts={s:kendalltau(pos,E[E.subj==s].groupby('letter').mean_pos.mean().reindex(T).values)[0] for s in 'AFGM'}
tv={v:kendalltau(pos,E[E.vowel==v].groupby('letter').mean_pos.mean().reindex(T).values)[0] for v in 'AIU'}
W=E.pivot_table(index=['subj','vowel'],columns='letter',values='mean_pos')
cmp=lambda a,b:(int((W[a]>W[b]).sum()),int((W[a]==W[b]).sum()),int(W[[a,b]].notna().all(1).sum()))
out=dict(tau_speaker=ts,tau_vowel=tv,q_behind_x=cmp('ق','خ'),gh_behind_x=cmp('غ','خ'),gh_behind_q=cmp('غ','ق'),k_ahead_q=cmp('ق','ك'))
print(out); import json; json.dump(out,open(O+'s51r_epg_extra.json','w'),ensure_ascii=False,indent=1)
