"""S5.1 articulatory validation: layout order vs place-of-articulation evidence. Deterministic (exact permutation over 8! orders)."""
import json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
OUT='results/s5.1/'
T=['ت','ر','ض','ي','ك','ق','خ','غ']; IPA=['t','r','dˤ','j','k','q','x','ɣ']
CODE={'ت':'TS','ر':'RS','ض':'DB','ي':'YS','ك':'KS','ق':'QS','خ':'XS','غ':'GS'}
LAYOUT={'tip':[1.5,1.5,1.5,11.0,15.0,18.0,20.5,23.0],   # ض at M1 (as in 5.5 tip variant)
        'main':[1.5,1.5,11.0,11.0,15.0,18.0,20.5,23.0]}  # ض at its first motor M2
# Evidence orderings (front=small). Sources: see 02-literature/articulatory-comparison.md, layout-sources.md
EVID={
 'KAPD_label':  [0,0,0,1,2,3,3,3],          # alveolar/palatal/velar/uvular (LINGUIST.HTM)
 'Modern_lit':  [0,0,0,1,2,3,2.5,2.5],      # Abo Mokh et al. 2020: /χ/ at or anterior to uvular stop; ɣ assumed = x (KAPD pairs them)
 'Classical':   [0,0,1,2,3,4,5,5],          # Sibawayh via Ibn Jinni / Ibn al-Jazari: ض edge behind tip, ي mid, ك<ق<خ=غ
}
epg=pd.read_csv('data/kapd/kapd_epg_summary.csv')
inv={v:k for k,v in CODE.items()}
e=epg[epg.code.isin(CODE.values())].copy(); e['L']=e.code.map(inv)
pooled=e.groupby('L').mean_rel_y.mean().reindex(T)
EVID['KAPD_EPG']=list(pooled.round(4).values)
def stats(x,y):
    rho=spearmanr(x,y)[0]; tau=kendalltau(x,y)[0]   # tau-b
    perm=[kendalltau(x,[y[i] for i in p])[0] for p in itertools.permutations(range(8))]
    perm=np.array(perm); perm=np.nan_to_num(perm)
    p=( (perm>=tau-1e-12).sum() )/len(perm)
    # pair concordance
    c=d=te=tl=tb=0
    for i,j in itertools.combinations(range(8),2):
        a=np.sign(x[j]-x[i]); b=np.sign(y[j]-y[i])
        if a==0 and b==0: tb+=1
        elif a==0: tl+=1
        elif b==0: te+=1
        elif a==b: c+=1
        else: d+=1
    return dict(spearman=round(rho,3),kendall_tau_b=round(tau,3),p_exact_one_sided=round(p,4),
                concordant=c,discordant=d,tied_layout_only=tl,tied_evidence_only=te,tied_both=tb)
res={}
for lv,lx in LAYOUT.items():
    for ev,ey in EVID.items(): res[f'{lv}|{ev}']=stats(lx,ey)
# per-speaker and per-vowel EPG
for s in sorted(e.subj.unique()):
    y=e[e.subj==s].groupby('L').mean_rel_y.mean().reindex(T).values
    res[f'tip|KAPD_EPG_spk{s}']=stats(LAYOUT['tip'],list(y))
for v in 'AIU':
    y=e[e.vowel==v].groupby('L').mean_rel_y.mean().reindex(T).values
    res[f'tip|KAPD_EPG_vowel{v}']=stats(LAYOUT['tip'],list(y))
# back-region order q vs x vs ɣ per speaker x vowel
cells=e[e.L.isin(['ق','خ','غ'])].pivot_table(index=['subj','vowel'],columns='L',values='mean_rel_y')
back=dict(n_cells=len(cells),q_behind_x=int((cells['ق']>cells['خ']).sum()),q_equal_x=int((cells['ق']==cells['خ']).sum()),
          gh_behind_x=int((cells['غ']>cells['خ']).sum()),gh_equal_x=int((cells['غ']==cells['خ']).sum()),
          gh_behind_q=int((cells['غ']>cells['ق']).sum()),k_ahead_q=int((e[e.L=='ك'].set_index(['subj','vowel']).mean_rel_y<cells['ق']).sum()))
res['back_region_cells']=back
# per-letter table
rows=[]
lr=rankdata(LAYOUT['tip'])
for ev in EVID:
    er=rankdata(EVID[ev])
    for i,l in enumerate(T): rows.append(dict(evidence=ev,letter=l,ipa=IPA[i],layout_cm_tip=LAYOUT['tip'][i],layout_cm_main=LAYOUT['main'][i],
         layout_rank=lr[i],evidence_value=EVID[ev][i],evidence_rank=er[i],rank_diff=er[i]-lr[i]))
pd.DataFrame(rows).to_csv(OUT+'s51_per_letter_ranks.csv',index=False)
pd.DataFrame([dict(test=k,**v) for k,v in res.items() if k!='back_region_cells']).to_csv(OUT+'s51_agreement.csv',index=False)
epgt=e.groupby('L').agg(n_on=('n_on','mean'),mean_pos=('mean_rel_y','mean'),sd_pos=('mean_rel_y','std'),
     front_closure=('front_central_closure','mean')).reindex(T).round(3); epgt.to_csv(OUT+'s51_kapd_epg_by_letter.csv')
json.dump(dict(results=res,evidence=EVID,layout=LAYOUT),open(OUT+'s51_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=0)); print(epgt)
