# Figure 5.1-1 (revision): sleeve position with ض at the Push midpoint (6.25 cm) against each front-to-back evidence code
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T=['t','r','dˤ','j','k','q','x','ɣ']; pos=np.array([1.5,1.5,6.25,11,15,18,20.5,23])
E={'KAPD place label':[0,0,0,1,2,3,3,3],'Modern literature':[0,0,0,1,2,3,2.5,2.5],'EPG mean contact':[0.386,0.471,0.325,0.525,0.608,0.645,0.637,0.649],
   'EPG anterior-most row':[0.014,0.124,0.000,0.187,0.304,0.347,0.345,0.390],'EPG max-contact row':[0.407,0.507,0.276,0.546,0.650,0.738,0.724,0.712],'Classical (circular)':[0,0,1,2,3,4,5,5]}
ag=pd.read_csv('05-results/s5.1-rev/s51r_agreement.csv'); ag=ag[ag.layout.str.startswith('midpoint')].set_index('evidence')
key={'KAPD place label':'KAPD_label','Modern literature':'Modern_lit','EPG mean contact':'EPG_mean_pos','EPG anterior-most row':'EPG_anterior_most','EPG max-contact row':'EPG_max_contact_row','Classical (circular)':'Classical'}
fig,ax=plt.subplots(2,3,figsize=(10,6))
for a,(k,v) in zip(ax.flat,E.items()):
    a.scatter(pos,v)
    for p,y,t in zip(pos,v,T): a.annotate('/'+t+'/',(p,y),fontsize=7,xytext=(3,2),textcoords='offset points')
    r=ag.loc[key[k]]; a.set_title(f'{k}\nρ={r.spearman:.3f}, τb={r.kendall_tau_b:.3f}',fontsize=9); a.set_xlabel('sleeve position (cm; ض at 6.25)',fontsize=8)
fig.tight_layout(); fig.savefig('06-figures/s51r_fig1_layout_vs_evidence.png',dpi=200)
