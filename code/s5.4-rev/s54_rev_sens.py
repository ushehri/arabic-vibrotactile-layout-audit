# 5.4 revision sensitivities on the corrected token set (erroneous خ excluded), ض midpoint primary
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import results_dir, figures_dir
src=open(Path(__file__).resolve().parent/'s54_rev.py',encoding='utf-8').read(); exec(src[:src.index("N,C=mats(x)")])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUTR=results_dir('s5.4-rev')
rows=[]
def run(label,df,dd,kind='rate',full=False):
    N_,C_=mats(df); S_=sim(N_,C_,kind); D=sleeve(pos(dd)); r=dict(analysis=label,**mantel_k(S_,D,8))
    if full:
        for zn,Z in [('feature',FEAT)]+list(ACOU.items()): p=pmantel_k(S_,D,Z,8); r[f'partial_{zn}_r']=p['r']; r[f'partial_{zn}_p2']=p['p_two_sided']
    rows.append(r)
run('main: rates, ض midpoint',x,6.25,full=True)
run('ض start 11.0',x,11.0); run('ض end 1.5',x,1.5)
run('counts instead of rates, ض midpoint',x,6.25,kind='count')
run('without sentence 5, ض midpoint',x[x.sentence!='5'],6.25)
d=pd.DataFrame(rows); d.to_csv(OUTR+'s54r_sensitivity.csv',index=False); print(d.round(4).to_string())
N,C=mats(x); S=sim(N,C); iu8=np.triu_indices(8,1)
fig,a=plt.subplots(figsize=(5,4)); D=sleeve(pos(6.25)); a.scatter(D[iu8],100*S[iu8])
for i,j in zip(*iu8):
    if S[i,j]>0: a.annotate(f'/{IPA[i]}/–/{IPA[j]}/',(D[i,j],100*S[i,j]),fontsize=7,xytext=(3,2),textcoords='offset points')
a.set_xlabel('sleeve distance (cm; ض at 6.25)'); a.set_ylabel('symmetrised confusion rate (%)'); fig.tight_layout(); fig.savefig(figures_dir()+'s54r_fig2_confusion_vs_sleeve.png',dpi=200)
