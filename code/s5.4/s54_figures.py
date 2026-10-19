import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import arabic_font, figures_dir, workdir
import json,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm; import arabic_reshaper; from bidi.algorithm import get_display
FP=arabic_font()
if FP: fm.fontManager.addfont(FP)
plt.rcParams['font.family']=[fm.FontProperties(fname=FP).get_name() if FP else 'Noto Sans Arabic','DejaVu Sans']
import os; os.chdir(workdir('s54'))   # reads the s54_* tables written by s54_analysis.py
A=lambda s:s  # matplotlib>=3.11 shapes Arabic natively; reshaper+bidi double-reverses (verified)
T=['ت','ر','ض','ي','ك','ق','خ','غ']; IPA=['t','r','dˤ','j','k','q','x','ɣ']; L=[f'{A(a)} /{b}/' for a,b in zip(T,IPA)]
R=pd.read_csv('s54_within8_rates.csv',index_col=0).values; C=pd.read_csv('s54_within8_counts.csv',index_col=0).values
M=R.copy(); np.fill_diagonal(M,np.nan)
fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(M*100,cmap='Reds')
for i in range(8):
    for j in range(8):
        if i!=j: ax.text(j,i,f'{int(C[i,j])}\n{M[i,j]*100:.1f}%',ha='center',va='center',fontsize=7)
ax.set_xticks(range(8),L,rotation=45); ax.set_yticks(range(8),L); ax.set_xlabel('realised as'); ax.set_ylabel('target (canonical)')
plt.colorbar(im,label='% of target opportunities'); ax.set_title('L2-KSU learners: within-8 substitutions (count, rate)'); plt.tight_layout(); plt.savefig(figures_dir()+'s54_fig1_confusion_heatmap.png',dpi=200); plt.close()
P=pd.read_csv('s54_pairs.csv'); J=json.load(open('s54_results.json'))
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for ax,col,lab,key in [(axs[0],'sleeve_cm','sleeve distance (cm), ض=11.0','main_rate_dd11'),(axs[1],'sleeve_dd15','sleeve distance (cm), ض=1.5','sens_dd1.5')]:
    ax.scatter(P[col],P.sym_rate*100,c='C3')
    for _,r in P[P.sym_rate>0].iterrows(): ax.annotate(A(r.pair),(r[col],r.sym_rate*100),fontsize=9,xytext=(3,3),textcoords='offset points')
    m=J[key]['mantel_sleeve']; b=J[key]['boot_sleeve']['ci95']
    ax.set_title(f"Spearman r={m['r']:.2f}, one-sided p={m['p_one_sided_neg']:.3f}, 95% CI [{b[0]:.2f}, {b[1]:.2f}]",fontsize=9)
    ax.set_xlabel(lab); ax.set_ylabel('symmetrised confusion rate (%)')
plt.tight_layout(); plt.savefig(figures_dir()+'s54_fig2_confusion_vs_sleeve.png',dpi=200); plt.close()
E=pd.read_csv('s54_per_letter_error_rates.csv')
fig,ax=plt.subplots(figsize=(8,4.5)); xs=np.arange(8)
ax.bar(xs-0.2,E.error_rate*100,0.4,yerr=[(E.error_rate-E.err_ci_lo)*100,(E.err_ci_hi-E.error_rate)*100],label='all errors',capsize=3)
ax.bar(xs+0.2,E.sub_rate*100,0.4,yerr=[(E.sub_rate-E.sub_ci_lo)*100,(E.sub_ci_hi-E.sub_rate)*100],label='substitutions only',capsize=3)
ax.set_xticks(xs,[f'{l}\nN={n}' for l,n in zip(L,E.N)]); ax.set_ylabel('% of tokens (95% speaker-bootstrap CI)'); ax.legend(); ax.set_title('Per-letter learner error rates (40 L2 speakers)')
plt.tight_layout(); plt.savefig(figures_dir()+'s54_fig3_error_rates.png',dpi=200); plt.close()
W=pd.read_csv('s54_word_level.csv'); W=W[W.within8&(W.n>=3)].copy()
IP=dict(zip(T,IPA)); W['lab']=[f'{w}   /{IP[c]}/→/{IP[r]}/' for c,r,w in zip(W.canon,W.real,W.word)]; W=W.sort_values('n')
fig,ax=plt.subplots(figsize=(7,0.3*len(W)+1.5)); ax.barh(W.lab,W.n,color='C1')
for i,(n,o) in enumerate(zip(W.n,W.opportunities)): ax.text(n,i,f' {n}/{o}',va='center',fontsize=8)
ax.set_xlabel('count (n / opportunities in that word)'); ax.set_title('Within-8 confusions by word (n≥3)'); plt.tight_layout(); plt.savefig(figures_dir()+'s54_fig4_word_level.png',dpi=200); plt.close()
print('figs done')
