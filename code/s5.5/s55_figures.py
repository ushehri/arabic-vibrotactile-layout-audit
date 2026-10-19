import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import arabic_font, workdir
import json,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm;import arabic_reshaper;from bidi.algorithm import get_display
fp=arabic_font()
if fp: fm.fontManager.addfont(fp)
AF=fm.FontProperties(family=['Noto Sans Arabic','DejaVu Sans'])
import os; os.chdir(workdir('s55','full'))   # reads the outputs of s55_full.py
# matplotlib >= 3.11 shapes and reorders Arabic itself (libraqm); a leading LRM fixes LTR base direction
A=lambda s:'\u200e'+s
R=json.load(open('s55_results.json'));CR=['blend','panphon','place','confusion','mfcc','combined']
NM={'main':'Fixed slots','dad_tip':'ض-at-tip slots'}
# Fig1: sleeve diagrams
fig,ax=plt.subplots(figsize=(11,4.2))
rows=[('Current (fixed slots)',R['main']['criteria']['combined']['score'],{'ت':1.5,'ر':1.5,'ض':11,'ي':11,'ك':15,'ق':18,'خ':20.5,'غ':23}),
      ('Fixed-slot alternative (ت↔ض)',R['main']['recommended']['combined'],R['main']['recommended']['positions']),
      ('Recommended (ض at tip)',R['dad_tip']['criteria']['combined']['score'],{'ت':1.5,'ر':1.5,'ض':1.5,'ي':11,'ك':15,'ق':18,'خ':20.5,'غ':23}),
      ('Best tip',R['dad_tip']['recommended']['combined'],R['dad_tip']['recommended']['positions'])]
for k,(lab,sc,pos) in enumerate(rows):
    y=-k; ax.plot([0,24],[y,y],c='0.85',lw=8,solid_capstyle='round')
    by={}
    for l,p in pos.items(): by.setdefault(p,[]).append(l)
    for p,ls in by.items():
        for m,l in enumerate(ls): ax.text(p,y+0.22*(m-(len(ls)-1)/2)*1.6,A(l),fontproperties=AF,fontsize=15,ha='center',va='center')
    ax.text(-0.8,y,A(lab)+f'\ncombined rho={sc:.3f}',ha='right',va='center',fontsize=9,fontproperties=AF)
ax.set_xlim(-12,25);ax.set_ylim(-3.7,0.7);ax.set_yticks([]);ax.set_xlabel('cm from wrist');ax.set_xticks([1.5,11,15,18,20.5,23])
for s in ('top','right','left'):ax.spines[s].set_visible(False)
ax.set_title('S5.5 layouts: current, recommended and alternatives (0.6 blend / 0.3 L2 confusion / 0.1 MFCC)')
plt.tight_layout();plt.savefig('s55_fig1_layouts.png',dpi=160);plt.close()
# Fig2: score distributions
fig,axs=plt.subplots(2,6,figsize=(17,5.5))
for r,var in enumerate(['main','dad_tip']):
    S=np.load(f'scores_{var}.npy')
    for c,k in enumerate(CR):
        a=axs[r,c];a.hist(S[c],bins=50,color='0.7');cu=R[var]['criteria'][k]['score'];re=R[var]['recommended'][k]
        a.axvline(cu,c='r',label='current');a.axvline(re,c='b',ls='--',label='best sensible')
        a.set_title(A(NM[var])+f": {k}\nrank {R[var]['criteria'][k]['rank_best']}/{R[var]['n_distinct']}, p≥={R[var]['criteria'][k]['p_ge']:.4f}",fontsize=8,fontproperties=AF)
        a.tick_params(labelsize=7)
axs[0,0].legend(fontsize=7);fig.supxlabel('Spearman ρ (sleeve distance vs target dissimilarity)');plt.tight_layout();plt.savefig('s55_fig2_distributions.png',dpi=150);plt.close()
# Fig3: weight stability
W=pd.read_csv('s55_weights.csv');fig,axs=plt.subplots(1,2,figsize=(11,4))
for a,var in zip(axs,['main','dad_tip']):
    w=W[W.variant==var];pv=w.pivot_table(index='w_artic',columns='w_conf',values='current_rank')
    im=a.imshow(pv.values,cmap='viridis_r',aspect='auto');a.set_xticks(range(len(pv.columns)));a.set_xticklabels(pv.columns);a.set_yticks(range(len(pv.index)));a.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            if not np.isnan(pv.values[i,j]):a.text(j,i,int(pv.values[i,j]),ha='center',va='center',color='w',fontsize=8)
    a.set_xlabel('w confusion (w acoustic = 1 − rest)');a.set_ylabel('w articulatory blend');a.set_title(A(NM[var])+f': rank of current layout (of {R[var]["n_distinct"]})',fontsize=9,fontproperties=AF)
plt.tight_layout();plt.savefig('s55_fig3_weight_stability.png',dpi=150)
