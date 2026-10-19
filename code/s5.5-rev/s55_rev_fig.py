import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import results_dir, figures_dir
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O=results_dir('s5.5-rev'); d=np.load(O+'s55r_scatter.npz'); p=pd.read_csv(O+'s55r_pareto.csv'); n=pd.read_csv(O+'s55r_named_layouts.csv')
fig,a=plt.subplots(figsize=(7,4.5)); a.scatter(d['SCS'],d['SS'],s=8,c='lightgrey',label='collision-free, place-order-preserving layouts (n=548)')
a.plot(p.SCS,p.SS_cm,'k.-',label='Pareto front')
mk={'current (ض midpoint, primary)':('r','o','current (ض midpoint)'),'current (ض start, M2)':('m','v','current, ض start (M2)'),'redesign RD1: ض at tip (M1, Hold)':('b','s','RD1: ض at tip')}
for nm,(c,m,l) in mk.items():
    r=n[n.layout==nm].iloc[0]; a.scatter(r.SCS_primary,r.SS_cm,c=c,marker=m,s=60,label=l,zorder=5)
a.set_xlabel('Spatial correspondence score (0.6 artic. + 0.3 conf. + 0.1 MFCC)'); a.set_ylabel('Confusion-weighted spatial separation\nof confused pairs (cm)')
a.legend(fontsize=7); plt.tight_layout(); plt.savefig(figures_dir()+'s55r_fig_pareto.png',dpi=600)
