"""S5.1 extra: Lebanese Arabic UTI (Sakr 2025 data via Coretta & Sakr mv_uti, Zenodo 10.5281/zenodo.15335918).
Emphatic minus plain difference in tongue position (X_z, z-scored within speaker; larger X = more anterior, tip to the right)
per tongue region, 35 ms before consonant offset. Focus /d/ vs /dˤ/ (ض analogue); all 4 pairs as context."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, results_dir
import pyreadr, pandas as pd, numpy as np, json
from scipy.stats import wilcoxon
d=pyreadr.read_r(data('s51data/stefanocoretta-mv_uti-7ce67cc/data/sakr2025/dlc_emph_f.rds'))[None]
d['pair']=d.consonant.astype(str).str.lower(); d['region']=d.knot_label.astype(str).str.split('_').str[0]
print(d.groupby(['knot','knot_label'],observed=True).X.mean())
print(d.consonant.unique())
tok=d.groupby(['participant','pair','vowel','emph','id','region'],observed=True)[['X_z','Y_z']].mean().reset_index()
cell=tok.groupby(['participant','pair','vowel','emph','region'],observed=True)[['X_z','Y_z']].mean().reset_index()
w=cell.pivot_table(index=['participant','pair','vowel','region'],columns='emph',values=['X_z','Y_z'],observed=True)
w['dX']=w[('X_z','emphatic')]-w[('X_z','plain')]; w['dY']=w[('Y_z','emphatic')]-w[('Y_z','plain')]
w=w[['dX','dY']].reset_index(); w.columns=['participant','pair','vowel','region','dX','dY']
out={}
for cons in ['d','ALL']:
    s=w if cons=='ALL' else w[w.pair==cons]
    for reg in s.region.unique():
        r=s[s.region==reg]; spk=r.groupby('participant',observed=True).dX.mean()
        out[f'{cons}|{reg}']=dict(n_cells=len(r),mean_dX=round(r.dX.mean(),3),median_dX=round(r.dX.median(),3),
            cells_backed=int((r.dX<0).sum()),spk_backed=int((spk<0).sum()),n_spk=len(spk),
            mean_dY=round(r.dY.mean(),3),wilcoxon_p=round(float(wilcoxon(r.dX).pvalue),4) if len(r)>5 else None)
ntok=tok[tok.pair=='d'].groupby(['emph'],observed=True).id.nunique()
O=results_dir('s5.1')
w.to_csv(O+'s51_uti_emphasis_cells.csv',index=False)
json.dump(out,open(O+'s51_uti_emphasis.json','w'),indent=1)
print(json.dumps(out,indent=0)); print(tok[tok.pair=='d'].groupby(['participant','emph'],observed=True).size()//tok.region.nunique())
