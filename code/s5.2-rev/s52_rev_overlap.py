"""Iqra dev vs Common Voice sample / EveryAyah sample: text overlap (undiacritised, normalised) and, for text matches, duration match (|Δ|<0.05 s)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, results_dir
import pandas as pd, re, json
m=pd.read_csv(data('full52-v2/manifest.csv'),keep_default_na=False)
def norm(s):
    s=re.sub('[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]','',s); s=re.sub('[إأآٱ]','ا',s); s=re.sub('[^\u0621-\u064A ]',' ',s); return ' '.join(s.split())
m['n']=m.text.map(norm)
used={c:set(pd.read_csv(data(f'full52-v2/tokens_{c}.csv'),keep_default_na=False).utt) for c in ['iqra','cv','everyayah']}
m=m[m.apply(lambda r:r.utt in used.get(r.source,set()),axis=1)]
iq=m[m.source=='iqra']; out={'n_utts':m.groupby('source').size().to_dict()}
for c in ['cv','everyayah']:
    o=m[m.source==c]; s=set(o.n); hit=iq[iq.n.isin(s)]
    j=hit.merge(o[['n','dur','utt','speaker']],on='n',suffixes=('','_o')); same=j[(j.dur-j.dur_o).abs()<0.05]
    out[c]=dict(iqra_utts_with_same_text=int(hit.utt.nunique()),distinct_texts=int(hit.n.nunique()),pairs_dur_within_50ms=int(len(same)),
                iqra_utts_dur_match=int(same.utt.nunique()),examples=same[['utt','utt_o','dur','dur_o']].head(5).values.tolist())
# within-Iqra duplicate texts
out['iqra_duplicate_text_utts']=int(iq.n.duplicated(keep=False).sum())
json.dump(out,open(results_dir('s5.2-rev')+'s52r_iqra_overlap.json','w'),ensure_ascii=False,indent=1); print(json.dumps(out,ensure_ascii=False,indent=1))
