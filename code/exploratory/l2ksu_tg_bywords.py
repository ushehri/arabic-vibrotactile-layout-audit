import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data
import re,glob,collections,json
ROOT=data('l2ksu/extracted/L2-KSU-Dataset')
T=['ت','ر','ض','ي','ك','ق','خ','غ']
def tiers(f):
    raw=open(f,'rb').read()
    for enc in ('utf-8','utf-16'):
        try: t=raw.decode(enc); break
        except: pass
    out=[]
    for m in re.finditer(r'name = "([^"]*)"(.*?)(?=item \[|\Z)',t,re.S):
        iv=[(float(a),float(b),c.strip()) for a,b,c in re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"',m.group(2))]
        out.append(iv)
    return out
byword=collections.defaultdict(collections.Counter)  # (target) -> Counter((word, realised))
utt=collections.Counter(); spk=collections.defaultdict(set)
for f in glob.glob(ROOT+'/non-native_speakers/*/*/*/annotation/*.TextGrid'):
    rel=f.split(ROOT+'/')[1].split('/'); sent=rel[3]; utt[sent]+=1
    a,b=tiers(f); ph,wd=(a,b) if len(a)>len(b) else (b,a)
    for x0,x1,lab in ph:
        if lab in ('','sil','sp'): continue
        parts=[p.strip() for p in re.split(r'[,،]',lab)]
        if len(parts)==1: c,r=parts[0],parts[0]
        elif len(parts)==3:
            c,r,t=parts
            if t in('i','a'): continue
            if t=='d': r='<del>'
        else: continue
        if c not in T: continue
        mid=(x0+x1)/2
        w=next((l for a0,a1,l in wd if a0<=mid<a1),'?')
        byword[c][(sent+':'+w,r)]+=1; spk[c].add('/'.join(rel[:3]))
print('NN utterances per sentence',dict(sorted(utt.items(),key=lambda x:int(x[0][8:]))))
out={}
for c in T:
    tot=collections.Counter(); err=collections.defaultdict(collections.Counter)
    for (w,r),n in byword[c].items():
        tot[w]+=n
        if r!=c: err[w][r]+=n
    out[c]={w:dict(total=tot[w],errors=dict(err[w])) for w in sorted(tot,key=lambda k:-tot[k])}
    print(c,'NN speakers with token:',len(spk[c])); 
    for w,v in out[c].items(): print('   ',w,v)
json.dump(out,open(data('l2ksu/textgrid_target_by_word.json'),'w'),ensure_ascii=False,indent=1)
