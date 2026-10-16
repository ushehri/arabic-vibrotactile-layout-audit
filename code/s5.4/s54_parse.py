"""Parse all L2-KSU TextGrids into a token table (one row per canonical letter interval)."""
import re,glob,csv,os
ROOT='data/l2ksu/extracted/L2-KSU-Dataset'
OUT='data/s54/tokens.csv'
def tiers(f):
    raw=open(f,'rb').read()
    for enc in ('utf-8','utf-16'):
        try: t=raw.decode(enc); break
        except Exception: pass
    out={}
    for m in re.finditer(r'name = "([^"]*)"(.*?)(?=item \[|\Z)',t,re.S):
        iv=re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"',m.group(2))
        out[m.group(1)]=[(float(a),float(b),c.strip()) for a,b,c in iv]
    return out
rows=[];odd=0
for f in sorted(glob.glob(ROOT+'/*/*/*/*/annotation/*.TextGrid')):
    rel=f.split(ROOT+'/')[1].split('/'); grp,sex,spk,sen=rel[:4]
    tt=tiers(f)
    names=sorted(tt,key=lambda k:len(tt[k]))
    ph=tt[names[-1]]; wd=tt[names[0]] if len(names)>1 else []
    swapped=int(names[-1]!='phones')
    for a,b,lab in ph:
        if lab in ('','sil','sp','SIL'): continue
        mid=(a+b)/2; word=next((w for x,y,w in wd if x<=mid<=y),'')
        parts=[p.strip() for p in re.split(r'[,،]',lab)]
        if len(parts)==1 and len(parts[0].split())==1: c=r=parts[0]; typ='c'
        elif len(parts)==3 and parts[2] in ('s','d','i','a'):
            c,r,typ=parts
            if typ in ('i','a'): c='<ins>'
            if typ=='d': r='<del>'
        else: odd+=1; continue
        rows.append(dict(group=grp,sex=sex,speaker=f'{grp[:2]}_{sex[0]}_{spk}',sentence=sen.replace('sentence',''),
            file=os.path.basename(f),swapped=swapped,canon=c,real=r,type=typ,word=word,t0=a,t1=b))
with open(OUT,'w',newline='') as fo:
    w=csv.DictWriter(fo,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('tokens',len(rows),'unparsed',odd)
