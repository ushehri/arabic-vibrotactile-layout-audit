import re,glob,collections,json
ROOT='data/l2ksu/extracted/L2-KSU-Dataset'
def tiers(f):
    raw=open(f,'rb').read()
    for enc in ('utf-8','utf-16'):
        try: t=raw.decode(enc); break
        except: pass
    out={}
    for m in re.finditer(r'name = "([^"]*)"(.*?)(?=item \[|\Z)',t,re.S):
        out[m.group(1)]=[x.strip() for x in re.findall(r'text = "([^"]*)"',m.group(2))]
    return out
pairs={'native_speakers':collections.Counter(),'non-native_speakers':collections.Counter()}
odd=collections.Counter(); files=collections.Counter(); swapped=0; spk_err=collections.defaultdict(set)
for f in sorted(glob.glob(ROOT+'/*/*/*/*/annotation/*.TextGrid')):
    g=f.split(ROOT+'/')[1].split('/')[0]; spk='/'.join(f.split(ROOT+'/')[1].split('/')[:3])
    tt=tiers(f); files[g]+=1
    ph=max(tt.values(),key=len)   # letter-level tier = the one with more intervals
    if tt.get('phones') is not ph: swapped+=1
    for lab in ph:
        if lab in ('','sil','sp'): continue
        parts=[p.strip() for p in re.split(r'[,،]',lab)]
        if len(parts)==1 and len(parts[0].split())==1: pairs[g][(parts[0],parts[0])]+=1
        elif len(parts)==3 and parts[2] in ('s','d','i','a'):
            c,r,t=parts
            c='<ins>' if t in ('i','a') else c; r='<del>' if t=='d' else r
            pairs[g][(c,r)]+=1; spk_err[c].add(spk) if g=='non-native_speakers' and c!=r else None
        else: odd[lab]+=1
print('files',dict(files),'letter tier not named phones:',swapped)
print('unparsed labels',odd.most_common(30))
T=['ت','ر','ض','ي','ك','ق','خ','غ']
res={}
for g,P in pairs.items():
    res[g]={}
    for L in T:
        row={y:c for (x,y),c in P.items() if x==L}; tot=sum(row.values())
        res[g][L]=dict(total=tot,correct=row.get(L,0),errors=dict(sorted(((y,c) for y,c in row.items() if y!=L),key=lambda t:-t[1])),
                       incoming={x:c for (x,y),c in P.items() if y==L and x!=L},
                       nn_speakers_with_error=len(spk_err[L]) if g=='non-native_speakers' else None)
    print(g, 'all errors:', [(x,y,c) for (x,y),c in P.most_common() if x!=y][:40])
for g in res:
    for L,v in res[g].items(): print(g,L,json.dumps(v,ensure_ascii=False))
json.dump({g:{f'{x}|{y}':c for (x,y),c in P.items()} for g,P in pairs.items()},open('data/l2ksu/textgrid_pairs.json','w'),ensure_ascii=False,indent=0)
json.dump(res,open('data/l2ksu/textgrid_target_confusions.json','w'),ensure_ascii=False,indent=1)
