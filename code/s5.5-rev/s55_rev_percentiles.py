# Percentiles, gaps to best, and near-neighbour counts for named layouts (uses s55_rev.py scoring, lines up to named table)
import json
src=open('s55_rev.py').read(); cut=src.index("named=pd.DataFrame(rows)"); exec(src[:cut])
out={}
for ni,(nm,pos) in enumerate(NAMED.items()):
    v=scs({k:rn[k][ni:ni+1] for k in res},W['primary'])[0]; out[nm]={'SCS':float(v)}
    for sn,m in sets.items():
        s=SC['primary'][m]; n=int(m.sum()); above=s>v+1e-9; rank=int(above.sum()+1)
        d=dict(n=n,rank=rank,top_pct=100*rank/n,best=float(s.max()),gap=float(s.max()-v))
        d['above_within_0.01']=int((above&(s<=v+0.01)).sum()); d['above_within_0.02']=int((above&(s<=v+0.02)).sum())
        if nm.startswith('current') or nm.startswith('redesign'):
            Pm=P[m][above]; refs=[pos.copy()]
            if pos[2] not in MOT: refs=[np.where(np.arange(8)==2,x,pos) for x in (1.5,11.0)]
            diff=np.min([(Pm!=r).sum(1) for r in refs],0)
            d['above_differ_le1_letter']=int((diff<=1).sum()); d['above_differ_le2_letters']=int((diff<=2).sum())
        out[nm][sn]=d
json.dump(out,open(OUT+'s55r_percentiles.json','w'),ensure_ascii=False,indent=1)
for nm,o in out.items():
    print(nm,round(o['SCS'],3))
    for sn in sets: print('   ',sn,{k:(round(x,4) if isinstance(x,float) else x) for k,x in o[sn].items()})
