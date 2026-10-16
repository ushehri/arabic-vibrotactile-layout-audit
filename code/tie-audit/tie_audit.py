# Audit: does quantising to 1e-9 change any rank vector used in 5.1/5.2/5.4 (sleeve cm distances, codes, PanPhon)?
import numpy as np, itertools
from scipy.stats import rankdata
q=lambda v:np.round(np.asarray(v,float),9); iu=np.triu_indices(8,1)
POS=lambda dd:np.array([1.5,1.5,dd,11.0,15.0,18.0,20.5,23.0])
codes={'KAPD place':[0,0,0,1,2,3,3,3],'modern lit':[0,0,0,1,2,3,2.5,2.5],'classical':[0,0,1,2,3,4,5,5],
 'B1 linear place (5.5)':1.5+np.array([0,0,0,1,2,3,3,3])/3*21.5,'B3 linear classical (5.5)':1.5+np.array([0,0,1,2,3,4,5,5])/5*21.5}
out=[]
for dd in (6.25,11.0,1.5):
    p=POS(dd); d=np.abs(p[:,None]-p[None,:])[iu]; out.append((f'sleeve distances ض={dd}',bool((rankdata(d)==rankdata(q(d))).all())))
    out.append((f'sleeve positions ض={dd}',bool((rankdata(p)==rankdata(q(p))).all())))
for k,c in codes.items():
    c=np.asarray(c,float); d=np.abs(c[:,None]-c[None,:])[iu]
    out.append((k+' positions',bool((rankdata(c)==rankdata(q(c))).all()))); out.append((k+' pair distances',bool((rankdata(d)==rankdata(q(d))).all())))
for k,v in out: print(f'{k:45s} unchanged={v}')
