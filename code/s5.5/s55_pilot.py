# Section 5.5 PILOT: layout scoring. Inputs: s54/s54_pairs.csv (feature_dist=PanPhon from 5.4,
# mfcc=pooled MFCC euclid from full52-v2 as carried into 5.4, sym_rate=L2 confusion).
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, workdir
import csv, itertools, json, random
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B=data()+"/"
import os; os.chdir(workdir("s55","pilot"))   # outputs go to this working folder
L=["ت","ر","ض","ي","ك","ق","خ","غ"]; SLOTS=[1.5,1.5,11.0,11.0,15.0,18.0,20.5,23.0]
idx={c:i for i,c in enumerate(L)}
F=np.zeros((8,8));M=np.zeros((8,8));C=np.zeros((8,8))
for r in csv.DictReader(open(B+"s54/s54_pairs.csv")):
    a,b=r["pair"].split("–");i,j=idx[a],idx[b]
    for X,k in((F,"feature_dist"),(M,"mfcc"),(C,"sym_rate")):X[i,j]=X[j,i]=float(r[k])
# place ordinal from KAPD labels in articulatory-comparison.md: alveolar 0, palatal 1, velar 2, uvular 3
PL={"ت":0,"ر":0,"ض":0,"ي":1,"ك":2,"ق":3,"خ":3,"غ":3}
P=np.array([[abs(PL[a]-PL[b]) for b in L] for a in L],float)
iu=np.triu_indices(8,1)
TGT={"artic_panphon":F[iu],"artic_place":P[iu],"confusion":-C[iu],"acoustic_mfcc":M[iu]}  # all as dissimilarity
def sleeve(pos): p=np.array(pos);return np.abs(p[:,None]-p[None,:])[iu]
def nz(x): x=np.asarray(x,float);r=x.max()-x.min();return (x-x.min())/r if r else x*0
def score(pos):
    d=sleeve(pos);o={}
    for k,t in TGT.items():
        o["rho_"+k]=spearmanr(d,t).correlation
        o["qap_"+k]=float(np.sum((nz(d)-nz(t))**2))  # lower=better
    return o
def lay(order): # order: letters in slot order
    pos=[0]*8
    for s,c in zip(SLOTS,order):pos[idx[c]]=s
    return pos
random.seed(55)
ALT={"A0 current":"ت ر ض ي ك ق خ غ",
 "A1 ض push->M1 (sens.)":None,
 "A2 swap ك/ق":"ت ر ض ي ق ك خ غ",
 "A3 uvulars grouped (ك,ق,خ,غ order kept, ي before ك)":"ت ر ض ي ك ق خ غ",
 "A4 modern-artic: ي ك mid, ق خ غ back, ق last":"ت ر ض ي ك خ غ ق",
 "A5 ض at tip, ر to M2":"ت ض ر ي ك ق خ غ",
 "A6 ض at tip + ق last":"ت ض ر ي ك خ غ ق",
 "A7 ي/ك share M2, ض to M3":"ت ر ي ك ض ق خ غ",
 "A8 reverse (control)":"غ خ ق ك ي ض ر ت",
 "A9 voicing-split back: غ M4, ق M6":"ت ر ض ي ك غ خ ق"}
del ALT["A3 uvulars grouped (ك,ق,خ,غ order kept, ي before ك)"]  # identical to A0
for n in range(3): o=L[:];random.shuffle(o);ALT[f"R{n+1} random"]=" ".join(o)
rows=[]
for name,o in ALT.items():
    if o is None: pos=lay(L);pos[idx["ض"]]=1.5
    else: pos=lay(o.split())
    s=score(pos);s["layout"]=name;s["slot_order"]=o or "ت ر ض(1.5) ي ك ق خ غ";rows.append(s)
W={"0.6/0.3/0.1":(.6,.3,.1),"0.8/0.15/0.05":(.8,.15,.05),"0.4/0.4/0.2":(.4,.4,.2),"0.34/0.33/0.33":(1/3,1/3,1/3)}
for r in rows:
    for wn,(a,b,c) in W.items(): r["comb_"+wn]=a*r["rho_artic_panphon"]+b*r["rho_confusion"]+c*r["rho_acoustic_mfcc"]
# full enumeration, distinct layouts (8!/(2!2!))
seen={}
for perm in itertools.permutations(L):
    key=(frozenset(perm[:2]),frozenset(perm[2:4]))+tuple(perm[4:])
    if key in seen: continue
    pos=lay(perm);d=sleeve(pos)
    seen[key]=(spearmanr(d,TGT["artic_panphon"]).correlation,spearmanr(d,TGT["artic_place"]).correlation,
               a:=0.6*spearmanr(d,TGT["artic_panphon"]).correlation+0.3*spearmanr(d,TGT["confusion"]).correlation+0.1*spearmanr(d,TGT["acoustic_mfcc"]).correlation, perm)
V=list(seen.values());n=len(V)
def rank(col,val): arr=np.array([v[col] for v in V]);return int((arr>val+1e-12).sum()+1),float((arr<=val+1e-12).mean())
cur=rows[0]
enum={"n_distinct":n,"n_raw":40320}
for col,k in((0,"rho_artic_panphon"),(1,"rho_artic_place"),(2,"comb_0.6/0.3/0.1")):
    rk,pct=rank(col,cur[k]);best=max(V,key=lambda v:v[col])
    enum[k]={"current":cur[k],"rank":rk,"percentile":pct,"best":best[col],"best_slot_order":" ".join(best[3]),
             "n_ties_best":int(sum(abs(v[col]-best[col])<1e-12 for v in V))}
json.dump({"rows":rows,"enumeration":enum,"weights":W},open("s55_pilot_results.json","w"),ensure_ascii=False,indent=1)
with open("s55_pilot_table.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
# figure
fig,ax=plt.subplots(1,2,figsize=(13,5.5))
names=[r["layout"] for r in rows];y=np.arange(len(rows))
for k,(lab,mk) in {"rho_artic_panphon":("PanPhon","o"),"rho_artic_place":("Place","s"),"rho_confusion":("L2 confusion","^"),"rho_acoustic_mfcc":("MFCC","x"),"comb_0.6/0.3/0.1":("Combined .6/.3/.1","D")}.items():
    ax[0].scatter([r[k] for r in rows],y,label=lab,marker=mk)
ax[0].set_yticks(y);ax[0].set_yticklabels(names,fontsize=8);ax[0].invert_yaxis();ax[0].axvline(0,c="k",lw=.5)
ax[0].set_xlabel("Spearman ρ (sleeve distance vs dissimilarity; higher = mirrors better)");ax[0].legend(fontsize=7);ax[0].set_title("Pilot layouts")
ax[1].hist([v[0] for v in V],bins=40,color="0.7");ax[1].axvline(cur["rho_artic_panphon"],c="r",label=f"current (rank {enum['rho_artic_panphon']['rank']}/{n})")
ax[1].set_xlabel("ρ PanPhon, all distinct layouts");ax[1].legend();ax[1].set_title(f"Enumeration (n={n})")
plt.tight_layout();plt.savefig("s55_pilot_fig.png",dpi=150)
print(json.dumps(enum,ensure_ascii=False,indent=1))
for r in rows: print(f"{r['layout'][:34]:34s} pan={r['rho_artic_panphon']:+.3f} place={r['rho_artic_place']:+.3f} conf={r['rho_confusion']:+.3f} mfcc={r['rho_acoustic_mfcc']:+.3f} qapP={r['qap_artic_panphon']:.2f} "+" ".join(f"{r['comb_'+w]:+.3f}" for w in W)+"  "+r["slot_order"])
