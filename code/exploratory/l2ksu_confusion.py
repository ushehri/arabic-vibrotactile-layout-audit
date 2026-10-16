import pandas as pd, collections, json, re
df=pd.read_csv('data/l2ksu/annotations.csv')
df['grp']=df.path.str.extract(r'dataset/([^/]+)/')[0]
df['sent']=df.path.str.extract(r'/(sentence\d+)/')[0]
DEL='<del>'; INS='<ins>'
def align(a,b):
    n,m=len(a),len(b); D=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0]=i
    for j in range(m+1): D[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i][j]=min(D[i-1][j-1]+(a[i-1]!=b[j-1]),D[i-1][j]+1,D[i][j-1]+1)
    i,j=n,m; out=[]
    while i>0 or j>0:
        if i>0 and j>0 and D[i][j]==D[i-1][j-1]+(a[i-1]!=b[j-1]): out.append((a[i-1],b[j-1])); i-=1;j-=1
        elif i>0 and D[i][j]==D[i-1][j]+1: out.append((a[i-1],DEL)); i-=1
        else: out.append((INS,b[j-1])); j-=1
    return out[::-1]
pairs=collections.Counter(); pairs_by=collections.defaultdict(collections.Counter)
mismatch_pos=0
for _,r in df.iterrows():
    a,b=r.canon.split(),r.ans.split()
    al=align(a,b)
    if len(a)==len(b):
        pos=list(zip(a,b))
        if pos!=al: mismatch_pos+=1
    for x,y in al:
        pairs[(x,y)]+=1; pairs_by[r.grp][(x,y)]+=1
print('equal-length rows where edit alignment != positional:',mismatch_pos)
T=[('ت','/t/'),('ر','/r/'),('ض','/dˤ/'),('ي','/j/'),('ك','/k/'),('ق','/q/'),('خ','/x/'),('غ','/ɣ/')]
res={}
for L,ipa in T:
    row={y:c for (x,y),c in pairs.items() if x==L}
    tot=sum(row.values()); ok=row.get(L,0)
    nn={y:c for (x,y),c in pairs_by['non-native_speakers'].items() if x==L}
    nat={y:c for (x,y),c in pairs_by['native_speakers'].items() if x==L}
    res[L]=dict(ipa=ipa,total=tot,correct=ok,errors={y:c for y,c in sorted(row.items(),key=lambda t:-t[1]) if y!=L},
                nn_total=sum(nn.values()),nn_correct=nn.get(L,0),nat_total=sum(nat.values()),nat_correct=nat.get(L,0))
    # what canon letters got realised as L (incoming)
    res[L]['incoming']={x:c for (x,y),c in pairs.items() if y==L and x!=L}
    # sentences containing L
    res[L]['sentences']=sorted({s for s,c in zip(df.sent,df.canon) if L in c.split()})
    res[L]['speakers_nn']=df[(df.grp=='non-native_speakers')&df.canon.apply(lambda c:L in c.split())].path.str.extract(r'(non-native_speakers/[^/]+/speaker\d+)')[0].nunique()
for L,v in res.items(): print(L,json.dumps(v,ensure_ascii=False))
# all errors overall
print('ALL ERRORS:'); print([(x,y,c) for (x,y),c in pairs.most_common() if x!=y])
json.dump({k:v for k,v in res.items()},open('data/l2ksu/target_confusions.json','w'),ensure_ascii=False,indent=1)
json.dump([[x,y,c] for (x,y),c in pairs.most_common()],open('data/l2ksu/all_pairs.json','w'),ensure_ascii=False)
