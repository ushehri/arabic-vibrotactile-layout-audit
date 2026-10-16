"""Stratified alignment audit (no listening). 10 tokens per letter per corpus (seed 52) from the QC'd token tables; Halabi manual phone
boundaries (all tokens) as benchmark. Objective landmark: within onset ±80 ms, the frame of largest rise in 2–8 kHz band energy
(stops ت ض ك ق: burst; fricatives خ: frication onset; غ: largest rise in 0.3–8 kHz energy); ر ي: no reliable landmark (reported n/a).
Metric: landmark − aligned onset (ms). Detector error is calibrated on Halabi manual onsets. Also segment duration vs Halabi consonant duration.
Spectrogram grids per corpus for visual inspection (5 per letter)."""
import numpy as np, pandas as pd, soundfile as sf, librosa, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import stft
LET=['t','r','dd','j','k','q','x','gh']; AR={'t':'ت','r':'ر','dd':'ض','j':'ي','k':'ك','q':'ق','x':'خ','gh':'غ'}
STOP={'t','dd','k','q'}; SR=16000; PAD=0.010
V='data/full52-v2/'; O='results/s5.2-rev/'; F='figures/'
t1=pd.read_csv('data/full52/tokens.csv',keep_default_na=False)
src={'l2ksu':t1[t1.corpus=='l2ksu'],'halabi':t1[t1.corpus=='halabi']}
for c in ['iqra','everyayah','cv']: src[c]=pd.read_csv(V+f'tokens_{c}.csv',keep_default_na=False)
def landmark(y,on,L):
    f,tt,Z=stft(y,SR,nperseg=160,noverlap=80); P=np.abs(Z)**2
    band=(f>=2000)&(f<=8000) if L!='gh' else (f>=300)
    e=10*np.log10(P[band].sum(0)+1e-10); de=np.diff(e); tc=tt[1:]
    w=(tc>=on-0.08)&(tc<=on+0.08)
    if not w.any(): return np.nan
    k=np.where(w)[0][np.argmax(de[w])]; return tc[k]
rows=[]; rng=np.random.default_rng(52); samp={}
for c,d in src.items():
    d=d.reset_index(drop=True)
    for L in LET:
        g=d[d.letter==L]
        s=g if c=='halabi' else g.sample(min(10,len(g)),random_state=52)
        samp[(c,L)]=s
        for _,r in s.iterrows():
            y,_=librosa.load(r.wav,sr=SR); on=float(r.seg_start)
            lm=landmark(y,on,L) if L not in ('r','j') else np.nan
            rows.append(dict(corpus=c,letter=L,utt=r.utt,onset=on,seg_end=float(r.seg_end),dur_ms=1000*(float(r.seg_end)-on),
                             landmark_minus_onset_ms=1000*(lm-on) if lm==lm else np.nan))
A=pd.DataFrame(rows); A.to_csv(O+'s52r_audit_tokens.csv',index=False)
A['abs']=A.landmark_minus_onset_ms.abs()
summ=A.groupby(['corpus']).agg(n=('utt','size'),n_landmark=('abs','count'),median_abs_ms=('abs','median'),
     within20=('abs',lambda v:float((v<=20).mean()) if v.notna().any() else np.nan),within40=('abs',lambda v:float((v<=40).mean()) if v.notna().any() else np.nan),
     median_dur_ms=('dur_ms','median')).reset_index()
byL=A.groupby(['corpus','letter']).agg(median_abs_ms=('abs','median'),median_signed_ms=('landmark_minus_onset_ms','median'),median_dur_ms=('dur_ms','median')).reset_index()
summ.to_csv(O+'s52r_audit_summary.csv',index=False); byL.to_csv(O+'s52r_audit_by_letter.csv',index=False)
print(summ.to_string()); print(byL.pivot(index='letter',columns='corpus',values='median_abs_ms').round(1).to_string())
print(byL.pivot(index='letter',columns='corpus',values='median_dur_ms').round(0).to_string())
# spectrogram grids
for c in ['l2ksu','iqra','everyayah','cv','halabi']:
    fig,axs=plt.subplots(8,5,figsize=(12,14))
    for i,L in enumerate(LET):
        s=samp[(c,L)].head(5) if c!='halabi' else samp[(c,L)].sample(min(5,len(samp[(c,L)])),random_state=1)
        for j in range(5):
            a=axs[i,j]; a.set_xticks([]); a.set_yticks([])
            if j>=len(s): a.axis('off'); continue
            r=s.iloc[j]; y,_=librosa.load(r.wav,sr=SR); on,en=float(r.seg_start),float(r.seg_end)
            a0=max(0,on-0.1); seg=y[int(a0*SR):int((en+0.1)*SR)]
            if len(seg)<400: a.axis('off'); continue
            a.specgram(seg,NFFT=256,Fs=SR,noverlap=200,cmap='Greys'); a.axvline(on-a0,c='r'); a.axvline(en-a0,c='b'); a.set_ylim(0,8000)
            if j==0: a.set_ylabel(f'/{L}/',rotation=0,labelpad=15)
    fig.suptitle(f'{c}: aligned onset (red) and segment end (blue); 5 random tokens per letter',fontsize=10)
    plt.tight_layout(); plt.savefig(F+f's52r_audit_spectrograms_{c}.png',dpi=80); plt.close()
print('DONE')
