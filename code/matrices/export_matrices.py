"""Export the complete 8x8 letter matrices used as ranking targets and in the pairwise tests, as CSV (one file per matrix).
No new analysis: every matrix is read from, or rebuilt with the code of, the stored analysis outputs, and checked against them.
Letter order (rows = columns): ت ر ض ي ك ق خ غ (/t r dˤ j k q x ɣ/).
L2-KSU (CC BY-NC-ND 4.0): no matrix computed from L2-KSU audio is exported (L2-KSU native P/W matrices and the ALL pool that includes them).
The learner confusion counts are exported only because they are the aggregate counts already given in Table S10 (s54r_pairs.csv, s54r_per_letter.csv)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import RESULTS, data, results_dir
import os, pickle, json, numpy as np, pandas as pd
R=str(RESULTS)+os.sep; OUT=results_dir('matrices')
HERE=os.path.dirname(os.path.abspath(__file__))
T=['ت','ر','ض','ي','ك','ق','خ','غ']; IPA=['t','r','dˤ','j','k','q','x','ɣ']; LAB=[f'{a} /{b}/' for a,b in zip(T,IPA)]
LET=['t','r','dd','j','k','q','x','gh']   # acoustic pipeline letter codes, same order
iu=np.triu_indices(8,1); files={}
def save(name,M,desc,fmt='%.6f'):
    df=pd.DataFrame(np.asarray(M,float),index=LAB,columns=LAB); df.index.name='letter'
    df.to_csv(OUT+name+'.csv',float_format=fmt,encoding='utf-8'); files[name]=desc; return df
# ---- articulatory / phonological (ranking targets: targets.py) ----
TGT=os.path.join(HERE,'..','s5.5-rev','targets.py'); g={'__file__':TGT}; exec(open(TGT,encoding='utf-8').read(),g)
FEAT,PLACE,BLENDv,CONF_rank=g['FEAT'],g['PLACE'],g['BLEND'],g['CONF']
B=np.zeros((8,8)); B[iu]=BLENDv; B=B+B.T
save('artic_panphon_feature_distance',FEAT,'PanPhon: number of differing features among 24 (ي coded /j/). Component of the blend ranking target; control in the learner partial Mantel tests (s5.5-rev/targets.py; s5.4-rev/s54_rev.py).','%g')
save('artic_kapd_place_absdiff',PLACE,'Place: |c_i - c_j| on the KAPD place code (ت ر ض 0, ي 1, ك 2, ق خ غ 3). Component of the blend ranking target (targets.py).','%g')
save('artic_blend_target',B,'Blend ranking target: 0.5*rn(PanPhon) + 0.5*rn(Place), rn(v) = (rank(v)-1)/27 over the 28 pairs (ties averaged); diagonal set to 0 (not used). Weight 0.6 in SCS (targets.py, s55_rev.py).')
# order-test codes (s51_rev.py EV) and their pairwise |difference| form
CODES={'KAPD place label':[0,0,0,1,2,3,3,3],'Modern-literature code':[0,0,0,1,2,3,2.5,2.5],'Classical code (source of layout; circular)':[0,0,1,2,3,4,5,5]}
epg=pd.read_csv(R+'s5.1-rev/s51r_epg_table.csv').set_index('letter').mean_pos.reindex(T).to_numpy(float); CODES['EPG mean contact position (final extraction)']=list(epg)
assert CODES['KAPD place label']==[g['PL'][c] for c in T]
cv=pd.DataFrame(CODES,index=LAB); cv.index.name='letter'; cv.to_csv(OUT+'artic_order_codes.csv',float_format='%.6f',encoding='utf-8')
files['artic_order_codes']='Front-to-back codes per letter used in the order tests of Table 4 / Table S1 (s5.1-rev/s51_rev.py) and, for EPG, in baseline B2 (s5.5-rev/s55_rev.py). EPG values from s5.1-rev/s51r_epg_table.csv.'
ad=lambda v:np.abs(np.subtract.outer(np.asarray(v,float),np.asarray(v,float)))
save('artic_modern_code_absdiff',ad(CODES['Modern-literature code']),'Pairwise |difference| of the modern-literature code. Pairwise form for completeness: the code itself enters only the order tests (8 positions), not a pairwise test or ranking target.','%g')
save('artic_classical_code_absdiff',ad(CODES['Classical code (source of layout; circular)']),'Pairwise |difference| of the classical code. Pairwise form for completeness (order tests and baseline B3 only).','%g')
save('artic_epg_meanpos_absdiff',ad(epg),'Pairwise |difference| of the EPG mean contact position (final extraction). Pairwise form for completeness (order tests and baseline B2 only).')
# ---- learner confusion (aggregate counts already in Table S10) ----
pl=pd.read_csv(R+'s5.4-rev/s54r_per_letter.csv').set_index('letter').reindex(T); pr=pd.read_csv(R+'s5.4-rev/s54r_pairs.csv')
C=np.zeros((8,8)); np.fill_diagonal(C,pl.correct.values)
for r in pr.itertuples():
    a,b=r.pair.split('–'); C[T.index(a),T.index(b)]=r.a_as_b; C[T.index(b),T.index(a)]=r.b_as_a
assert np.array_equal(C.sum(1)-np.diag(C),pl.within8_sub.values), 'within-set substitutions do not match Table S10'
cc=pd.DataFrame(C.astype(int),index=LAB,columns=[f'realised {l}' for l in LAB]); cc.index.name='canonical letter'
cc['substitution outside the eight']=pl.offtarget_sub.values; cc['deletion']=pl.deletions.values; cc['N tokens']=pl.N.values
assert (cc.iloc[:,:8].sum(axis=1)+cc['substitution outside the eight']+cc['deletion']==cc['N tokens']).all()
cc.to_csv(OUT+'learner_confusion_counts.csv',encoding='utf-8')
files['learner_confusion_counts']='L2-KSU learners (40; 9,766 tokens): rows = canonical letter, columns = realised letter among the eight (diagonal = correct), plus substitutions outside the eight, deletions and N. Aggregate counts identical to Table S10 (s5.4-rev/s54r_pairs.csv, s54r_per_letter.csv; s54_rev.py).'
N=pl.N.values.astype(float); Rm=C/N[:,None]; np.fill_diagonal(Rm,0); S=(Rm+Rm.T)/2*100
chk={tuple(p.split('–')):v for p,v in zip(pr.pair,pr.S_pct)}
for (a,b),v in chk.items(): assert abs(S[T.index(a),T.index(b)]-v)<1e-9
save('learner_confusion_symmetrised_pct',S,'Symmetrised confusion S_ij = (C_ij/N_i + C_ji/N_j)/2, in %, from the counts above; the matrix tested against sleeve distance in Table 7 / Table S9 (s54_rev.py). Its negation is the confusion ranking target (weight 0.3 in SCS).')
Srank=CONF_rank*100; np.fill_diagonal(Srank,0)
note_rank=dict(max_abs_diff_pct=float(np.abs(Srank-S).max()),same_pair_order=bool((np.argsort(np.argsort(Srank[iu],kind='stable'))==np.argsort(np.argsort(S[iu],kind='stable'))).all()))
# ---- native acoustic matrices: non-L2-KSU corpora only ----
PB=pickle.load(open(data('full52-v2/pooled_boots.pkl'),'rb')); MV=pd.read_csv(data('full52-v2/matrices_v2.csv'))
def mv(src,est,meas):
    X=np.full((8,8),np.nan); I={l:i for i,l in enumerate(LET)}
    for r in MV[(MV.source==src)&(MV.estimator==est)&(MV.measure==meas)].itertuples(): X[I[r.a],I[r.b]]=r.d
    return X
SRC={'iqra':'Iqra_train dev','everyayah':'EveryAyah','cv':'Common Voice 26.0','halabi':'Arabic Speech Corpus (Halabi; single-speaker reference)'}
for meas,mn in [('mfcc_euclid','MFCC Euclidean'),('ssl_L23','XLS-R layer 23 cosine')]:
    for s,sn in SRC.items():
        M=PB[s][0][meas]; assert np.allclose(M,mv(s,'P',meas),atol=1e-5,equal_nan=True)
        save(f'acoustic_{s}_P_{meas}',M,f'{sn}: pooled-token (P) letter distance matrix, {mn}; diagonal = mean within-letter distance (full-5.2-v2/analysis_v2.py -> pooled_boots.pkl). Full float64 precision (%.17g). Used in Table 6 / Table S5.','%.17g')
    M=np.mean([PB[s][0][meas] for s in ['iqra','everyayah','cv']],0)
    save(f'acoustic_ALL_without_L2KSU_P_{meas}',M,f'ALL without L2-KSU: equal-weight mean of the Iqra, EveryAyah and Common Voice P matrices, {mn} (s5.2-rev/s52_rev_mantel.py). Full float64 precision (%.17g). Used in Table 6 / Table S6.','%.17g')
lg=MV[MV.source!='l2ksu'].copy(); lg.to_csv(OUT+'acoustic_all_measures_non_L2KSU_long.csv',index=False,encoding='utf-8')
files['acoustic_all_measures_non_L2KSU_long']='Long format (source, estimator P/W, measure, letter a, letter b, distance d, bootstrap lo/hi) of every stored acoustic matrix of the non-L2-KSU corpora: MFCC Euclidean, 30-token MFCC DTW and XLS-R layers 0-24 (copy of the non-L2-KSU rows of full52-v2/matrices_v2.csv written by full-5.2-v2/report_v2.py; values rounded to 5 decimals there). Letter codes t r dd j k q x gh.'
# ---- sleeve distances of the current layout ----
for nm,dd in [('midpoint_6.25_primary',6.25),('start_11.0',11.0),('end_1.5_RD1',1.5)]:
    p=np.array([1.5,1.5,dd,11,15,18,20.5,23]); save(f'sleeve_distance_current_dad_{nm}',ad(p),f'Sleeve distance |pos_i - pos_j| (cm) of the current layout with ض at {dd:g} cm ({"primary" if dd==6.25 else "sensitivity"}{"; = RD1 positions" if dd==1.5 else ""}).','%g')
excluded={'L2-KSU native acoustic matrices (P and W; MFCC, DTW, XLS-R layers)':'computed from L2-KSU audio (CC BY-NC-ND 4.0); not released',
 'ALL (4 sources) acoustic matrices, including the MFCC ranking target':'average includes the L2-KSU native matrix; not released',
 'L2-KSU-derived token-level data':'never released'}
json.dump(dict(files=files,excluded=excluded,confusion_ranking_target_vs_tested=note_rank),open(OUT+'manifest.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(note_rank)); print(len(files),'files')
