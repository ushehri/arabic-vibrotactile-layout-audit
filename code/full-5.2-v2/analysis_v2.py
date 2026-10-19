"""Stage 3: per-source and pooled 8x8 distance matrices, CIs, Mantel tests, word-confound checks.
Sources: l2ksu, halabi (features from full52, unchanged), iqra, everyayah, cv (+ any new tokens_<src>.csv).
Estimators
  W (within-unit, as in full52): letter matrix per unit having all 8 letters, averaged over units;
    95% CI = percentile bootstrap over units (1,000). Units = speakers (L2-KSU, EveryAyah reciters, CV client_id).
  P (pooled tokens): mean pairwise token distance over all tokens of the source; 95% CI = cluster bootstrap
    over units (500). Iqra has no speaker IDs -> unit = utterance; only P is possible for Iqra.
    SSL cosine is computed exactly from per-(unit,letter) sums of unit-normalised vectors; MFCC Euclidean
    uses <=300 random tokens per letter per replicate.
  ALL = equal-weight mean of the P matrices of l2ksu, iqra, everyayah, cv (replicate-wise for CI).
Mantel: Spearman over 28 cells, 9,999 permutations, one-sided (r > 0).
Word confound: (a) word-level resampling (bootstrap word types within letter, <=5 random tokens per drawn type, so frequent words no longer dominate);
(b) linear mixed models on token-level contrasts with random intercept for word (see wordmodel)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, workdir
import sys, pickle, json, collections, glob
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'full-5.2'))
from analysis import LET, AR, IPA, POS, sleeve, mantel, IU, dtw_d
import os
S = sleeve(POS); NB_W, NB_P, NB_WORD = (1000, 500, 500) if not os.environ.get('QUICK') else (5, 5, 5); L_PRE = 23
F1 = data('full52')
rng = np.random.default_rng(0)

def load():
    src = {}
    t = pd.read_csv(f'{F1}/tokens.csv', keep_default_na=False); mf = pickle.load(open(f'{F1}/feats_mfcc.pkl', 'rb'))
    ssl = np.load(f'{F1}/feats_ssl.npy', mmap_mode='r')
    for c in ['l2ksu', 'halabi']:
        ix = np.where((t.corpus == c) & (t.n_mfcc_frames.astype(int) >= 3))[0]
        d = t.iloc[ix].copy(); d['unit'] = d.speaker; d['source'] = c
        src[c] = dict(tok=d.reset_index(drop=True), mf=[mf[i] for i in ix], ssl=ssl, ssl_ix=ix)
    for f in sorted(glob.glob('tokens_*.csv')):
        c = f[7:-4]; d = pd.read_csv(f, keep_default_na=False, dtype={'unit': str})
        mf = pickle.load(open(f'feats_mfcc_{c}.pkl', 'rb')); ssl = np.load(f'feats_ssl_{c}.npy', mmap_mode='r')
        ix = np.where(d.n_mfcc_frames.astype(int) >= 3)[0]
        src[c] = dict(tok=d.iloc[ix].reset_index(drop=True), mf=[mf[i] for i in ix], ssl=ssl, ssl_ix=ix)
    for c, s in src.items():
        s['tok']['li'] = s['tok'].letter.map({l: i for i, l in enumerate(LET)})
        s['mmean'] = np.array([m.mean(0) for m in s['mf']], dtype=np.float32)
        s['units'] = sorted(s['tok'].unit.unique()); um = {u: k for k, u in enumerate(s['units'])}
        s['ui'] = s['tok'].unit.map(um).values
        # per-(unit,letter) sums of unit-norm SSL vectors, chunked over tokens
        U = len(s['units']); Ssum = np.zeros((U, 8, 25, 1024), np.float32); N = np.zeros((U, 8))
        li = s['tok'].li.values; ix = s['ssl_ix']
        for a in range(0, len(ix), 4000):
            E = np.asarray(s['ssl'][ix[a:a + 4000]], np.float32); E /= np.linalg.norm(E, axis=2, keepdims=True) + 1e-9
            np.add.at(Ssum, (s['ui'][a:a + 4000], li[a:a + 4000]), E)
        np.add.at(N, (s['ui'], li), 1)
        s['Ssum'], s['N'] = Ssum, N
        print(c, len(s['tok']), 'tokens', U, 'units', flush=True)
    return src

def cos_from_sums(T, n):
    """T: (8, L, D) summed unit vectors, n: (8,) -> (L, 8, 8) mean pairwise cosine distance."""
    G = np.einsum('ild,jld->lij', T, T)
    nn = np.outer(n, n); M = 1 - G / np.where(nn > 0, nn, np.nan)
    d = np.diagonal(G, axis1=1, axis2=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        w = 1 - (d - n) / (n * (n - 1))
    for i in range(8): M[:, i, i] = np.where(n[i] >= 2, w[:, i], np.nan)
    return M

def euc_matrix(X, li, cap=None, r=None):
    G = [np.where(li == i)[0] for i in range(8)]
    if cap: G = [g if len(g) <= cap else r.choice(g, cap, replace=False) for g in G]
    M = np.full((8, 8), np.nan)
    for i in range(8):
        for j in range(i, 8):
            if len(G[i]) and len(G[j]):
                D = cdist(X[G[i]], X[G[j]])
                M[i, j] = M[j, i] = (D[np.triu_indices(len(G[i]), 1)].mean() if len(G[i]) > 1 else np.nan) if i == j else D.mean()
    return M

def mats_P(s, w):
    """pooled matrices for unit weights w (counts per unit). Returns dict measure->(8,8)."""
    T = np.tensordot(w, s['Ssum'], 1); n = w @ s['N']
    C = cos_from_sums(T, n)
    keep = np.repeat(w, 1)[s['ui']] > 0
    ix = np.where(keep)[0]; rep = w[s['ui'][ix]].astype(int); ix = np.repeat(ix, rep)
    out = {'mfcc_euclid': euc_matrix(s['mmean'][ix], s['tok'].li.values[ix], 300, rng)}
    for L in range(25): out[f'ssl_L{L:02d}'] = C[L]
    return out

def mats_W(s):
    full = np.where((s['N'] >= 1).all(1))[0]; per = collections.defaultdict(list)
    for u in full:
        sel = s['ui'] == u
        per['mfcc_euclid'].append(euc_matrix(s['mmean'][sel], s['tok'].li.values[sel]))
        C = cos_from_sums(s['Ssum'][u], s['N'][u])
        for L in range(25): per[f'ssl_L{L:02d}'].append(C[L])
    return full, {k: np.stack(v) for k, v in per.items()}

def summ(mean, boots):
    r, p = mantel(mean, S)
    rb = np.array([spearmanr(b[IU], S[IU])[0] for b in boots]) if boots is not None else None
    d = dict(mean=np.round(mean, 5).tolist(), mantel_r=r, mantel_p=p)
    if boots is not None:
        d.update(lo=np.round(np.nanpercentile(boots, 2.5, 0), 5).tolist(), hi=np.round(np.nanpercentile(boots, 97.5, 0), 5).tolist(),
                 r_ci=np.nanpercentile(rb, [2.5, 97.5]).tolist())
    return d

def key_pairs(M):
    off = sorted(M[IU]); rank = lambda v: int(np.searchsorted(off, v) + 1)
    i = {l: k for k, l in enumerate(LET)}
    return {p: dict(d=float(M[i[a], i[b]]), rank_of_28=rank(M[i[a], i[b]])) for p, (a, b) in
            {'q-x': ('q', 'x'), 'k-q': ('k', 'q'), 't-k': ('t', 'k')}.items()}

def main():
    src = load(); R = {'sources': {}}; Pboots = {}
    for c, s in src.items():
        t = s['tok']; r = {'n_tokens': {l: int((t.letter == l).sum()) for l in LET}, 'n_units': len(s['units']),
                           'word_types': {l: int(t[t.letter == l].word.nunique()) for l in LET}}
        U = len(s['units']); w1 = np.ones(U)
        P0 = mats_P(s, w1)
        if U > 1:
            bs = collections.defaultdict(list)
            for b in range(NB_P):
                for k, v in mats_P(s, np.bincount(rng.integers(0, U, U), minlength=U).astype(float)).items(): bs[k].append(v)
            bs = {k: np.stack(v) for k, v in bs.items()}
        else: bs = {k: None for k in P0}
        Pboots[c] = (P0, bs)
        r['P'] = {k: summ(P0[k], bs[k]) for k in P0}
        # DTW point estimate, 30 tokens per letter
        G = [rng.permutation(np.where(t.li.values == i)[0])[:30] for i in range(8)]
        Md = np.full((8, 8), np.nan)
        for i in range(8):
            for j in range(i, 8):
                D = dtw_d([s['mf'][a] for a in G[i]], [s['mf'][b] for b in G[j]])
                Md[i, j] = Md[j, i] = D[np.triu_indices(len(G[i]), 1)].mean() if i == j else D.mean()
        r['P']['mfcc_dtw_30tok'] = summ(Md, None)
        if c not in ('iqra', 'halabi'):
            full, W = mats_W(s); r['W_n_units_all8'] = int(len(full))
            if len(full) >= 2:
                r['W'] = {}
                for k, st in W.items():
                    bb = np.stack([np.nanmean(st[rng.integers(0, len(st), len(st))], 0) for _ in range(NB_W)])
                    r['W'][k] = summ(np.nanmean(st, 0), bb)
                    r['W'][k]['per_unit_r_median'] = float(np.nanmedian([spearmanr(m[IU], S[IU])[0] for m in st]))
        for est in ('P', 'W'):
            if est in r:
                r[est]['ssl_best_layer_posthoc'] = int(np.argmax([r[est][f'ssl_L{L:02d}']['mantel_r'] for L in range(25)]))
                r[est]['key_pairs'] = {k: key_pairs(np.array(r[est][k]['mean'])) for k in ['mfcc_euclid', f'ssl_L{L_PRE:02d}']}
        R['sources'][c] = r; print(c, 'done', flush=True)
    # ALL: equal-weight average of P over the four multi-speaker/native sources
    A = [c for c in ['l2ksu', 'iqra', 'everyayah', 'cv'] if c in Pboots]
    R['ALL'] = {'sources': A}
    for k in Pboots[A[0]][0]:
        M = np.mean([Pboots[c][0][k] for c in A], 0); B = np.mean([Pboots[c][1][k] for c in A], 0)
        R['ALL'][k] = summ(M, B)
    R['ALL']['ssl_best_layer_posthoc'] = int(np.argmax([R['ALL'][f'ssl_L{L:02d}']['mantel_r'] for L in range(25)]))
    R['ALL']['key_pairs'] = {k: key_pairs(np.array(R['ALL'][k]['mean'])) for k in ['mfcc_euclid', f'ssl_L{L_PRE:02d}']}
    R['word'] = word_checks(src, A)
    json.dump(R, open('results_v2.json', 'w'), ensure_ascii=False, indent=1)
    pickle.dump({c: (Pboots[c][0], {k: (v if v is None else v.astype(np.float32)) for k, v in Pboots[c][1].items()}) for c in Pboots},
                open('pooled_boots.pkl', 'wb'))
    print('DONE', flush=True)

def layer_vecs(s, L):
    E = np.asarray(s['ssl'][s['ssl_ix'], L], np.float32); return E / np.linalg.norm(E, axis=1, keepdims=True)

def word_checks(src, A):
    import statsmodels.formula.api as smf
    out = {}
    for feat in ['mfcc_euclid', f'ssl_L{L_PRE:02d}']:
        res = {}
        for c in A + ['ALL']:
            cs = A if c == 'ALL' else [c]
            X = {cc: (src[cc]['mmean'] if feat == 'mfcc_euclid' else layer_vecs(src[cc], L_PRE)) for cc in cs}
            dist = (lambda a, b: cdist(a, b)) if feat == 'mfcc_euclid' else (lambda a, b: cdist(a, b, 'cosine'))
            reps, rs = [], []
            for b in range(NB_WORD if c != 'ALL' else 200):
                Ms = []
                for cc in cs:
                    t = src[cc]['tok']; G = []
                    for l in LET:
                        g = t[t.letter == l]; wt = g.word.unique()
                        drawn = rng.choice(wt, len(wt)); byw = g.groupby('word').indices
                        G.append(np.concatenate([g.index[rng.choice(byw[w], min(5, len(byw[w])), replace=False)] for w in drawn]))  # <=5 tokens per drawn word type
                        if len(G[-1]) > 400: G[-1] = rng.choice(G[-1], 400, replace=False)  # cap 400 tokens/letter/replicate (compute)
                    M = np.zeros((8, 8))
                    for i in range(8):
                        for j in range(i, 8):
                            D = dist(X[cc][G[i]], X[cc][G[j]])
                            M[i, j] = M[j, i] = D[~np.eye(len(G[i]), dtype=bool)].mean() if i == j else D.mean()
                    Ms.append(M)
                M = np.mean(Ms, 0); reps.append(M); rs.append(spearmanr(M[IU], S[IU])[0])
            reps = np.stack(reps); kp = [key_pairs(m) for m in reps]
            res[c] = dict(mean=np.round(reps.mean(0), 5).tolist(), r_mean=float(np.mean(rs)), r_ci=np.percentile(rs, [2.5, 97.5]).tolist(),
                          frac_r_gt0=float(np.mean(np.array(rs) > 0)),
                          key={p: dict(d_ci=np.percentile([k[p]['d'] for k in kp], [2.5, 50, 97.5]).tolist(),
                                       rank_ci=np.percentile([k[p]['rank_of_28'] for k in kp], [2.5, 50, 97.5]).tolist()) for p in kp[0]},
                          qx_lt_kq=float(np.mean([k['q-x']['d'] < k['k-q']['d'] for k in kp])),
                          tk_lt_kq=float(np.mean([k['t-k']['d'] < k['k-q']['d'] for k in kp])))
        # mixed models: token-level contrasts d(token, centroid of near letter) - d(token, centroid of far letter)
        mm = {}
        rows = []
        for cc in A:
            t = src[cc]['tok']; Xc = src[cc]['mmean'] if feat == 'mfcc_euclid' else layer_vecs(src[cc], L_PRE)
            dist = (lambda a, b: np.linalg.norm(a - b, axis=-1)) if feat == 'mfcc_euclid' else (lambda a, b: 1 - (a * b).sum(-1) / np.linalg.norm(b, axis=-1))
            cent = {l: Xc[(t.letter == l).values].mean(0) for l in LET}
            for focal, near, far in [('q', 'x', 'k'), ('x', 'q', 'k'), ('k', 't', 'q')]:
                sel = np.where(t.letter == focal)[0]
                if len(sel) > 3000: sel = rng.choice(sel, 3000, replace=False)
                dd = dist(Xc[sel], cent[near][None]) - dist(Xc[sel], cent[far][None])
                rows += [dict(test=f'{focal}: d({focal},{near})-d({focal},{far})', y=v, word=f'{cc}:{w}', unit=f'{cc}:{u}', source=cc)
                         for v, w, u in zip(dd, t.word.values[sel], t.unit.values[sel])]
        df = pd.DataFrame(rows)
        for (scope, test), g in [(('pooled', t_), g_) for t_, g_ in df.groupby('test')] + [((sc, t_), g_) for (sc, t_), g_ in df.groupby(['source', 'test'])]:
            # aggregate to (unit, word) cell means, then random intercept for word (fixed effect of source when pooled)
            g = g.groupby(['source', 'unit', 'word'], as_index=False).y.mean(); g['y'] = g.y / g.y.std()
            wm = g.groupby('word').y.mean().values
            bsw = [rng.choice(wm, len(wm)).mean() for _ in range(1000)]
            mm[f'{scope} | {test} | word-means'] = dict(n_words=len(wm), mean_sd_units=float(wm.mean()),
                                                         ci=np.percentile(bsw, [2.5, 97.5]).tolist(), note='bootstrap over word types of word-mean contrasts')
            try:
                f = 'y ~ 1' if g.source.nunique() == 1 else 'y ~ 1 + C(source, Sum)'
                m = smf.mixedlm(f, g, groups='word').fit(method='lbfgs')
                ci = m.conf_int().loc['Intercept'].tolist()
                mm[f'{scope} | {test}'] = dict(n_cells=len(g), n_words=int(g.word.nunique()), intercept_sd_units=float(m.params['Intercept']),
                                ci=ci, p=float(m.pvalues['Intercept']), model=f + ' + (1|word), on unit x word cell means')
            except Exception as e:
                mm[f'{scope} | {test}'] = dict(error=str(e))
        res['mixed_model'] = mm; out[feat] = res; print('word checks', feat, flush=True)
    return out

if __name__ == '__main__':
    os.chdir(workdir('full52-v2'))   # tokens_<src>.csv, feats_*_<src> and outputs live in this working folder
    main()
