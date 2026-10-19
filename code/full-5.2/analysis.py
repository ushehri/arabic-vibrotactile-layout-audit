"""Stage 3: distances, bootstrap CIs, Mantel tests, figures for Section 5.2.

Distance of letters (i, j) for one speaker = mean distance over all token pairs (a in i, b in j);
within-letter (i, i) = mean over distinct token pairs. Speaker matrices are averaged across speakers.
95% CIs: percentile bootstrap over speakers (1,000 resamples). Mantel: Spearman on the 28 upper-triangle
cells, 9,999 joint row/column permutations, one-sided (H1: r > 0).
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import workdir
import csv, pickle, json, collections, sys, os
import numpy as np, librosa
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

LET = ['t', 'r', 'dd', 'j', 'k', 'q', 'x', 'gh']
AR = {'t': 'ت', 'r': 'ر', 'dd': 'ض', 'j': 'ي', 'k': 'ك', 'q': 'ق', 'x': 'خ', 'gh': 'غ'}
IPA = {'t': 't', 'r': 'r', 'dd': 'dˤ', 'j': 'j', 'k': 'k', 'q': 'q', 'x': 'x', 'gh': 'ɣ'}
POS = {'t': 1.5, 'r': 1.5, 'dd': 11.0, 'j': 11.0, 'k': 15.0, 'q': 18.0, 'x': 20.5, 'gh': 23.0}
NB, NPERM, SEED = 1000, 9999, 0
DTW_MAX = 10          # max tokens per letter per speaker used for DTW (random, seed 0)
IU = np.triu_indices(8, 1)
rng0 = np.random.default_rng(SEED)


def sleeve(pos):
    p = np.array([pos[l] for l in LET]); return np.abs(p[:, None] - p[None, :])


def pair_mean(D, same):
    if same:
        n = D.shape[0]
        return np.nan if n < 2 else D[np.triu_indices(n, 1)].mean()
    return D.mean()


def letter_matrix(groups, dist):
    M = np.full((8, 8), np.nan)
    for i in range(8):
        for j in range(i, 8):
            if len(groups[i]) and len(groups[j]):
                M[i, j] = M[j, i] = pair_mean(dist(groups[i], groups[j]), i == j)
    return M


def cos_d(A, B): return cdist(A, B, 'cosine')
def euc_d(A, B): return cdist(A, B, 'euclidean')


def dtw_d(A, B):
    out = np.zeros((len(A), len(B)))
    for a, x in enumerate(A):
        for b, y in enumerate(B):
            D, wp = librosa.sequence.dtw(X=x.T, Y=y.T, metric='euclidean')   # steps (1,1),(1,0),(0,1), unit weights
            out[a, b] = D[-1, -1] / len(wp)                                  # normalised by warping-path length
    return out


def mantel(A, B, nperm=NPERM, seed=SEED):
    r0 = spearmanr(A[IU], B[IU])[0]
    rng = np.random.default_rng(seed); hits = 0
    for _ in range(nperm):
        p = rng.permutation(8)
        hits += spearmanr(A[p][:, p][IU], B[IU])[0] >= r0 - 1e-12
    return float(r0), (hits + 1) / (nperm + 1)


def boot(stack, nb=NB, seed=SEED):
    """stack: (n_spk, 8, 8). Return mean, lo, hi matrices and the bootstrap means (nb, 8, 8)."""
    rng = np.random.default_rng(seed); n = len(stack)
    bs = np.stack([np.nanmean(stack[rng.integers(0, n, n)], 0) for _ in range(nb)])
    return np.nanmean(stack, 0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), bs


def summarise(stack, S, name):
    mean, lo, hi, bs = boot(stack)
    r, p = mantel(mean, S)
    rb = np.array([spearmanr(b[IU], S[IU])[0] for b in bs])
    per = np.array([spearmanr(m[IU], S[IU])[0] if not np.isnan(m[IU]).any() else np.nan for m in stack])
    return dict(name=name, mean=mean, lo=lo, hi=hi, boot=bs, r=r, p=p, r_ci=np.percentile(rb, [2.5, 97.5]),
                per_spk=per)


def main():
    toks = list(csv.DictReader(open('tokens.csv')))
    mf = pickle.load(open('feats_mfcc.pkl', 'rb'))
    ssl = np.load('feats_ssl.npy', mmap_mode='r')
    S, S15 = sleeve(POS), sleeve({**POS, 'dd': 1.5})
    res = {}
    for corpus in ['l2ksu', 'halabi']:
        idx = collections.defaultdict(lambda: collections.defaultdict(list))
        for i, t in enumerate(toks):
            if t['corpus'] == corpus and int(t['n_mfcc_frames']) >= 3:   # 12 Halabi tokens lie past the end of truncated wavs
                idx[t['speaker']][t['letter']].append(i)
        spks = sorted(s for s in idx if all(len(idx[s][l]) >= 1 for l in LET))
        print(corpus, 'speakers with all 8 letters:', len(spks), flush=True)
        st = collections.defaultdict(list)
        for s in spks:
            G = [idx[s][l] for l in LET]
            pooled = [np.array([mf[i].mean(0) for i in g]) for g in G]
            st['mfcc_euclid'].append(letter_matrix(pooled, euc_d))
            st['mfcc_cosine'].append(letter_matrix(pooled, cos_d))
            sub = [list(np.random.default_rng(SEED).permutation(g)[:DTW_MAX]) if corpus == 'l2ksu' else g[:30]
                   for g in G]
            st['mfcc_dtw'].append(letter_matrix([[mf[i] for i in g] for g in sub], dtw_d))
            E = np.asarray(ssl[sorted(set(sum(G, [])))], dtype=np.float32)
            pos = {i: k for k, i in enumerate(sorted(set(sum(G, []))))}
            for L in range(25):
                st[f'ssl_L{L:02d}'].append(letter_matrix([E[[pos[i] for i in g], L] for g in G], cos_d))
            print(' ', s, flush=True)
        res[corpus] = dict(speakers=spks, stacks={k: np.stack(v) for k, v in st.items()})
    pickle.dump(res, open('stacks.pkl', 'wb'))
    print('DONE stacks', flush=True)


if __name__ == '__main__':
    os.chdir(workdir('full52'))   # stage files (bare file names) live in this working folder
    main()
