"""Stage 4: statistics tables + figures from stacks.pkl."""
import pickle, json, glob, csv, collections
import numpy as np
from scipy.stats import spearmanr
from analysis import LET, AR, IPA, POS, sleeve, summarise, mantel, IU, boot
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import arabic_reshaper
from bidi.algorithm import get_display

for f in glob.glob('/usr/share/fonts/**/NotoSansArabic*.ttf', recursive=True):
    fm.fontManager.addfont(f)
matplotlib.rcParams['font.family'] = ['Noto Sans Arabic', 'DejaVu Sans']
ARS = {l: get_display(arabic_reshaper.reshape(AR[l])) for l in LET}
LAB = [f"/{IPA[l]}/  {ARS[l]}" for l in LET]   # Latin first: labels starting with Arabic get reordered by matplotlib
S, S15 = sleeve(POS), sleeve({**POS, 'dd': 1.5})
res = pickle.load(open('stacks.pkl', 'rb'))
P = res['l2ksu']['stacks']; H = res['halabi']['stacks']
nL = 25
out = {'n_speakers': len(res['l2ksu']['speakers']), 'speakers': res['l2ksu']['speakers']}

# --- main summaries
main = {k: summarise(P[k], S, k) for k in ['mfcc_euclid', 'mfcc_cosine', 'mfcc_dtw']}
layers = [summarise(P[f'ssl_L{L:02d}'], S, f'ssl_L{L:02d}') for L in range(nL)]
best = int(np.argmax([x['r'] for x in layers]))
main['ssl_best'] = layers[best]
sens = {k: mantel(v['mean'], S15) for k, v in main.items()}
sens_ci = {k: np.percentile([spearmanr(b[IU], S15[IU])[0] for b in v['boot']], [2.5, 97.5]).tolist()
           for k, v in main.items()}
for k, v in main.items():
    out[k] = dict(mean=v['mean'].round(4).tolist(), lo=v['lo'].round(4).tolist(), hi=v['hi'].round(4).tolist(),
                  mantel_r=v['r'], mantel_p=v['p'], mantel_r_boot_ci=v['r_ci'].tolist(),
                  per_speaker_r=np.round(v['per_spk'], 4).tolist(),
                  per_speaker_r_summary=dict(median=float(np.nanmedian(v['per_spk'])),
                                             q1=float(np.nanpercentile(v['per_spk'], 25)),
                                             q3=float(np.nanpercentile(v['per_spk'], 75)),
                                             n_pos=int((v['per_spk'] > 0).sum()), n=int(np.isfinite(v['per_spk']).sum())),
                  sensitivity_dd_1p5=dict(r=sens[k][0], p=sens[k][1], r_boot_ci=sens_ci[k]))
out['ssl_best_layer'] = best
out['ssl_layers'] = [dict(layer=L, r=x['r'], p=x['p'], ci=x['r_ci'].tolist()) for L, x in enumerate(layers)]
# Halabi reference (1 speaker; no speaker CI)
out['halabi'] = {}
for k in ['mfcc_euclid', 'mfcc_cosine', 'mfcc_dtw', f'ssl_L{best:02d}']:
    M = H[k][0]; r, p = mantel(M, S)
    out['halabi'][k] = dict(matrix=M.round(4).tolist(), mantel_r=r, mantel_p=p)
hl = [mantel(H[f'ssl_L{L:02d}'][0], S, nperm=999)[0] for L in range(nL)]
out['halabi']['ssl_layer_r'] = hl
json.dump(out, open('results.json', 'w'), ensure_ascii=False, indent=1)


# --- figures
def heat(ax, M, title, fmt):
    im = ax.imshow(M, cmap='viridis')
    ax.set_xticks(range(8), LAB, rotation=45, ha='right', fontsize=11); ax.set_yticks(range(8), LAB, fontsize=11)
    mid = (np.nanmin(M) + np.nanmax(M)) / 2
    for i in range(8):
        for j in range(8):
            ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', fontsize=7.5,
                    color='white' if M[i, j] < mid else 'black')
    ax.set_title(title, fontsize=11); plt.colorbar(im, ax=ax, fraction=0.046)


fig, ax = plt.subplots(1, 3, figsize=(21, 6.6))
heat(ax[0], main['mfcc_euclid']['mean'], 'MFCC+Δ+ΔΔ (CMVN), mean-pooled Euclidean', '{:.2f}')
heat(ax[1], main['ssl_best']['mean'], f'XLS-R 300M layer {best}, cosine distance', '{:.3f}')
heat(ax[2], S, 'Sleeve distance |pos_i − pos_j| (cm)', '{:.1f}')
fig.suptitle(f"Native acoustic distances, L2-KSU natives (n = {out['n_speakers']} speakers; mean of speaker matrices)", fontsize=13)
fig.tight_layout(); fig.savefig('fig1_heatmaps.png', dpi=150); plt.close(fig)

# MDS with bootstrap points (Procrustes-aligned to the mean solution)
from sklearn.manifold import MDS
from scipy.spatial import procrustes
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, key, ttl in [(axs[0], 'mfcc_euclid', 'MFCC Euclidean'), (axs[1], 'ssl_best', f'XLS-R layer {best} cosine')]:
    v = main[key]
    def emb(M):
        M = M.copy(); M = (M + M.T) / 2; np.fill_diagonal(M, 0)
        return MDS(2, dissimilarity='precomputed', random_state=0, n_init=4, normalized_stress='auto').fit_transform(M)
    ref = emb(v['mean']); ref = ref - ref.mean(0); sc = np.linalg.norm(ref)
    cmap = plt.get_cmap('plasma')
    for b in v['boot'][:200]:
        X = emb(b); X = X - X.mean(0)
        U, _, Vt = np.linalg.svd(X.T @ ref); X = X @ (U @ Vt); X *= sc / np.linalg.norm(X)
        ax.scatter(X[:, 0], X[:, 1], c=[cmap(POS[l] / 25) for l in LET], s=4, alpha=0.15)
    for (x, y), l, lab in zip(ref, LET, LAB):
        ax.scatter(x, y, s=150, color=cmap(POS[l] / 25), edgecolor='k')
        ax.annotate(lab, (x, y), xytext=(7, 6), textcoords='offset points', fontsize=13)
    ax.set_title(f'2-D MDS, {ttl}\n(small dots: 200 speaker-bootstrap solutions, Procrustes-aligned)', fontsize=11)
    ax.set_aspect('equal'); ax.margins(0.15)
fig.colorbar(plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 25)), ax=axs, label='sleeve position (cm from wrist)')
fig.savefig('fig2_mds.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# forest plot
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
for ax, key, ttl in [(axs[0], 'mfcc_euclid', 'MFCC Euclidean'), (axs[1], 'ssl_best', f'XLS-R L{best} cosine')]:
    v = main[key]; pairs = list(zip(*IU))
    order = sorted(pairs, key=lambda p: v['mean'][p])
    y = np.arange(len(order))
    m = [v['mean'][p] for p in order]
    ax.errorbar(m, y, xerr=[[v['mean'][p] - v['lo'][p] for p in order], [v['hi'][p] - v['mean'][p] for p in order]],
                fmt='o', color='k', ecolor='grey', capsize=3)
    ax.set_yticks(y, [f"/{IPA[LET[i]]}/–/{IPA[LET[j]]}/  ({S[i, j]:.1f} cm)" for i, j in order], fontsize=10)
    for yy, (i, j) in zip(y, order):   # Arabic pair as its own text object (mixed-direction tick labels get reordered)
        ax.text(1.01, yy, f"{ARS[LET[j]]}  {ARS[LET[i]]}", transform=ax.get_yaxis_transform(), va='center', fontsize=12)
    for d in range(8):
        ax.axvspan(v['lo'][d, d], v['hi'][d, d], alpha=0.04, color='C0')
    ax.set_xlabel('mean between-letter distance (95% speaker-bootstrap CI)')
    ax.set_title(f'{ttl}: 28 letter pairs (sleeve distance in brackets)\nshaded = range of within-letter CIs', fontsize=11)
fig.tight_layout(); fig.savefig('fig3_forest.png', dpi=150); plt.close(fig)

# layer-wise
fig, ax = plt.subplots(figsize=(9, 5))
r = [x['r'] for x in layers]; lo = [x['r_ci'][0] for x in layers]; hi = [x['r_ci'][1] for x in layers]
ax.fill_between(range(nL), lo, hi, alpha=0.25, label='95% speaker-bootstrap CI')
ax.plot(range(nL), r, 'o-', label='L2-KSU natives (mean matrix)')
ax.plot(range(nL), hl, 's--', color='grey', label='Halabi (1 speaker, reference)')
ax.axhline(main['mfcc_euclid']['r'], color='C3', ls=':', label='MFCC Euclidean')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('XLS-R 300M layer (0 = CNN projection)'); ax.set_ylabel('Mantel Spearman r vs sleeve')
ax.set_title('Layer-wise Mantel r (acoustic cosine vs sleeve distance)'); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig('fig4_ssl_layers.png', dpi=150); plt.close(fig)

# per-speaker histogram
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-1, 1, 41)
ax.hist(main['mfcc_euclid']['per_spk'], bins, alpha=0.6, label='MFCC Euclidean')
ax.hist(main['ssl_best']['per_spk'], bins, alpha=0.6, label=f'XLS-R L{best} cosine')
ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('per-speaker Mantel Spearman r'); ax.set_ylabel('speakers')
ax.set_title(f"Per-speaker Mantel r (n = {out['n_speakers']})"); ax.legend()
fig.tight_layout(); fig.savefig('fig5_per_speaker_mantel.png', dpi=150); plt.close(fig)
print('best layer', best)
for k, v in main.items():
    print(k, 'r=%.3f p=%.4f CI=%s' % (v['r'], v['p'], np.round(v['r_ci'], 3)), 'sens', np.round(sens[k], 4))
