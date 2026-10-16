"""Stage 4: tables (md/csv) and figures from results_v2.json. Every number printed comes from results_v2.json."""
import json, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager as fm
import sys; sys.path.insert(0, 'data/full52')
from analysis import LET, AR, IPA
for f in glob.glob('/usr/share/fonts/**/NotoSansArabic*.ttf', recursive=True): fm.fontManager.addfont(f)
matplotlib.rcParams['font.family'] = ['Noto Sans Arabic', 'DejaVu Sans']
LAB = [f"/{IPA[l]}/ {get_display(arabic_reshaper.reshape(AR[l]))}" for l in LET]
R = json.load(open('results_v2.json')); SR = R['sources']; ORDER = [c for c in ['l2ksu', 'iqra', 'everyayah', 'cv', 'halabi'] if c in SR]
F = lambda x, n=3: 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'
md = []
# counts
rows = []
for c in ORDER:
    r = SR[c]
    for l in LET: rows.append(dict(source=c, letter=l, arabic=AR[l], tokens=r['n_tokens'][l], word_types=r['word_types'][l], units=r['n_units'], units_all8=r.get('W_n_units_all8', '')))
cnt = pd.DataFrame(rows); cnt.to_csv('token_counts_v2.csv', index=False)
md.append('## Token counts (tokens / word types)\n\n| source | units | units with all 8 | ' + ' | '.join(f'{AR[l]} /{IPA[l]}/' for l in LET) + ' |\n|' + '---|' * 11)
for c in ORDER:
    r = SR[c]; md.append(f"| {c} | {r['n_units']} | {r.get('W_n_units_all8', '–')} | " + ' | '.join(f"{r['n_tokens'][l]:,} / {r['word_types'][l]:,}" for l in LET) + ' |')
# Mantel table
mrows = []
for c in ORDER + ['ALL']:
    blocks = [('P', SR[c]['P'])] + ([('W', SR[c]['W'])] if c in SR and 'W' in SR[c] else []) if c != 'ALL' else [('P-avg', R['ALL'])]
    for est, b in blocks:
        bl = b['ssl_best_layer_posthoc']
        for k in ['mfcc_euclid', 'mfcc_dtw_30tok', 'ssl_L23', f'ssl_L{bl:02d}']:
            if k in b:
                mrows.append(dict(source=c, estimator=est, measure=k + (' (post-hoc best)' if k == f'ssl_L{bl:02d}' and bl != 23 else ''),
                                  r=b[k]['mantel_r'], p=b[k]['mantel_p'], ci_lo=(b[k].get('r_ci') or [None, None])[0], ci_hi=(b[k].get('r_ci') or [None, None])[1]))
mt = pd.DataFrame(mrows).drop_duplicates(['source', 'estimator', 'measure']); mt.to_csv('mantel_v2.csv', index=False)
md.append('\n## Mantel tests (Spearman, 28 cells, 9,999 permutations, one-sided r > 0)\n\nP = pooled tokens, CI = cluster bootstrap over units (500); W = mean of within-speaker matrices, CI = speaker bootstrap (1,000); P-avg = equal-weight mean of the P matrices of l2ksu, iqra, everyayah, cv. ssl_L23 is the layer chosen in v1 (so it is pre-specified for the new sources); the "post-hoc best" layer is picked on the same data and its p is optimistic.\n\n| source | est. | measure | r | p | 95% CI of r |\n|---|---|---|---|---|---|')
for x in mt.itertuples(): md.append(f'| {x.source} | {x.estimator} | {x.measure} | {F(x.r)} | {F(x.p, 4)} | ' + (f'[{F(x.ci_lo)}, {F(x.ci_hi)}]' if x.ci_lo is not None and not pd.isna(x.ci_lo) else '–') + ' |')
# key pairs
md.append('\n## Key pairs: distance and rank among the 28 letter pairs (1 = closest)\n\n| source | est. | measure | q–x | k–q | t–k |\n|---|---|---|---|---|---|')
for c in ORDER + ['ALL']:
    blocks = [('P', SR[c]['P'])] + ([('W', SR[c]['W'])] if 'W' in SR[c] else []) if c != 'ALL' else [('P-avg', R['ALL'])]
    for est, b in blocks:
        for k, v in b['key_pairs'].items():
            ci = lambda p: (f" [{F(b[k]['lo'][LET.index(p[0])][LET.index(p[1])], 4)}, {F(b[k]['hi'][LET.index(p[0])][LET.index(p[1])], 4)}]" if 'lo' in b[k] else '')
            md.append(f"| {c} | {est} | {k} | " + ' | '.join(f"{F(v[p]['d'], 4)}{ci(pp)} (#{v[p]['rank_of_28']})" for p, pp in [('q-x', 'qx'), ('k-q', 'kq'), ('t-k', 'tk')]) + ' |')
# word resampling
md.append('\n## Word-level resampling (bootstrap word types within letter, ≤5 tokens per drawn type, ≤400 tokens/letter/replicate; 500 replicates per source, 200 for ALL)\n\n| source | measure | Mantel r mean [95%] | q–x rank median [95%] | k–q rank | t–k rank | P(q–x < k–q) | P(t–k < k–q) |\n|---|---|---|---|---|---|---|---|')
for feat, res in R['word'].items():
    for c, v in res.items():
        if c == 'mixed_model': continue
        k = v['key']; rk = lambda p: f"{k[p]['rank_ci'][1]:.0f} [{k[p]['rank_ci'][0]:.0f}, {k[p]['rank_ci'][2]:.0f}]"
        md.append(f"| {c} | {feat} | {F(v['r_mean'])} [{F(v['r_ci'][0])}, {F(v['r_ci'][1])}] | {rk('q-x')} | {rk('k-q')} | {rk('t-k')} | {F(v['qx_lt_kq'], 2)} | {F(v['tk_lt_kq'], 2)} |")
md.append('\n## Token-level contrasts with word as random effect\n\ny = d(token, centroid of near letter) − d(token, centroid of far letter), in SD units; negative = token closer to the near letter. Fitted on unit×word cell means with a random intercept for word (pooled: plus source as sum-coded fixed effect). "word-means" rows: plain bootstrap over word types (robust fallback when the mixed model is singular → n/a CI).\n\n| feature | scope / test | estimate | 95% CI | p | n words |\n|---|---|---|---|---|---|')
for feat, res in R['word'].items():
    for k, v in res['mixed_model'].items():
        if 'error' in v: md.append(f'| {feat} | {k} | error: {v["error"][:60]} | | | |'); continue
        est = v.get('intercept_sd_units', v.get('mean_sd_units'))
        md.append(f"| {feat} | {k} | {F(est)} | [{F(v['ci'][0])}, {F(v['ci'][1])}] | {F(v.get('p'), 4) if 'p' in v else '–'} | {v['n_words']} |")
open('tables_v2.md', 'w').write('\n'.join(md) + '\n')
# long-format matrices
L = []
for c in ORDER:
    for est in ['P', 'W']:
        if est in SR[c]:
            for k, b in SR[c][est].items():
                if isinstance(b, dict) and 'mean' in b:
                    for i in range(8):
                        for j in range(8):
                            L.append(dict(source=c, estimator=est, measure=k, a=LET[i], b=LET[j], d=b['mean'][i][j],
                                          lo=b.get('lo', [[None] * 8] * 8)[i][j], hi=b.get('hi', [[None] * 8] * 8)[i][j]))
pd.DataFrame(L).to_csv('matrices_v2.csv', index=False)
# figures
def heat(ax, M, t):
    M = np.array(M, float); ax.imshow(M, cmap='viridis'); ax.set_title(t, fontsize=11)
    ax.set_xticks(range(8), LAB, rotation=60, fontsize=8); ax.set_yticks(range(8), LAB, fontsize=8)
    mid = np.nanmean(M)
    for i in range(8):
        for j in range(8): ax.text(j, i, f'{M[i, j]:.3g}', ha='center', va='center', fontsize=6, color='w' if M[i, j] < mid else 'k')
srcs = ORDER + ['ALL']
fig, axs = plt.subplots(2, len(srcs), figsize=(4.6 * len(srcs), 9.5))
for k, c in enumerate(srcs):
    b = R['ALL'] if c == 'ALL' else SR[c]['P']
    heat(axs[0, k], b['mfcc_euclid']['mean'], f'{c}: MFCC Euclidean (pooled)'); heat(axs[1, k], b['ssl_L23']['mean'], f'{c}: XLS-R L23 cosine (pooled)')
fig.tight_layout(); fig.savefig('fig1_v2_heatmaps.png', dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 7)); yy = 0; yl = []
for x in mt.itertuples():
    col = {'P': 'C0', 'W': 'C1', 'P-avg': 'C3'}[x.estimator]
    ax.plot([x.r], [yy], 'o', color=col)
    if x.ci_lo is not None and not pd.isna(x.ci_lo): ax.plot([x.ci_lo, x.ci_hi], [yy, yy], color=col)
    yl.append(f'{x.source} {x.estimator} {x.measure}'); yy += 1
ax.axvline(0, color='k', lw=.8); ax.set_yticks(range(yy), yl, fontsize=7); ax.set_xlabel('Mantel r (acoustic vs sleeve distance)')
fig.tight_layout(); fig.savefig('fig2_v2_mantel_forest.png', dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 5))
for c in srcs:
    b = R['ALL'] if c == 'ALL' else SR[c]['P']; ax.plot(range(25), [b[f'ssl_L{L:02d}']['mantel_r'] for L in range(25)], marker='.', label=c)
ax.axhline(0, color='k', lw=.8); ax.set_xlabel('XLS-R layer'); ax.set_ylabel('Mantel r (pooled)'); ax.legend(); ax.set_title('Layer curve (best layer = post-hoc choice)')
fig.tight_layout(); fig.savefig('fig3_v2_ssl_layers.png', dpi=130); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, (feat, res) in zip(axs, R['word'].items()):
    cs = [c for c in res if c != 'mixed_model']
    for j, p in enumerate(['q-x', 'k-q', 't-k']):
        ci = [res[c]['key'][p]['rank_ci'] for c in cs]
        ax.errorbar(np.arange(len(cs)) + (j - 1) * .2, [x[1] for x in ci], yerr=[[x[1] - x[0] for x in ci], [x[2] - x[1] for x in ci]], fmt='o', label=p)
    ax.set_xticks(range(len(cs)), cs); ax.set_ylabel('rank among 28 pairs (1 = closest)'); ax.set_title(f'{feat}: word-resampled ranks'); ax.legend(); ax.invert_yaxis()
fig.tight_layout(); fig.savefig('fig4_v2_word_resampled_ranks.png', dpi=130); plt.close(fig)
print('OK')
