"""Pilot: native acoustic distance between 8 Arabic letters vs. their sleeve positions.

Data: Halabi Arabic Speech Corpus (1 male speaker), phone-level TextGrids (tier "phones").
Scales to the full run by changing CONFIG (max_tokens, utterance list) only.

Usage: python pilot.py [--max-tokens 30] [--perms 9999]
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data
import argparse, glob, json, os, random
from dataclasses import dataclass

import numpy as np
import librosa, soundfile as sf
import textgrid
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = data("pilot", "data", "arabic-speech-corpus")   # Halabi Arabic Speech Corpus
SR, N_MFCC, N_FFT, HOP = 16000, 13, 400, 160   # 25 ms window, 10 ms hop
PAD, MIN_DUR = 0.010, 0.030                       # s
SEED = 0


@dataclass(frozen=True)
class Letter:
    key: str      # file-safe id
    ar: str       # Arabic letter
    ipa: str
    halabi: str   # phone label in the Halabi TextGrids (singleton, non-geminate)
    pos: float    # sleeve position, cm from wrist
    motor: str


LETTERS = [
    Letter("t",  "ت", "t",  "t", 1.5,  "M1"),
    Letter("r",  "ر", "r",  "r", 1.5,  "M1"),
    Letter("dd", "ض", "dˤ", "D", 11.0, "M2(+M1)"),  # two-motor letter: distance uses M2 = 11.0
    Letter("j",  "ي", "j",  "y", 11.0, "M2"),
    Letter("k",  "ك", "k",  "k", 15.0, "M3"),
    Letter("q",  "ق", "q",  "q", 18.0, "M4"),
    Letter("x",  "خ", "x",  "x", 20.5, "M5"),
    Letter("gh", "غ", "ɣ",  "g", 23.0, "M6"),
]


# ---------------------------------------------------------------- tokens
def collect_tokens(max_tokens, corpus=CORPUS):
    """Return {letter.key: [(wav, start, end)]} using TextGrid boundaries, padded by PAD."""
    found = {L.key: [] for L in LETTERS}
    by_label = {L.halabi: L.key for L in LETTERS}
    for tg_path in sorted(glob.glob(os.path.join(corpus, "textgrid", "*.TextGrid"))):
        wav = os.path.join(corpus, "wav", os.path.basename(tg_path)[:-9] + ".wav")
        if not os.path.exists(wav):
            continue
        tg = textgrid.TextGrid.fromFile(tg_path)
        tier = tg.getFirst("phones")
        for iv in tier:
            key = by_label.get(iv.mark.strip())
            if key and (iv.maxTime - iv.minTime) >= MIN_DUR:
                found[key].append((wav, max(0.0, iv.minTime - PAD), iv.maxTime + PAD))
    rng = random.Random(SEED)
    available = {k: len(v) for k, v in found.items()}
    chosen = {k: (rng.sample(v, max_tokens) if len(v) > max_tokens else v) for k, v in found.items()}
    return chosen, available


_cache = {}
def load(wav):
    if wav not in _cache:
        _cache[wav], _ = librosa.load(wav, sr=SR)
    return _cache[wav]


def segment(tok):
    wav, s, e = tok
    y = load(wav)
    return y[int(s * SR): int(e * SR)]


def frames(y):
    """13 MFCC + 13 delta, shape (frames, 26)."""
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    d = librosa.feature.delta(m, width=3, mode="nearest")
    return np.vstack([m, d]).T


# ---------------------------------------------------------------- distances
def letter_matrix(groups, pair_dist):
    """Mean distance over all token pairs between letters i and j (within-letter: i != j tokens)."""
    n = len(groups); M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A, B = groups[i], groups[j]
            ds = [pair_dist(a, b) for ai, a in enumerate(A) for bi, b in enumerate(B)
                  if not (i == j and bi <= ai)]
            M[i, j] = M[j, i] = float(np.mean(ds))
    return M


def euclid(a, b): return float(np.linalg.norm(a - b))
def cosine(a, b): return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
def dtw(a, b):
    D, wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
    return float(D[-1, -1] / len(wp))   # path-length normalised


def mantel(A, B, perms, seed=SEED):
    """Spearman Mantel test on upper triangles; one-sided (positive), joint row/col permutation."""
    iu = np.triu_indices_from(A, 1)
    r0 = spearmanr(A[iu], B[iu])[0]
    rng = np.random.default_rng(seed); n = len(A); hits = 0
    for _ in range(perms):
        p = rng.permutation(n)
        if spearmanr(A[p][:, p][iu], B[iu])[0] >= r0:
            hits += 1
    return float(r0), (hits + 1) / (perms + 1)


# ---------------------------------------------------------------- figures
def ar_label(L):
    import arabic_reshaper
    from bidi.algorithm import get_display
    return f"{get_display(arabic_reshaper.reshape(L.ar))} /{L.ipa}/"


def setup_font():
    import matplotlib; matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    for f in glob.glob("/usr/share/fonts/**/NotoSansArabic*.ttf", recursive=True):
        fm.fontManager.addfont(f)
    matplotlib.rcParams["font.family"] = ["Noto Sans Arabic", "DejaVu Sans"]


def plot_heatmaps(acoustic, sleeve, title, path):
    import matplotlib.pyplot as plt
    labs = [ar_label(L) for L in LETTERS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, M, t, fmt in [(axes[0], acoustic, title, "{:.2f}"), (axes[1], sleeve, "Sleeve distance |pos_i − pos_j| (cm)", "{:.1f}")]:
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(8), labs, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(8), labs, fontsize=12)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < (M.min() + M.max()) / 2 else "black")
        ax.set_title(t, fontsize=12); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("PILOT — 1 speaker (Halabi corpus), ≤30 tokens/letter", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_mds(D, path, title):
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    D0 = D.copy(); np.fill_diagonal(D0, 0)
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=8).fit_transform(D0)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("plasma")
    for (x, y), L in zip(xy, LETTERS):
        ax.scatter(x, y, s=160, color=cmap(L.pos / 25))
        ax.annotate(ar_label(L), (x, y), xytext=(8, 6), textcoords="offset points", fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 25))
    fig.colorbar(sm, ax=ax, label="sleeve position (cm from wrist)")
    ax.set_title(title, fontsize=11); ax.margins(0.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- clips
def write_clips(chosen, outdir, n=2, gap=0.5):
    os.makedirs(outdir, exist_ok=True)
    combined = []
    for L in LETTERS:
        for k, tok in enumerate(chosen[L.key][:n], 1):
            sf.write(os.path.join(outdir, f"{L.key}_{k:02d}.wav"), segment(tok), SR)
        if chosen[L.key]:
            combined += [segment(chosen[L.key][0]), np.zeros(int(gap * SR), dtype=np.float32)]
    sf.write(os.path.join(outdir, "all_letters.wav"), np.concatenate(combined), SR)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--perms", type=int, default=9999)
    ap.add_argument("--out", default=data("pilot", "out"))
    ap.add_argument("--letters", default="q,k,x,gh", help="comma list of Letter.key; first = focus letter")
    ap.add_argument("--full", action="store_true", help="8x8 heatmaps, MDS, Mantel, all-letters wav")
    a = ap.parse_args()
    global LETTERS
    if not a.full:
        keys = a.letters.split(",")
        LETTERS = [next(L for L in LETTERS if L.key == k) for k in keys]
    os.makedirs(a.out, exist_ok=True)

    chosen, available = collect_tokens(a.max_tokens)
    counts = {k: len(v) for k, v in chosen.items()}
    print("available:", available, "\nused:", counts)

    F = {L.key: [frames(segment(t)) for t in chosen[L.key]] for L in LETTERS}
    # z-score every feature dimension using all frames of all tokens
    allf = np.vstack([f for L in LETTERS for f in F[L.key]])
    mu, sd = allf.mean(0), allf.std(0)
    Fz = {k: [(f - mu) / sd for f in v] for k, v in F.items()}
    pooled = {k: [f.mean(0) for f in v] for k, v in Fz.items()}
    groups_pool = [pooled[L.key] for L in LETTERS]
    groups_fr = [Fz[L.key] for L in LETTERS]

    mats = {
        "euclidean_meanpooled": letter_matrix(groups_pool, euclid),
        "cosine_meanpooled": letter_matrix(groups_pool, cosine),
        "dtw_frames": letter_matrix(groups_fr, dtw),
    }
    pos = np.array([L.pos for L in LETTERS])
    sleeve = np.abs(pos[:, None] - pos[None, :])
    results = {"available": available, "used": counts, "letters": [L.__dict__ for L in LETTERS],
               "sleeve": sleeve.tolist(), "matrices": {k: M.tolist() for k, M in mats.items()}}
    setup_font()
    if a.full:
        results["mantel"] = {}
        for name, M in mats.items():
            r, p = mantel(M, sleeve, a.perms)
            results["mantel"][name] = {"spearman_r": r, "p_one_sided": p, "perms": a.perms}
        plot_heatmaps(mats["euclidean_meanpooled"], sleeve,
                      "Acoustic: mean between-letter Euclidean\n(z-scored 13 MFCC+Δ, mean-pooled)",
                      os.path.join(a.out, "fig_a_heatmaps.png"))
        plot_mds(mats["euclidean_meanpooled"], os.path.join(a.out, "fig_b_mds.png"),
                 "2-D MDS of acoustic distances (Euclidean, mean-pooled)\ncolour = sleeve position")
        write_clips(chosen, data("pilot", "clips"))
    else:
        plot_focus_bars(mats, sleeve[0], os.path.join(a.out, f"fig_{LETTERS[0].key}_vs_neighbours.png"))
        write_focus_clips(chosen[LETTERS[0].key], LETTERS[0].key, data("pilot", "clips"), n=3)
    print(json.dumps({k: np.round(M, 3).tolist() for k, M in mats.items()}))
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), ensure_ascii=False, indent=1)


def plot_focus_bars(mats, sleeve_row, path):
    """Bars: focus letter (LETTERS[0]) vs each other letter, per metric; within-letter mean as dashed line."""
    import matplotlib.pyplot as plt
    names = [("euclidean_meanpooled", "Euclidean (mean-pooled)"), ("cosine_meanpooled", "Cosine (mean-pooled)"),
             ("dtw_frames", "DTW (frames)")]
    others = LETTERS[1:]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (k, t) in zip(axes, names):
        M = np.asarray(mats[k]); v = M[0, 1:]
        bars = ax.bar([ar_label(L) for L in others], v, color=["#4C72B0", "#DD8452", "#55A868"][:len(others)])
        ax.axhline(M[0, 0], ls="--", c="grey", label=f"within {ar_label(LETTERS[0])}: {M[0,0]:.3f}")
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(t); ax.tick_params(axis="x", labelsize=13); ax.legend(fontsize=9, loc="lower right")
    ttl = ", ".join(f"{ar_label(L)} {d:.1f} cm" for L, d in zip(others, sleeve_row[1:]))
    fig.suptitle(f"PILOT (1 speaker): acoustic distance of {ar_label(LETTERS[0])} to its sleeve neighbours\n"
                 f"sleeve distance: {ttl}", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def write_focus_clips(tokens, key, outdir, n=3):
    os.makedirs(outdir, exist_ok=True)
    for k, tok in enumerate(tokens[:n], 1):
        sf.write(os.path.join(outdir, f"{key}_{k:02d}.wav"), segment(tok), SR)


if __name__ == "__main__":
    main()
