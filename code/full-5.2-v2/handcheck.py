"""Plot spectrograms of 12 random retained tokens (seed 1; stratified over sources) with the CTC segment
marked, for visual alignment checking. Output: fig_alignment_check.png, handcheck_sample.csv."""
import glob, numpy as np, pandas as pd, librosa, librosa.display, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager as fm
for f in glob.glob('/usr/share/fonts/**/NotoSansArabic*.ttf', recursive=True): fm.fontManager.addfont(f)
matplotlib.rcParams['font.family'] = ['Noto Sans Arabic', 'DejaVu Sans']
T = pd.concat([pd.read_csv(f, keep_default_na=False, dtype={'unit': str}) for f in sorted(glob.glob('tokens_*.csv'))])
smp = pd.concat([g.sample(min(len(g), 12 // T.source.nunique() + 1), random_state=1) for _, g in T.groupby('source')]).head(12)
smp.to_csv('handcheck_sample.csv', index=False)
fig, axs = plt.subplots(4, 3, figsize=(18, 16))
for ax, r in zip(axs.flat, smp.itertuples()):
    y, sr = librosa.load(r.wav, sr=16000); a = max(0, r.seg_start - 0.3); b = min(len(y) / sr, r.seg_end + 0.3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y[int(a * sr):int(b * sr)], n_fft=512, hop_length=80)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=80, x_axis='time', y_axis='hz', ax=ax, x_coords=None)
    ax.axvline(r.seg_start - a, color='w', lw=2); ax.axvline(r.seg_end - a, color='w', lw=2, ls='--')
    ax.set_title(f"{r.source} | /{r.letter}/ | {get_display(arabic_reshaper.reshape(r.word))} | post {r.post:.2f}", fontsize=11)
    ax.set_ylim(0, 8000)
fig.tight_layout(); fig.savefig('fig_alignment_check.png', dpi=110)
