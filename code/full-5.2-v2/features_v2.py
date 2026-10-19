"""Stage 2: token selection + MFCC/XLS-R features for extra sources (identical feature code/params to
full52/features.py). Token rules (same as full52): segment = onset to next onset, capped at 250 ms;
CTC letter posterior >= 0.5; segment >= 30 ms; ي only when consonantal (diacritic rule; undiacritised
non-initial ي = ambiguous -> excluded). Iqra: per-utterance per-letter count check against phoneme_ref
(t/tt, r/rr, D/DD, y/yy, k/kk, q/qq, x/xx, g/gg; ت count may be exceeded by ة realised as t); letters failing
it are excluded for that utterance. CV: ق tokens whose greedy CTC symbol (non-blank majority over the segment) is ء/أ/إ/ج/ك/غ (possible dialectal
[ʔ]/[g]) are excluded. MFCC CMVN unit: speaker (EveryAyah reciter, CV client_id) or utterance (Iqra).
Usage: python features_v2.py <source>  -> tokens_<source>.csv, feats_mfcc_<source>.pkl, feats_ssl_<source>.npy"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import workdir
import os, sys, csv, pickle, collections, time
import numpy as np, pandas as pd, librosa, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'full-5.2'))
from features import mfcc_utt, frame_slice, PAD, MIN_DUR, SR, HOP_S, MAX_SEG, TARGETS, SSL_MODEL
torch.set_num_threads(int(__import__("os").environ.get("NTH", 8)))
PH = {'t': ('t', 'tt'), 'r': ('r', 'rr'), 'dd': ('D', 'DD'), 'j': ('y', 'yy'), 'k': ('k', 'kk'), 'q': ('q', 'qq'),
      'x': ('x', 'xx'), 'gh': ('g', 'gg')}
DIALECT_Q = set('ءأإجكغ')  # 'ا' not used: segments include the following vowel, so ا is often the greedy symbol for normal [q]

def select(src):
    man = pd.read_csv('manifest.csv', keep_default_na=False).set_index('utt')
    al = pd.read_csv(f'alignments_{src}.csv', keep_default_na=False)
    rows, excl = [], []
    for utt, g in al.groupby('utt', sort=False):
        m = man.loc[utt]
        bad = set()
        if src == 'iqra':
            ph = collections.Counter(m.phonemes.split())
            txt = collections.Counter(TARGETS[l] for l, ya in zip(g.letter, g.ya) if l != 'ي' or ya == 'cons')
            n_ta_marb = m.text.count('ة')
            for L, (a, b) in PH.items():
                n = ph[a] + ph[b]
                ok = (txt[L] <= n <= txt[L] + n_ta_marb) if L == 't' else n == txt[L]
                if not ok: bad.add(L)
        for r in g.itertuples():
            L = TARGETS[r.letter]
            if L == 'j' and r.ya != 'cons': continue
            st, en = float(r.start), float(r.end); capped = en - st > MAX_SEG; en = min(en, st + MAX_SEG)
            unit = m.unit
            tok = dict(source=src, unit=unit, speaker=m.speaker, gender=m.gender, utt=utt, wav=m.wav, word=r.word,
                       letter=L, seg_start=st, seg_end=en, start=max(0, st - PAD), end=en + PAD, post=float(r.post),
                       capped=int(capped), greedy=r.greedy)
            reason = None
            if L in bad: reason = 'Iqra phoneme count mismatch'
            elif tok['post'] < 0.5: reason = 'CTC letter posterior < 0.5'
            elif en - st < MIN_DUR: reason = 'segment < 30 ms'
            elif src == 'cv' and L == 'q' and r.greedy in DIALECT_Q: reason = 'possible dialectal q (greedy ' + r.greedy + ')'
            (excl.append({**tok, 'reason': reason}) if reason else rows.append(tok))
    return rows, excl

def main(src):
    toks, excl = select(src)
    pd.DataFrame(excl).to_csv(f'excluded_{src}.csv', index=False)
    print(src, 'tokens', len(toks), 'excluded', len(excl), flush=True)
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(SSL_MODEL); ssl = Wav2Vec2Model.from_pretrained(SSL_MODEL).eval()
    by = collections.defaultdict(list)
    for i, t in enumerate(toks): by[t['wav']].append(i)
    sslf = np.lib.format.open_memmap(f'feats_ssl_{src}.npy', 'w+', np.float16, (len(toks), 25, 1024))
    raw = [None] * len(toks); unitF = collections.defaultdict(list); t0 = time.time()
    for n, (wav, idx) in enumerate(by.items()):
        y, _ = librosa.load(wav, sr=SR); M = mfcc_utt(y); unitF[toks[idx[0]]['unit']].append(M)
        with torch.inference_mode():
            hs = torch.stack(ssl(fe(y, sampling_rate=SR, return_tensors='pt').input_values, output_hidden_states=True).hidden_states)[:, 0].numpy()
        for i in idx:
            t = toks[i]; raw[i] = M[frame_slice(t['start'], t['end'], HOP_S, len(M))]
            sl = frame_slice(t['start'] - 0.0125, t['end'] - 0.0125, 0.02, hs.shape[1])
            if sl.stop <= sl.start:
                k = min(hs.shape[1] - 1, int(round((t['start'] + t['end']) / 2 / 0.02))); sl = slice(k, k + 1)
            sslf[i] = hs[:, sl].mean(1)
        if n % 100 == 0: print(f'{n+1}/{len(by)} {time.time()-t0:.0f}s', flush=True)
    stats = {u: (np.vstack(F).mean(0), np.vstack(F).std(0) + 1e-8) for u, F in unitF.items()}
    mf = [(raw[i] - stats[t['unit']][0]) / stats[t['unit']][1] for i, t in enumerate(toks)]
    for i, t in enumerate(toks): t['n_mfcc_frames'] = len(mf[i])
    pd.DataFrame(toks).to_csv(f'tokens_{src}.csv', index=False)
    pickle.dump(mf, open(f'feats_mfcc_{src}.pkl', 'wb')); sslf.flush(); print('DONE', flush=True)

if __name__ == '__main__':
    os.chdir(workdir('full52-v2'))   # manifest.csv, alignments_<src>.csv and outputs live in this working folder
    main(sys.argv[1])
