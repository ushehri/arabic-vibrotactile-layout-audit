"""Stage 2: token table + per-token features for L2-KSU natives (primary) and Halabi (reference).

Features A: 13 MFCC + Δ + ΔΔ (librosa; 16 kHz; 25 ms Hann window = 400 samples, 10 ms hop = 160; 40 mel bands
default; Δ/ΔΔ width 9 over the whole utterance), then per-speaker CMVN (mean/std over all frames of all of that
speaker's analysed utterances). Token = frames whose centre lies inside the padded segment.
Features B: facebook/wav2vec2-xls-r-300m, utterance-level forward pass, all 25 hidden states
(CNN-projection output + 24 transformer layers), mean over the 20 ms frames whose centre lies in the padded segment.
Output: tokens.csv, feats_mfcc.pkl (list of (n_frames x 39) arrays), feats_ssl.npy (n_tok x 25 x 1024, float16).
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, workdir
import csv, glob, os, sys, pickle, time, collections
import numpy as np, librosa, torch
import textgrid

PAD, MIN_DUR, SR, HOP_S = 0.010, 0.030, 16000, 0.010
MAX_SEG = 0.250   # L2-KSU CTC segments: cap length from onset (word-final letters otherwise absorb pauses)
TARGETS = {'ت': 't', 'ر': 'r', 'ض': 'dd', 'ي': 'j', 'ك': 'k', 'ق': 'q', 'خ': 'x', 'غ': 'gh'}
J_WORDS = {'إياك', 'يبتغون', 'غير', 'عليهم'}
HALABI = {'t': 't', 'r': 'r', 'D': 'dd', 'y': 'j', 'k': 'k', 'q': 'q', 'x': 'x', 'g': 'gh'}
HALABI_DIR = data('pilot/data/arabic-speech-corpus')
SSL_MODEL = 'facebook/wav2vec2-xls-r-300m'
torch.set_num_threads(8)


def l2ksu_tokens(min_post=0.5):
    rows, excl = [], []
    for r in csv.DictReader(open('alignments_l2ksu.csv')):
        key = TARGETS.get(r['letter'])
        if not key:
            continue
        parts = r['tg'].split('/')
        spk = f"{'F' if parts[-5] == 'females' else 'M'}{int(parts[-4][7:]):02d}"
        st, en = float(r['start']), float(r['end'])
        capped = en - st > MAX_SEG
        en = min(en, st + MAX_SEG)
        tok = dict(corpus='l2ksu', speaker=spk, sentence=parts[-3], utt=r['utt'], wav=r['wav'], word=r['word'],
                   letter=key, seg_start=st, seg_end=en, start=max(0, st - PAD), end=en + PAD,
                   post=float(r['post']), capped=int(capped))
        if key == 'j' and r['word'] not in J_WORDS:
            continue                                   # long vowel /iː/: not a candidate (counted separately)
        if tok['post'] < min_post:
            excl.append({**tok, 'reason': f'CTC letter posterior < {min_post}'}); continue
        if en - st < MIN_DUR:
            excl.append({**tok, 'reason': 'segment < 30 ms'}); continue
        rows.append(tok)
    return rows, excl


def halabi_tokens():
    rows, excl = [], []
    for tg in sorted(glob.glob(f'{HALABI_DIR}/textgrid/*.TextGrid')):
        wav = f"{HALABI_DIR}/wav/{os.path.basename(tg)[:-9]}.wav"
        if not os.path.exists(wav):
            continue
        for iv in textgrid.TextGrid.fromFile(tg).getFirst('phones'):
            key = HALABI.get(iv.mark.strip())
            if not key:
                continue
            tok = dict(corpus='halabi', speaker='HALABI', sentence='', utt=os.path.basename(wav)[:-4], wav=wav,
                       word='', letter=key, seg_start=iv.minTime, seg_end=iv.maxTime,
                       start=max(0, iv.minTime - PAD), end=iv.maxTime + PAD, post='', capped=0)
            (rows if iv.maxTime - iv.minTime >= MIN_DUR else excl).append(
                tok if iv.maxTime - iv.minTime >= MIN_DUR else {**tok, 'reason': 'segment < 30 ms'})
    return rows, excl


def mfcc_utt(y):
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=400, win_length=400, hop_length=160, center=True)
    return np.vstack([m, librosa.feature.delta(m, order=1), librosa.feature.delta(m, order=2)]).T  # (T,39)


def frame_slice(start, end, hop, n):
    """Indices of frames whose centre (frame k centre = k*hop for center=True) lies in [start, end]."""
    a = int(np.ceil(start / hop)); b = int(np.floor(end / hop))
    return slice(max(0, a), min(n, b + 1))


def main():
    t1, e1 = l2ksu_tokens(); t2, e2 = halabi_tokens()
    toks = t1 + t2
    with open('excluded_tokens.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list((e1 + e2)[0])); w.writeheader(); w.writerows(e1 + e2)
    by_utt = collections.defaultdict(list)
    for i, t in enumerate(toks):
        by_utt[t['wav']].append(i)
    # speaker CMVN statistics need all frames of the speaker's utterances (analysed sentences)
    spk_utts = collections.defaultdict(set)
    for t in toks:
        spk_utts[t['speaker']].add(t['wav'])
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(SSL_MODEL)
    ssl = Wav2Vec2Model.from_pretrained(SSL_MODEL).eval()
    raw_mfcc = [None] * len(toks); ssl_feat = np.zeros((len(toks), 25, 1024), dtype=np.float16)
    spk_frames = collections.defaultdict(list)
    t0 = time.time()
    for n, (wav, idx) in enumerate(by_utt.items()):
        y, _ = librosa.load(wav, sr=SR)
        M = mfcc_utt(y); spk_frames[toks[idx[0]]['speaker']].append(M)
        with torch.inference_mode():
            iv = fe(y, sampling_rate=SR, return_tensors='pt').input_values
            hs = torch.stack(ssl(iv, output_hidden_states=True).hidden_states)[:, 0].numpy()  # (25, T, 1024)
        for i in idx:
            t = toks[i]
            raw_mfcc[i] = M[frame_slice(t['start'], t['end'], HOP_S, len(M))]
            # wav2vec2 frame k covers ~[k*20ms, k*20ms+25ms]; centre ≈ k*0.02+0.0125
            sl = frame_slice(t['start'] - 0.0125, t['end'] - 0.0125, 0.02, hs.shape[1])
            if sl.stop <= sl.start:  # guarantee ≥1 frame
                k = min(hs.shape[1] - 1, int(round((t['start'] + t['end']) / 2 / 0.02)))
                sl = slice(k, k + 1)
            ssl_feat[i] = hs[:, sl].mean(1)
        if n % 50 == 0:
            print(f'{n+1}/{len(by_utt)} utts {time.time()-t0:.0f}s', flush=True)
    stats = {s: (np.vstack(F).mean(0), np.vstack(F).std(0)) for s, F in spk_frames.items()}
    mfcc = [(raw_mfcc[i] - stats[t['speaker']][0]) / stats[t['speaker']][1] for i, t in enumerate(toks)]
    for i, t in enumerate(toks):
        t['n_mfcc_frames'] = len(mfcc[i])
    with open('tokens.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(toks[0])); w.writeheader(); w.writerows(toks)
    pickle.dump(mfcc, open('feats_mfcc.pkl', 'wb'))
    np.save('feats_ssl.npy', ssl_feat)
    print('DONE', len(toks), 'tokens', flush=True)


if __name__ == '__main__':
    os.chdir(workdir('full52'))   # stage files (bare file names) live in this working folder
    main()
