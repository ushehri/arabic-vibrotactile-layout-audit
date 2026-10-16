"""Letter-level forced alignment of L2-KSU native utterances.

Why: every native TextGrid of a given sentence is the SAME file (1 unique md5 per sentence across ~200
utterances) and its xmax does not match the wav durations, so the shipped native boundaries are a template,
not an alignment of each recording. We therefore re-align each utterance.

How: Arabic character CTC model (jonatasgrosman/wav2vec2-large-xlsr-53-arabic, 20 ms frames) +
exact CTC Viterbi forced alignment (numpy) to the canonical letter sequence of the sentence (taken from the
template TextGrid letter tier, word boundaries as '|').
Letter segment = [first frame emitting the letter, first frame of the next emitted symbol).
Output: alignments_l2ksu.csv (one row per letter per utterance) with mean CTC posterior of the letter frames.
"""
import glob, os, csv, sys, time
import numpy as np, torch, librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tokens_l2ksu import ROOT, read_tiers

MODEL = 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic'
SENTENCES = {f'sentence{i}' for i in range(1, 9)}   # sentences 9-10 contain no target consonant tokens
HOP = 0.02
torch.set_num_threads(8)


def ctc_viterbi(logp, tokens, blank=0):
    """Return list of (token_index, frame) for the best CTC path of `tokens` through logp (T x V)."""
    T = logp.shape[0]; S = 2 * len(tokens) + 1
    ext = [blank] + [x for t in tokens for x in (t, blank)]
    NEG = -1e30
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, ext[0]]; dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        c0 = prev
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.concatenate([[NEG, NEG], prev[:-2]])
        allow2 = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
        c2 = np.where(allow2, c2, NEG)
        stack = np.stack([c0, c1, c2]); best = stack.argmax(0)
        dp[t] = stack.max(0) + logp[t, ext]; bp[t] = best
    s = S - 1 if dp[-1, S - 1] > dp[-1, S - 2] else S - 2
    path = []
    for t in range(T - 1, -1, -1):
        path.append((t, s)); s -= bp[t, s]
    path.reverse()
    return [((s - 1) // 2, t) for t, s in path if s % 2 == 1], float(dp[-1].max())


def main():
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    vocab = proc.tokenizer.get_vocab()
    tgs = [t for t in sorted(glob.glob(f'{ROOT}/*/*/*/annotation/*.TextGrid')) if t.split('/')[-3] in SENTENCES]
    out = open('alignments_l2ksu.csv', 'w', newline=''); w = csv.writer(out)
    w.writerow(['utt', 'tg', 'wav', 'pos', 'letter', 'word', 'start', 'end', 'post', 'utt_logp_per_frame'])
    t0 = time.time()
    for n, tg in enumerate(tgs):
        wav = tg.replace('/annotation/', '/wav/')[:-9] + '.wav'
        if not os.path.exists(wav):
            continue
        phones, words = read_tiers(tg)
        # canonical letter sequence with word ids
        seq = []  # (letter_label, word, [vocab ids])
        for x0, x1, lab in phones:
            if lab in ('', 'sil', 'sp'):
                continue
            wd = next((ww for a, b, ww in words if a <= (x0 + x1) / 2 < b), '?')
            ids = [vocab[c] for c in lab if c in vocab]
            if ids:
                seq.append((lab, wd, ids))
        toks, owner = [], []
        prev_word = None
        for i, (lab, wd, ids) in enumerate(seq):
            if prev_word is not None and wd != prev_word:
                toks.append(vocab['|']); owner.append(-1)
            for c in ids:
                toks.append(c); owner.append(i)
            prev_word = wd
        y, _ = librosa.load(wav, sr=16000)
        with torch.inference_mode():
            iv = proc(y, sampling_rate=16000, return_tensors='pt').input_values
            logp = torch.log_softmax(model(iv).logits[0], -1).numpy()
        if logp.shape[0] < 2 * len(toks) + 1:
            print('too short', wav, flush=True); continue
        path, score = ctc_viterbi(logp, toks)
        first, frames = {}, {}
        for k, t in path:
            first.setdefault(k, t); frames.setdefault(k, []).append(t)
        starts = sorted(first.items())
        nxt = {k: (starts[j + 1][1] if j + 1 < len(starts) else logp.shape[0]) for j, (k, _) in enumerate(starts)}
        for i, (lab, wd, ids) in enumerate(seq):
            ks = [k for k, o in enumerate(owner) if o == i]
            st = first[ks[0]] * HOP; en = nxt[ks[-1]] * HOP
            post = float(np.mean([np.exp(logp[t, toks[k]]) for k in ks for t in frames[k]]))
            w.writerow([os.path.basename(wav)[:-4], tg, wav, i, lab, wd, round(st, 3), round(en, 3),
                        round(post, 4), round(score / logp.shape[0], 4)])
        if n % 25 == 0:
            out.flush(); print(f'{n+1}/{len(tgs)} {time.time()-t0:.0f}s', flush=True)
    out.close(); print('DONE', flush=True)


if __name__ == '__main__':
    main()
