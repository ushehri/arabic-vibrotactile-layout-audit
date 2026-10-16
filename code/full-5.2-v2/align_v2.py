"""Stage 1: CTC letter forced alignment for extra sources (same model and exact CTC Viterbi as
full52/align_l2ksu.py). Canonical sequence = letters of the transcript (diacritics/Quranic marks removed,
word boundaries '|'). Per letter we store: segment [first emitting frame, onset of next emitted symbol),
mean CTC posterior, the greedy (unconstrained) top non-blank symbol over the segment frames (dialect check),
and a consonantal-ي flag from the diacritics. Usage: python align_v2.py [source ...]"""
import csv, sys, time, re, unicodedata
import numpy as np, pandas as pd, torch, librosa
sys.path.insert(0, 'data/full52')
from align_l2ksu import MODEL, HOP
import inspect, align_l2ksu
# same algorithm as full52 ctc_viterbi; only fix: cast backpointer to int (numpy-2 int8 overflow for >127 states)
exec(inspect.getsource(align_l2ksu.ctc_viterbi).replace('s -= bp[t, s]', 's -= int(bp[t, s])'))
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
torch.set_num_threads(int(__import__("os").environ.get("NTH", 8)))
HARAKAT = set('\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652')
DROP = set(chr(c) for c in range(0x0653, 0x0670)) | set(chr(c) for c in range(0x06D6, 0x06EE)) | {'\u0670', 'ـ'}
MAP = {'ٱ': 'ا', 'ی': 'ي', 'ک': 'ك'}

def parse(text, vocab):
    """-> list of words; word = list of (letter, own_marks, prev_marks)."""
    words = []
    for w in re.split(r'\s+', unicodedata.normalize('NFC', text)):
        out = []
        for ch in w:
            ch = MAP.get(ch, ch)
            if ch in HARAKAT:
                if out: out[-1][1].add(ch)
            elif ch in DROP: continue
            elif ch in vocab and ch not in '|-' and len(ch) == 1 and '\u0621' <= ch <= '\u064a':
                out.append([ch, set(), None])
        for i, x in enumerate(out):
            x[2] = out[i - 1][1] if i else None
        if out: words.append(out)
    return words

def ya_class(word, i):
    own, prev = word[i][1], word[i][2]
    diac = any(x[1] for x in word)
    if diac:
        if own & set('\u064e\u064f\u0650\u0651\u064b\u064c\u064d'): return 'cons'
        if '\u0652' in own and not (prev and '\u0650' in prev): return 'cons'
        if i == 0: return 'cons'
        return 'vowel'
    return 'cons' if i == 0 else 'ambiguous'

def main():
    srcs = sys.argv[1:] or ['iqra', 'everyayah', 'cv']
    proc = Wav2Vec2Processor.from_pretrained(MODEL); model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    vocab = proc.tokenizer.get_vocab(); inv = {v: k for k, v in vocab.items()}
    man = pd.read_csv('manifest.csv', keep_default_na=False); man = man[man.source.isin(srcs)]
    f = open(f"alignments_{'_'.join(srcs)}.csv", 'w', newline=''); w = csv.writer(f)
    w.writerow(['source', 'utt', 'pos', 'letter', 'word', 'word_pos', 'ya', 'start', 'end', 'post', 'greedy', 'utt_logp_per_frame'])
    t0 = time.time()
    for n, r in enumerate(man.itertuples()):
        words = parse(r.text, vocab)
        if not words: continue
        toks, owner, info = [], [], []
        for wi, wd in enumerate(words):
            if wi: toks.append(vocab['|']); owner.append(-1)
            wstr = ''.join(x[0] for x in wd)
            for li, x in enumerate(wd):
                owner.append(len(info)); toks.append(vocab[x[0]])
                info.append((x[0], wstr, li, ya_class(wd, li) if x[0] == 'ي' else ''))
        y, _ = librosa.load(r.wav, sr=16000)
        with torch.inference_mode():
            logp = torch.log_softmax(model(proc(y, sampling_rate=16000, return_tensors='pt').input_values).logits[0], -1).numpy()
        if logp.shape[0] < 2 * len(toks) + 1: continue
        path, score = ctc_viterbi(logp, toks, blank=vocab['<pad>'])
        first, frames = {}, {}
        for k, t in path: first.setdefault(k, t); frames.setdefault(k, []).append(t)
        st = sorted(first.items()); nxt = {k: (st[j + 1][1] if j + 1 < len(st) else logp.shape[0]) for j, (k, _) in enumerate(st)}
        for k, o in enumerate(owner):
            if o < 0: continue
            lab, wd, li, ya = info[o]
            if lab not in 'ترضيكقخغ': continue
            a, b = first[k], nxt[k]
            seg = logp[a:b].copy(); seg[:, [vocab['<pad>'], vocab['|']]] = -np.inf
            greedy = inv[int(np.bincount(seg.argmax(1)).argmax())]
            post = float(np.mean([np.exp(logp[t, toks[k]]) for t in frames[k]]))
            w.writerow([r.source, r.utt, o, lab, wd, li, ya, round(a * HOP, 3), round(b * HOP, 3), round(post, 4), greedy, round(score / logp.shape[0], 4)])
        if n % 100 == 0: f.flush(); print(f'{n+1}/{len(man)} {time.time()-t0:.0f}s', flush=True)
    f.close(); print('DONE', flush=True)

if __name__ == '__main__': main()
