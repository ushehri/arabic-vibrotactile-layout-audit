"""Build the native-speaker token table for the 8 target letters from L2-KSU letter-level TextGrids.

Output: tokens_l2ksu.csv (all candidate tokens, with include flag and exclusion reason).
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, workdir
import re, glob, os, csv, collections
import soundfile as sf

ROOT = data('l2ksu/extracted/L2-KSU-Dataset/native_speakers')
TARGETS = {'ت': 't', 'ر': 'r', 'ض': 'dd', 'ي': 'j', 'ك': 'k', 'ق': 'q', 'خ': 'x', 'غ': 'gh'}
J_WORDS = {'إياك', 'يبتغون', 'غير', 'عليهم'}   # consonantal /j/ (glide); all other ي = long /iː/
PAD, MIN_DUR = 0.010, 0.030


def read_tiers(path):
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'utf-16'):
        try:
            txt = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    tiers = {}
    for m in re.finditer(r'name = "([^"]*)"(.*?)(?=item \[|\Z)', txt, re.S):
        tiers[m.group(1)] = [(float(a), float(b), c.strip()) for a, b, c in
                             re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', m.group(2))]
    a, b = list(tiers.values())[:2]
    # sentence-5 files have swapped tier names: the letter tier is the one with more intervals
    return (a, b) if len(a) > len(b) else (b, a)


def main():
    rows = []
    for tg in sorted(glob.glob(f'{ROOT}/*/*/*/annotation/*.TextGrid')):
        gender, spk, sent = tg.split('/')[-5:-2]
        base = os.path.basename(tg)[:-9]
        wav = tg.replace('/annotation/', '/wav/')[:-9] + '.wav'
        speaker = f"{'F' if gender == 'females' else 'M'}{int(spk[7:]):02d}"
        if not os.path.exists(wav):
            rows.append(dict(speaker=speaker, gender=gender, sentence=sent, utt=base, wav='', word='', letter='',
                             start='', end='', dur='', include=0, reason='wav missing')); continue
        wav_dur = sf.info(wav).duration
        phones, words = read_tiers(tg)
        for x0, x1, lab in phones:
            if lab not in TARGETS:
                continue
            mid = (x0 + x1) / 2
            word = next((w for a, b, w in words if a <= mid < b), '?')
            key = TARGETS[lab]
            reason = ''
            if key == 'j' and word not in J_WORDS:
                reason = 'ي = long vowel /iː/'
            elif x1 - x0 < MIN_DUR:
                reason = f'shorter than {MIN_DUR*1000:.0f} ms'
            elif x1 > wav_dur + 1e-3:
                reason = 'interval beyond wav end'
            rows.append(dict(speaker=speaker, gender=gender, sentence=sent, utt=base, wav=wav, word=word, letter=key,
                             start=round(max(0, x0 - PAD), 4), end=round(min(wav_dur, x1 + PAD), 4),
                             dur=round(x1 - x0, 4), include=int(reason == ''), reason=reason))
    with open('tokens_l2ksu.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    inc = [r for r in rows if r['include']]
    c = collections.Counter(r['letter'] for r in inc)
    print('included per letter', dict(c))
    print('excluded', collections.Counter((r['letter'], r['reason']) for r in rows if not r['include']))
    print('words', collections.Counter((r['letter'], r['word']) for r in inc))


if __name__ == '__main__':
    os.chdir(workdir('full52'))   # stage files (bare file names) live in this working folder
    main()
