"""Stage 0: build utterance manifest for extra sources and write 16 kHz mono wavs.
Sources: iqra (IqraEval/Iqra_train dev, all 2,588 utts; no speaker ID -> unit = utterance),
everyayah (9 selected train shards, 19 reciters; <=150 random ayat per reciter with 1.5-15 s duration, seed 0),
cv (Common Voice ar 26.0 Male+Female subsets; <=20 random clips per client_id, seed 0; 1-15 s).
Output: manifest.csv (source, unit, speaker, gender, utt, wav, text, phonemes)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on the path, for lmp_paths
from lmp_paths import data, workdir
import io, os, csv, glob, json, sys
import numpy as np, pandas as pd, soundfile as sf, librosa, pyarrow.parquet as pq
EX = data('extra'); W = f'{EX}/wav16'; SR = 16000
os.chdir(workdir('full52-v2'))   # manifest.csv is written to the v2 working folder
rng = np.random.default_rng(0)
rows = []
def save(src, utt, y):
    os.makedirs(f'{W}/{src}', exist_ok=True); p = f'{W}/{src}/{utt}.wav'
    if not os.path.exists(p): sf.write(p, y.astype(np.float32), SR)
    return p
def dec(b):
    y, sr = sf.read(io.BytesIO(b), dtype='float32', always_2d=True); y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
# Iqra
t = pq.read_table(f'{EX}/iqra/data/dev-00000-of-00001.parquet').to_pylist()
for r in t:
    y = dec(r['audio']['bytes']); u = f"iqra_{r['id']}"
    rows.append(dict(source='iqra', unit=u, speaker='', gender='', utt=u, wav=save('iqra', u, y),
                     text=r['tashkeel_sentence'], phonemes=r['phoneme_ref'], dur=len(y)/SR))
print('iqra', len(t), flush=True)
# EveryAyah
for f in sorted(glob.glob(f'{EX}/everyayah/data/*.parquet')):
    pf = pq.ParquetFile(f); meta = pf.read(columns=['reciter', 'duration', 'text']).to_pandas()
    meta['i'] = range(len(meta)); pick = []
    for rec, g in meta[(meta.duration >= 1.5) & (meta.duration <= 15)].groupby('reciter'):
        pick += list(rng.permutation(g.i.values)[:150])
    pick = set(pick); off = 0
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg).to_pylist()
        for k, r in enumerate(tb):
            if off + k in pick:
                u = f"ea_{os.path.basename(f)[6:11]}_{off+k}"; y = dec(r['audio']['bytes'])
                rows.append(dict(source='everyayah', unit=r['reciter'], speaker=r['reciter'], gender='M', utt=u,
                                 wav=save('everyayah', u, y), text=r['text'], phonemes='', dur=len(y)/SR))
        off += len(tb)
    print(f, len(pick), flush=True)
# Common Voice
for g in ['Male', 'Female']:
    d = pd.concat([pd.read_csv(f'{EX}/cv/ar_{g}/{s}.tsv', sep='\t', quoting=3) for s in ['train', 'dev', 'test']])
    d = d[d.down_votes <= d.up_votes]
    for cid, gg in d.groupby('client_id'):
        for _, r in gg.sample(frac=1, random_state=0).head(20).iterrows():
            y, _ = librosa.load(f'{EX}/cv/ar_{g}/clips/{r.path}', sr=SR)
            if not 1 <= len(y)/SR <= 15: continue
            u = 'cv_' + r.path[:-4].split('_')[-1]
            rows.append(dict(source='cv', unit=cid[:12], speaker=cid[:12], gender=g[0], utt=u, wav=save('cv', u, y),
                             text=r.sentence, phonemes='', dur=len(y)/SR))
    print('cv', g, flush=True)
pd.DataFrame(rows).to_csv('manifest.csv', index=False); print('DONE', len(rows))
