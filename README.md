# Arabic vibrotactile layout audit

Analysis code for: *A computational audit of an 8-letter Arabic tongue-to-forearm vibrotactile layout* (Alshehri & Newbury, submitted to MDPI *Computers*, 2026).

This repository contains **code only**. No audio, token-level data, raw datasets or derived per-token tables are included. Each dataset must be obtained from its official source under its own licence.

## Pipeline overview
1. **Distance matrices** – articulatory (EPG/KAPD, ultrasound), phonological (PanPhon) and acoustic (MFCC, XLS-R layer embeddings after CTC forced alignment) 8×8 letter distance matrices.
2. **Mantel / rank tests** – agreement between each matrix and the forearm sleeve layout geometry (Mantel permutation tests, rank correlations, bootstrap CIs, learner confusion comparisons).
3. **Exhaustive layout ranking** – all 8! = 40,320 letter-to-actuator assignments scored against the distance matrices; percentile of the published layout and Pareto analysis.

## Script-to-result map
| Folder | Manuscript section / result |
|---|---|
| `code/s5.1`, `code/s5.1-rev` | §5.1 articulatory distances (KAPD EPG, Coretta & Sakr ultrasound), Table/Figure 1 |
| `code/full-5.2`, `code/full-5.2-v2` | §5.2 acoustic pipeline: alignment, MFCC/XLS-R features, letter distance matrices (`chain.sh` runs v2 end to end) |
| `code/s5.2-rev` | §5.2 Mantel tests, de-duplication, window sensitivity, main table |
| `code/s5.4`, `code/s5.4-rev` | §5.4 learner (L2-KSU) confusion vs sleeve distance, leave-one-out sensitivity |
| `code/s5.5`, `code/s5.5-rev` | §5.5 exhaustive layout ranking, percentiles, Pareto figure |
| `code/matrices` | export of the complete 8×8 matrices (`results/matrices/`, Tables S17–S20) |
| `code/tie-audit` | tie audit of rank statistics |
| `code/exploratory` | early pilot / exploratory L2-KSU scripts (not used for final results) |

`results/matrices/` holds the complete 8 × 8 matrices used as ranking targets and in the tests (CSV; see `results/matrices/README.md`); acoustic matrices computed from L2-KSU audio are not included.

Scripts expect relative paths: `data/` (datasets you download), `results/` and `figures/` (outputs). Some paths may need adjusting to your local layout.

## Obtaining the datasets
- **KAPD** (King Abdulaziz Phonetics Database, EPG) – from its official distributor/authors.
- **Coretta & Sakr Arabic ultrasound data** – from the authors' public OSF/repository release accompanying their paper.
- **Halabi** Arabic phonetic corpus/lexicon – from Nawar Halabi's official release.
- **Iqra** Qur'anic recitation dataset – from its official release.
- **Common Voice Arabic** – https://commonvoice.mozilla.org/ (Mozilla Data Collective), CC0.
- **EveryAyah** recitations – https://everyayah.com/.
- **L2-KSU** learner corpus – on request from its authors (King Saud University); licensed CC BY-NC-ND. No L2-KSU-derived data is redistributed here.

## Software
Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, PanPhon 0.22.2, matplotlib 3.11.2. Acoustic environment: NumPy 2.5.4, librosa 1.0.0, PyTorch 2.14.1 (CPU), Transformers 5.19.0, statsmodels 0.15.0. Models: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (alignment), `facebook/wav2vec2-xls-r-300m` (features).

## Seeds
0 – acoustic token sampling, bootstraps and permutations; 54 – learner analyses; 55 – layout-ranking bootstrap (others, e.g. 1 and 52, appear in sensitivity scripts).

## Licence
Code: MIT (see `LICENSE`). Datasets remain under their own licences.
