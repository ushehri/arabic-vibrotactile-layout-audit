# Arabic vibrotactile layout audit

Analysis code for: *Auditing a Wearable Vibrotactile Letter Layout Against Open Speech Data: A Distance-Matrix and Exhaustive-Ranking Pipeline Applied to Arabic Pronunciation Training* (Alshehri & Newbury, submitted to MDPI *Computers*, 2026).

This repository contains **code only**. No audio, token-level data, raw datasets or derived per-token tables are included. Each dataset must be obtained from its official source under its own license.

## Pipeline overview
1. **Distance matrices**: 8 × 8 letter distance matrices from articulatory evidence (KAPD place codes and EPG), phonology (PanPhon), native acoustics (MFCC and XLS-R layer embeddings after CTC forced alignment) and learner confusions (L2-KSU). Lebanese Arabic ultrasound data are used for a separate emphatic-versus-plain test of ض.
2. **Order and pairwise tests**: order tests of the eight sleeve positions against each front-to-back code (Spearman ρ and Kendall τb, exact p over all 8! = 40,320 orderings), and Mantel and partial Mantel tests of each evidence matrix against sleeve distance (Spearman, 9,999 permutations, two-sided p primary), with bootstrap intervals.
3. **Exhaustive layout ranking**: all 6^8 = 1,679,616 assignments of the eight letters to the six channels are enumerated (1,679,610 after excluding the six single-channel assignments) and scored by the spatial correspondence score (SCS) and the confusion-weighted separation (SS). Layouts are ranked in the unrestricted set, the collision-free set (259,200) and the collision-free, place-order-preserving set (548). The paper ranks the current layout among the 548 collision-free, place-order-preserving assignments and computes the Pareto front of SCS and SS in that set.

## Which code produced the paper's results
Folder names keep the draft section numbers (5.1, 5.2, 5.4, 5.5). They correspond to manuscript Sections 4.1–4.4. Each folder has its own `README.md`.

| Manuscript result | Final code (reported numbers) | Inputs from earlier folders | Superseded, not reported |
|---|---|---|---|
| §4.1 articulatory correspondence: Table 4, Figure 1, Table 5 (EPG values), Tables S1–S4, Figure S1 | `s5.1-rev/` (`s51_rev.py`, `s51_rev_epg_table.py`, `s51_rev_fig1.py`) | `s5.1/s51_uti_emphasis.py` (ultrasound cells) | `s5.1/s51_analysis.py` and the order-test/EPG figures of `s5.1/s51_figures_tables.py` |
| §4.1 Figure 2 (ultrasound, /dˤ/ vs. /d/) | `s5.1/s51_figures_tables.py` (`s51_fig3_uti_emphasis.png`) | `s5.1/s51_uti_emphasis.py` | – |
| §4.2 native acoustics: Table 6, Tables S5–S7, robustness and alignment-audit statements | `s5.2-rev/` | `full-5.2-v2/` matrices and bootstraps, built on `full-5.2/` tokens and features | Mantel values in `full-5.2/` and in `full-5.2-v2/results_v2.json` (ض at 11.0 cm, one-sided) |
| §4.2 Figure 3, Table S8 (word-type resampling) | `full-5.2-v2/` (`analysis_v2.py`, `report_v2.py`) | `full-5.2/` | – |
| §4.3 learner confusions: Table 7, Figures 4 and 5, Tables S9–S11, Figure S4 | `s5.4-rev/` (`s54_rev.py`, `s54_rev_sens.py`) | `s5.4/s54_parse.py`, first part of `s5.4/s54_analysis.py` | Mantel tests and figures of `s5.4/` |
| §4.4 layout ranking: Table 8, Figure 6, Tables S12–S15 | `s5.5-rev/` (`targets.py`, `s55_rev.py`, `s55_rev_percentiles.py`, `s55_rev_fig.py`) | `s5.4/` (`tokens.csv`, `s54_pairs.csv`), `full-5.2-v2/`, `s5.1-rev/` (EPG table), `s5.4-rev/` (cross-check) | all of `s5.5/` (fixed-slot ranking) |
| Table S16 (ض coding sensitivity across analyses) | `s5.1-rev/`, `s5.2-rev/`, `s5.4-rev/`, `s5.5-rev/` | as above | – |
| Tables S17–S20 and `results/matrices/` | `matrices/export_matrices.py` | final outputs of `s5.1-rev/`, `s5.4-rev/`, `s5.5-rev/targets.py`, `full-5.2-v2/` | – |
| §3.4 tie handling (rounding to 10⁻⁹ before ranking) | `tie-audit/tie_audit.py` (check only) | – | – |
| none | – | – | `exploratory/` (pilot and early L2-KSU counts) |

`results/matrices/` holds the complete 8 × 8 matrices used as ranking targets and in the tests (CSV; see `results/matrices/README.md`). Acoustic matrices computed from L2-KSU audio are not included.

## Data and output locations
All data, intermediate and output paths come from one root, the environment variable `LMP_DATA`. It defaults to `data/` inside the repository, i.e. `./data` when you run from the repository root. Scripts resolve this root and their imports between `code/` folders relative to their own file (see `code/lmp_paths.py`), so they can be run from any working directory, e.g. `LMP_DATA=/path/to/data python code/s5.5-rev/s55_rev.py`.

Expected layout under `$LMP_DATA`:

| Path | Content |
|---|---|
| `l2ksu/extracted/L2-KSU-Dataset/` | L2-KSU corpus (`native_speakers/`, `non-native_speakers/`, as distributed) |
| `kapd/P_sample/` | KAPD EPG files of experiment P; `kapd/kapd_epg_summary.csv` (EPG summary, used only by the superseded `s5.1` scripts) |
| `s51data/stefanocoretta-mv_uti-7ce67cc/` | Coretta and Sakr research compendium (ultrasound) |
| `pilot/data/arabic-speech-corpus/` | Arabic Speech Corpus (Halabi), `wav/` and `textgrid/` |
| `extra/iqra/`, `extra/everyayah/`, `extra/cv/` | Iqra_train (dev parquet), EveryAyah (train parquet shards), Common Voice Arabic 26.0 (`ar_Male/`, `ar_Female/`) |
| `full52/`, `full52-v2/`, `s54/`, `s55/` | working folders written by `full-5.2`, `full-5.2-v2`, `s5.4` and `s5.5` |
| `results/<folder>/`, `figures/`, `tables/` | tables/JSON, figures and the `s5.1` Excel sheet written by the analysis scripts |

`data/` is ignored by git. Many outputs are derived from L2-KSU (CC BY-NC-ND 4.0) and must not be committed or redistributed.

## Running
Run the folders in dependency order: `full-5.2` → `full-5.2-v2` (`code/full-5.2-v2/chain.sh` runs its stages in sequence) → `s5.1` (`s51_uti_emphasis.py`) → `s5.1-rev` → `s5.2-rev` → `s5.4` (`s54_parse.py`, `s54_analysis.py`) → `s5.4-rev` → `s5.5-rev` → `matrices`. Each folder `README.md` gives the order within the folder. The acoustic stages need PyTorch and Transformers.

## Obtaining the datasets
- **KAPD** (KACST Arabic Phonetics Database; labels and EPG): https://sourceforge.net/projects/kapd/.
- **Coretta and Sakr ultrasound compendium** (Lebanese Arabic, Sakr data): https://doi.org/10.5281/zenodo.15335918.
- **Arabic Speech Corpus** (Halabi; reference speaker): https://huggingface.co/datasets/halabi2016/arabic_speech_corpus.
- **Iqra_train** (Iqra'Eval shared task; dev split used): https://huggingface.co/datasets/IqraEval/Iqra_train.
- **EveryAyah** recitations: https://huggingface.co/datasets/tarteel-ai/everyayah.
- **Common Voice Arabic 26.0**: Mozilla Common Voice project (https://commonvoice.mozilla.org/), CC0.
- **L2-KSU** (native and learner speech): Linguistic Data Consortium LDC2024S11 (https://catalog.ldc.upenn.edu/LDC2024S11) and https://huggingface.co/datasets/IWAN/L2-KSU-Dataset; licensed CC BY-NC-ND 4.0. No L2-KSU-derived token-level data is redistributed here.

## Software
Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, PanPhon 0.22.2, matplotlib 3.11.2. Acoustic environment: NumPy 2.5.4, librosa 1.0.0, PyTorch 2.14.1 (CPU), Transformers 5.19.0, statsmodels 0.15.0. Models: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (alignment), `facebook/wav2vec2-xls-r-300m` (features). See `requirements.txt`. Figures with Arabic labels use Noto Sans Arabic if it is installed (set `LMP_ARABIC_FONT` to a font file to choose one).

## Seeds
0: acoustic token sampling, bootstraps and permutations. 54: learner analyses. 55: layout ranking and its learner bootstrap. Others appear in checks: 1 for the alignment spot-check sample (`handcheck.py`), 52 for the alignment audit (`s52_rev_audit.py`).

## License
Code: MIT (see `LICENSE`). Datasets remain under their own licenses.
