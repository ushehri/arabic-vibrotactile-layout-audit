# full-5.2-v2

Extended native-acoustic pipeline for Section 4.2: it adds Iqra_train (dev), EveryAyah and Common Voice Arabic 26.0 to the `full-5.2` tokens (`prep.py`, `align_v2.py`, `features_v2.py`), computes pooled and within-speaker MFCC and XLS-R matrices, cluster bootstraps, the ALL pool and word-type resampling (`analysis_v2.py`), and writes tables and figures (`report_v2.py`, `handcheck.py`); `chain.sh` runs these stages in order.
Its matrices and bootstrap replicates (`matrices_v2.csv`, `pooled_boots.pkl`) are final and feed Table 6 (via `s5.2-rev`), Figure 3 (`fig1_v2_heatmaps.png`), Tables S5, S8 and S19; the Mantel values in `results_v2.json` (ض at 11.0 cm, one-sided) were not reported and were replaced by `s5.2-rev`.
It imports code from `../full-5.2` (resolved relative to the script file), needs the `full-5.2` outputs in `$LMP_DATA/full52`, the extra corpora in `$LMP_DATA/extra`, PyTorch, Transformers and pyarrow, and writes to `$LMP_DATA/full52-v2`.
