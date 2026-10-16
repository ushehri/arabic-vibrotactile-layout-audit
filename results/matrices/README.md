# Complete 8 × 8 letter matrices

Written by `code/matrices/export_matrices.py` from stored analysis outputs (no new analysis). The same matrices are in Supplementary File S2, sheets Table S17–S20.
Rows = columns, in sleeve order: ت /t/, ر /r/, ض /dˤ/, ي /j/, ك /k/, ق /q/, خ /x/, غ /ɣ/ (UTF-8 CSV).

| File(s) | Content | Used in |
|---|---|---|
| `artic_panphon_feature_distance.csv` | PanPhon: differing features among 24 | blend ranking target; control in learner partial Mantel tests |
| `artic_kapd_place_absdiff.csv` | KAPD place code difference | blend ranking target |
| `artic_blend_target.csv` | 0.5·rn(PanPhon) + 0.5·rn(Place) (diagonal unused) | ranking target (weight 0.6) |
| `artic_order_codes.csv` | KAPD, modern-literature, classical and EPG (final extraction) codes | order tests (Table 4/S1); baselines B1–B3 |
| `artic_modern_code_absdiff.csv`, `artic_classical_code_absdiff.csv`, `artic_epg_meanpos_absdiff.csv` | pairwise abs-difference forms of these codes | completeness only (not pairwise test targets) |
| `learner_confusion_counts.csv` | L2-KSU learner confusion counts (aggregate; = Table S10) | Table 7/S9 |
| `learner_confusion_symmetrised_pct.csv` | symmetrised confusion S (%) | Mantel tests; negated = ranking target (weight 0.3) |
| `acoustic_{iqra,everyayah,cv,halabi}_P_{mfcc_euclid,ssl_L23}.csv` | pooled-token MFCC Euclidean / XLS-R layer 23 cosine | Table 6/S5 |
| `acoustic_ALL_without_L2KSU_P_*.csv` | mean of the Iqra, EveryAyah, Common Voice matrices | Table 6/S6 |
| `acoustic_all_measures_non_L2KSU_long.csv` | every stored non-L2-KSU matrix (P/W, DTW, XLS-R layers 0–24), long format | Tables S5/S6, Figure S2 |
| `sleeve_distance_current_dad_*.csv` | sleeve distance (cm), ض at 6.25 / 11.0 / 1.5 cm | all pairwise tests and ranking |


**Precision:** MFCC Euclidean and XLS-R layer 23 square matrices (`acoustic_*_P_mfcc_euclid.csv`, `acoustic_*_P_ssl_L23.csv`) are written at full float64 precision (`%.17g` from `pooled_boots.pkl`). The long-format file `acoustic_all_measures_non_L2KSU_long.csv` keeps the 5-decimal values of `matrices_v2.csv`.

**Not included (L2-KSU licence, CC BY-NC-ND 4.0):** acoustic matrices computed from L2-KSU audio — the L2-KSU native P/W matrices and the four-corpus ALL pool (which is also the MFCC ranking target, weight 0.1). They can be reproduced with this code from the licensed corpus and are available from the corresponding author for verification under the dataset licence. No token-level data are included.

Note: the ranking script (`code/s5.5-rev/targets.py`) applies the same exclusion of the one خ token as the learner analysis (N = 205 for خ; 9,766 tokens). Until 16 Oct 2026 it used the table before this exclusion (N = 206); the fix changed only SS values in the second decimal (e.g. current layout 3.5705 → 3.5692 cm), not any SCS or rank.
