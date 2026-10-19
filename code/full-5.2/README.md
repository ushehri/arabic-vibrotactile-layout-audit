# full-5.2

First native-acoustic pipeline for Section 4.2 (folder names keep the draft numbering): CTC re-alignment of the L2-KSU native utterances (`align_l2ksu.py`), MFCC and XLS-R features for L2-KSU and the Halabi reference speaker (`features.py`), and L2-KSU-only matrices and Mantel tests (`analysis.py`, `report.py`); `tokens_l2ksu.py` is used only for its TextGrid reader and corpus path.
Its own Mantel results (ض at 11.0 cm, one-sided) were superseded by `full-5.2-v2` and `s5.2-rev`, but its token table and features (`tokens.csv`, `feats_mfcc.pkl`, `feats_ssl.npy`) remain the L2-KSU and Halabi inputs to the final Table 6, and `full-5.2-v2` imports its alignment, feature and statistics code.
It depends on L2-KSU (`$LMP_DATA/l2ksu`), the Halabi corpus (`$LMP_DATA/pilot/data/arabic-speech-corpus`), PyTorch and Transformers, and writes to `$LMP_DATA/full52` (run `align_l2ksu.py`, `features.py`, `analysis.py`, `report.py`).
