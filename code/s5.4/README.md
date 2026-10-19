# s5.4

First learner-confusion analysis for Section 4.3: `s54_parse.py` turns every L2-KSU TextGrid into a token table, `s54_analysis.py` builds the within-8 confusion matrices, PanPhon distances and first Mantel tests, and `s54_figures.py` draws the first figures.
Its Mantel results and figures (ض at 11.0 cm, erroneous خ token still included) were superseded by `s5.4-rev`, but the code is still required: `s5.4-rev/s54_rev.py` runs the first part of `s54_analysis.py`, and `s5.5-rev/targets.py` reads `tokens.csv` and `s54_pairs.csv` from `$LMP_DATA/s54`.
It depends on the L2-KSU corpus (`$LMP_DATA/l2ksu`) and `$LMP_DATA/full52-v2/results_v2.json`; its outputs are token-level or L2-KSU-derived (CC BY-NC-ND 4.0) and must not be redistributed.
