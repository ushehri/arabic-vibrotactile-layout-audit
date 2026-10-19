# s5.5

First layout-ranking runs for Section 4.4: `s55_pilot.py` and `s55_full.py` rank permutations of the eight letters over fixed sleeve slots against PanPhon, place, confusion and MFCC targets, and `s55_figures.py` plots them.
They were superseded by `s5.5-rev` (all 6^8 channel assignments, spatial correspondence score, collision-free and place-order-preserving sets); nothing from this folder appears in the paper.
They depend on `s54_pairs.csv` and `tokens.csv` in `$LMP_DATA/s54` and on `$LMP_DATA/full52-v2/results_v2.json`, and write to `$LMP_DATA/s55/pilot` and `$LMP_DATA/s55/full`.
