# s5.5-rev

Final exhaustive layout ranking for Section 4.4: `targets.py` builds the ranking targets, and `s55_rev.py` scores all 6^8 = 1,679,616 letter-to-channel assignments (1,679,610 after removing the six single-channel ones) by SCS and SS, ranking the current layout, RD1 and baselines B1–B3 in the unrestricted, collision-free (259,200) and collision-free, place-order-preserving (548) sets, with leave-one-term-out weights, the Pareto front and the learner bootstrap; `s55_rev_percentiles.py` adds percentiles and gaps to the best, and `s55_rev_fig.py` draws Figure 6.
It produced the reported numbers in Table 8, Figure 6, Tables S12–S15 and the ranking rows of Table S16.
It depends on `$LMP_DATA/s54` (`s54_pairs.csv`, `tokens.csv` from `s5.4`), `$LMP_DATA/full52-v2/results_v2.json`, `s5.1-rev/s51r_epg_table.csv` (B2) and, as a cross-check, `s5.4-rev/s54r_pairs.csv`; run `s55_rev.py` before the other two.
