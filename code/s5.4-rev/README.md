# s5.4-rev

Final learner-confusion analysis for Section 4.3: `s54_rev.py` excludes the erroneous خ token (9,766 tokens, 914 errors) and runs Mantel and partial Mantel tests for the 8-letter and 7-consonant sets under three ض codings, learner bootstraps (10,000), leave-one-word-out, the per-letter and pair tables and Figures 4 and S4; `s54_rev_sens.py` adds the sensitivity analyses and Figure 5.
It produced the reported numbers in Table 7, Figures 4 and 5, Tables S9–S11, Figure S4 and the learner rows of Table S16.
It depends on `s5.4` (it runs the first part of `s54_analysis.py` and reads `$LMP_DATA/s54/tokens.csv`) and on `$LMP_DATA/full52-v2/results_v2.json`; run `s54_rev.py` before `s54_rev_sens.py`.
