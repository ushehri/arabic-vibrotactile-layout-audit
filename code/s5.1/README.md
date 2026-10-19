# s5.1

First articulatory analysis for Section 4.1: order tests of the layout against the KAPD label, modern-literature and classical codes and the KAPD EPG summary (`s51_analysis.py`), the Lebanese ultrasound emphasis cells (`s51_uti_emphasis.py`), and figures plus an Excel sheet (`s51_figures_tables.py`).
The order tests and EPG figure were superseded by `s5.1-rev` (ض at the 6.25 cm midpoint, re-extracted EPG), but two parts are final: the ultrasound cells feed the speaker-level tests in `s5.1-rev` (Table S4), and `s51_figures_tables.py` draws Figure 2 (`s51_fig3_uti_emphasis.png`).
It depends on `$LMP_DATA/kapd/kapd_epg_summary.csv` and the Coretta and Sakr compendium in `$LMP_DATA/s51data`; `s51_figures_tables.py` also reads two hand-made tables (`s51_mismatch_table.csv`, `s51_agreement_extra.csv`, not produced by any script here) from `$LMP_DATA/results/s5.1/` before it reaches Figure 2.
