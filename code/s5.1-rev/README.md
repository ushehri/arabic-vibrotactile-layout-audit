# s5.1-rev

Final articulatory analysis for Section 4.1: exact order tests over all 8! = 40,320 orderings for the ض midpoint (6.25 cm), start (11.0 cm) and end (1.5 cm) codings, extra EPG place measures and speaker-level ultrasound tests (`s51_rev.py`), the authoritative EPG extraction (`s51_rev_epg_table.py`) and Figure 1 (`s51_rev_fig1.py`).
It produced the reported numbers in Table 4, Figure 1, the EPG values of Table 5, Tables S1–S4, Figure S1 and the articulatory rows of Table S16; the ultrasound t-tests (df = 4) quoted in Section 4.1 are not written by any script and were checked against `s51r_uti_speaker_level.csv`.
It depends on the KAPD EPG files (`$LMP_DATA/kapd/P_sample`) and the ultrasound cells from `s5.1/s51_uti_emphasis.py`; its `s51r_epg_table.csv` is read by `s5.5-rev` (baseline B2) and `matrices`.
