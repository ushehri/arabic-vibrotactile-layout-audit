# s5.2-rev

Final acoustic tests for Section 4.2: two-sided Mantel tests with ض at 6.25 cm (`s52_rev_main_table.py`), the three ض codings for ALL and ALL without L2-KSU (`s52_rev_mantel.py`), and robustness checks for onset windows (`s52_rev_window.py`, `s52_rev_window_mantel.py`), Iqra clips duplicating Common Voice (`s52_rev_overlap.py`, `s52_rev_dedup.py`), the alignment landmark audit (`s52_rev_audit.py`) and the reproduction diagnostic (`s52_rev_repro_check.py`).
It produced the reported numbers in Table 6, Tables S5–S7, the acoustic rows of Table S16 and the robustness and alignment-audit statements of Section 4.2.
It depends on the `full-5.2-v2` outputs in `$LMP_DATA/full52-v2` and the L2-KSU token table in `$LMP_DATA/full52` (the window, audit and reproduction scripts also reread the audio); run `s52_rev_window.py` before `s52_rev_window_mantel.py` and `s52_rev_repro_check.py`.
