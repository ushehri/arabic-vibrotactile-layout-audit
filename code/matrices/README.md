# matrices

`export_matrices.py` writes the complete 8 × 8 letter matrices (articulatory and phonological, learner confusion counts, native acoustic without L2-KSU, sleeve distances) as CSV plus `manifest.json`: Supplementary Tables S17–S20 and the repository folder `results/matrices/`.
It runs no new analysis but rereads the final stored outputs, so its files are final; acoustic matrices computed from L2-KSU audio are deliberately excluded (CC BY-NC-ND 4.0).
It depends on `s5.5-rev/targets.py` (executed in place) and on the outputs of `s5.1-rev`, `s5.4-rev` and `full-5.2-v2`, and writes to `$LMP_DATA/results/matrices/` (the tracked `results/matrices/` files are copies from there).
