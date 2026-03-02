# Tools (reference implementations)

This directory contains **optional** scripts for dataset preparation and embedding.

- These scripts are **not** required to train/evaluate the MIL models.
- The only required input to `scripts/train.py` / `scripts/evaluate.py` is `bags.npz`.

Why this is separated:
- Preprocessing / scVI training is dataset-specific (QC, HVG, batch covariates, file formats).
- Keeping it under `tools/` prevents it from being mistaken as a guaranteed end-to-end pipeline.
