# scMIL_residual: Residualized Multi-Head Attention MIL for Progressive Disease Staging (scRNA-seq)
**Residualized severity classification + (Experimental) Gated Orthogonal Residual (GOR)**

## TL;DR
This repo implements a bag/donor-level framework for progressive disease staging (**Control / Mild / Severe**) from scRNA-seq, designed for **severe-specific subpopulation/state discovery** via cell-level evidence scoring (planned attribution).

Core idea: learn a **disease prototype subspace** (heterogeneous K programs) and force the severity head to use **orthogonal residual evidence** (plus an experimental gated aligned component via GOR).  
**Key constraint:** the severity classifier never sees the raw severity summary `z_s`; it only sees the orthogonal residual `(I−P)z_s` (plus optional gated aligned terms in GOR).

> **Data and pretrained embeddings are NOT included.**
> Users must obtain the dataset and generate cell embeddings (e.g., scVI) themselves.

---

## What this contributes
- **Dual-branch attention MIL** on the same bag:
  - **Disease head:** learns `p(sick)` via K-prototype pooling over heterogeneous disease programs.
  - **Progression head:** learns `p(sev_given_sick)` from residual evidence orthogonal to the disease subspace.
- **Prototype disease subspace (K programs)** → well-defined projector onto the disease subspace (span of prototypes).
  - Configurable: `prototype_mode: general` (default, exact projector via `(W W^T)^{-1}`) or `prototype_mode: qr` (hard orthonormal via QR).
- **Residualized severity constraint (non-negotiable):** severity classifier never sees raw severity summary; it sees only the **orthogonal residual**.
- **(Experimental) GOR:** re-introduces a *gated* portion of aligned components to capture “severe = amplified mild-shared” while reducing shortcut takeover.
- **Ordinal staging composition:** enforces **Severe ⊆ Sick** by construction.

---

## Core idea (diagram)
```mermaid
flowchart LR
  X["Bag of cell embeddings (h_i)"] --> D["Disease MIL branch"] --> zd["z_d"]
  X --> S["Severity MIL branch"] --> zs["z_s"]

  zd --> W["Prototypes W (K programs; optional QR-orthonormal)"]
  W --> P["Projector P (disease subspace)"]

  zs --> R["Residual r = (I − P) z_s"]
  R --> Sev["Severity head"] --> psev["p(sev_given_sick)"]

  zd --> Sick["Disease head"] --> psick["p(sick)"]
  psick --> Staging["Compose C/M/S (Severe ⊆ Sick)"]
  psev --> Staging --> yhat["p(C), p(M), p(S)"]
````

---

## Empirical status

* Strong validation is often possible, but **Val → Test drop** can occur for 3-class staging.
* Conditional severity (`p(sev_given_sick)`) can be more stable than direct 3-class accuracy.
* Observed failure mode: 3-class generalization is sensitive to donor split; conditional severity tends to be more stable.
* Interpretation: use as a **mechanistic probe for evidence discovery** under small donor N + noisy/proxy severity labels; staging performance is secondary at current data quality.

---

## Method details

* Full method description (the long, math-heavy writeup): **`docs/method.md`**

---

## Evidence extraction (planned)

Attention is not attribution. Planned cell-level evidence scoring:

* Integrated Gradients on the `sev_given_sick` logit w.r.t. cell embeddings
* gradient×input or leave-one-out approximations when feasible

This repo already provides training/evaluation/export hooks; attribution scripts will be added next.

---

## Next steps (short list)

* Stronger leakage audits (donor overlap, split sensitivity, class imbalance reports)
* Domain invariance objectives (site-adversarial / MMD / calibration)
* External validation on datasets with clinically grounded severity labels
* Evidence validation: enrichment for known severe-associated programs + stability across seeds

---

<details>
<summary><b>Reproducibility (if you want to run it)</b></summary>

### Install

```bash
pip install -e .
```

### Data contract

All training/eval scripts consume a single file: `bags.npz` (not provided). Typical contents:

* `X`: float32 `[N_bags, CAP, D]` padded cell embeddings
* `mask`: bool `[N_bags, CAP]` valid-cell mask
* `y`: int64 `[N_bags]` with mapping `{control:0, mild/moderate:1, severe/critical:2}`
* `split`: int8 `[N_bags]` with mapping `{train:0, val:1, test:2}` **or** `idx_train/idx_val/idx_test`
* metadata arrays: `sample_id`, `donor_id`

### Quickstart (dummy)

```bash
python scripts/make_dummy_data.py --out data/dummy --seed 0
python scripts/train.py --config configs/residual_baseline.yaml --data data/dummy/bags.npz --exp_id dummy_resid
python scripts/evaluate.py --run runs/dummy_resid --data data/dummy/bags.npz
```

### Real data (recommended flow)

> Note: `tools/` contains **reference implementations** and may require dataset-specific adaptation.

1. (Optional) preprocess h5ad with desired cell cap: `tools/preprocess/build_capped_h5ad.py`
2. Donor-holdout split: `scripts/make_splits_donor_holdout.py`
3. (Optional) scVI embedding: `tools/embeddings/scvi/train_scvi.py` + `tools/embeddings/scvi/export_latent.py`
4. Build MIL bags: `scripts/build_bags_from_h5ad.py`
5. Train/eval: `scripts/train.py`, `scripts/evaluate.py`

</details>

<details>
<summary><b>Repository map</b></summary>

* `src/scmil_residual/` : core library (models, training, evaluation, audits)
* `scripts/`            : CLI entrypoints (build_bags / train / evaluate / dummy)
* `configs/`            : experiment configs (residual baseline, GOR)
* `tools/`              : optional reference scripts (preprocess, scVI embedding)
* `docs/`               : method writeup
* `notebooks/`          : original development notebooks (archived)

</details>
```
