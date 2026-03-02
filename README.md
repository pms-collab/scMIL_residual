# Hierarchical Residual Multi-Head MIL for Progressive Disease Staging (scRNA-seq)
**Residualized severity classification + (Experimental) Gated Orthogonal Residual (GOR)**

This repository implements a bag/donor-level framework for progressive disease staging
(**Control / Mild / Severe**) from scRNA-seq, designed for **severe-specific subpopulation/state discovery**
through cell-level evidence scoring.

> **Data and pretrained embeddings are NOT included.**  
> Users must obtain the dataset and run embeddings (e.g., **scVI**) themselves.  
> This repo provides the **modeling, training, evaluation, and evidence-extraction** pipeline.

---

## 1) Problem

Progressive disease modeling with scRNA-seq is hard because:

- **Confounding / shortcut learning**: site/batch effects can dominate stage classification.
- **Heterogeneity**: multiple disease programs/subtypes coexist in patient cohorts.
- **Rare severe states**: severe-specific cellular programs are sparse and diluted by donor-level averaging.
- We want both:
  1) **staging performance** (C/M/S), and
  2) **cell-level evidence** for severe progression.

---

## 2) Hypothesis: Severe = amplified mild-shared + de novo severe-specific

We model the severe-stage signal as:
- **Amplified mild-shared signal** (shared disease program with larger magnitude), plus
- **De novo severe-specific signal** (additional program not explainable by mild-shared components).

Two-stage strategy:

- **Stage A (Residual baseline)**: emphasize specificity for **de novo** severe evidence by removing mild/disease-aligned components.
- **Stage B (GOR, experimental)**: partially retain mild/disease-aligned components to capture **amplification** effects.

---

## 3) Step 0) Embedding (scVI; optional pipeline module)

We treat scVI as a feature extractor producing per-cell latent embeddings.

- Input: raw scRNA-seq counts + batch/site covariates (if available)
- Output: per-cell embeddings \( h_i \in \mathbb{R}^d \)

scVI is used to reduce technical confounders before MIL aggregation.

> This repo may include scVI scripts under `pipelines/scvi/`, but ships no data/weights.

---

## 4) Core modeling idea: Dual-branch attention MIL + residualized severity classifier

### 4.1 Attention-based MIL (bag representation from cells)
Given a bag \(X=\{h_i\}_{i=1}^{n}\) of cell embeddings, we build bag-level representations via **attention-based MIL**:
\[
z = \sum_{i=1}^{n}\alpha_i h_i,\quad \alpha = \mathrm{softmax}(\cdot)
\]

We use two independent attention MIL branches on the same bag:

- **Disease branch (D)** produces \(z_d\): enriched for onset/disease-common evidence.
- **Severity branch (S)** produces \(z_s\): enriched for progression-related evidence.

> MIL reference: Ilse, Tomczak, Welling, *Attention-based Deep Multiple Instance Learning*, 2018. :contentReference[oaicite:0]{index=0}

### 4.2 Critical design constraint (what is fed to the severity classifier)
A key design choice is:

- Severity branch input is still cell embeddings \(h_i\).
- The branch aggregates cells into a bag vector \(z_s\).
- **However, the severity classifier does NOT take \(z_s\) directly.**
  It takes a residualized vector:
  \[
  \tilde z_s = (I-P)z_s
  \]
  where \(P\) is derived from the disease branch prototypes.

This forces severity prediction to rely on components **orthogonal** to the learned disease/mild-common subspace.

---

## 5) Stage A: Residual Hypothesis baseline (de novo severe-specific)

### Step 1) Disease branch learns a K-program disease subspace and pools to \(p_{\text{sick}}\)

**Disease MIL aggregation**
\[
z_d=\sum_i \alpha^{(d)}_i h_i
\]

**K disease programs (prototypes)**
We represent heterogeneity via \(K\) prototype weight vectors:
\[
W=[w_1;\dots;w_K]\in\mathbb{R}^{K\times d},\quad w_k\in\mathbb{R}^{d}
\]

**Per-program logits and probabilities**
\[
s_k = w_k^\top z_d + b_k,\qquad p_k=\sigma(s_k),\quad k=1,\dots,K
\]

**Pooling across subtypes to obtain \(p_{\text{sick}}\)**
We interpret “sick” as activation of **any** disease program:

- **Max pooling (default)**
\[
p_{\text{sick}}=\max_{k} p_k
\]
This is the explicit “OR over subtypes”: a bag is sick if it aligns strongly with at least one disease program.

- **Noisy-OR (optional)**
\[
p_{\text{sick}} = 1 - \prod_{k=1}^{K}(1-p_k)
\]
This is a smooth OR alternative that aggregates evidence across multiple moderately active programs.

**Hard orthonormality (recommended)**
To make projection well-defined, we construct \(W\) via differentiable QR:

- Learn \( V \in \mathbb{R}^{d\times K} \)
- \( Q,R=\mathrm{QR}(V)\), \(Q\in\mathbb{R}^{d\times K}\)
- \( W = Q^\top \in\mathbb{R}^{K\times d} \)

This yields a valid projector:
\[
P = W^\top W,\quad P^2=P,\quad P^\top=P
\]

### Step 2) Severity branch aggregates cells to \(z_s\) (still cell → bag)
\[
z_s=\sum_i \alpha^{(s)}_i h_i
\]

### Step 3) Residualize *before* severity classification (classifier input is \(\tilde z_s\))
\[
\tilde z_s = (I-P)z_s = z_s - Pz_s
\]

**Crucially, the severity classifier receives only \(\tilde z_s\), not \(z_s\):**
\[
p_{\text{sev}\mid\text{sick}} = \sigma\!\left(H_{\text{sev}}(\tilde z_s)\right)
\]

Interpretation: \(p_{\text{sev}\mid\text{sick}}\) is driven by components that cannot be explained by the disease/mild-common subspace, i.e., candidate **de novo severe-specific** evidence.

---

## 6) Stage B: GOR (Gated Orthogonal Residual; experimental)

Residual baseline prioritizes de novo evidence by discarding aligned components \(Pz_s\).
If severe includes **amplification of mild-shared programs**, discarding \(Pz_s\) may reduce sensitivity.

GOR augments the severity classifier input with a **gated** portion of the aligned component.

Decompose:
\[
z_{\parallel}=Pz_s,\quad \tilde z_s=(I-P)z_s
\]

A practical parameterization uses coefficients on disease programs:
\[
a = z_s W^\top \in \mathbb{R}^K,\quad z_{\parallel}=aW
\]
Gate:
\[
\beta=\sigma(\beta_{\text{logit}})
\]
Keep:
\[
a_{\text{keep}} = a\odot (1-\beta)
\]

Classifier input:
\[
z_{\text{prog}}=[\,\tilde z_s\;\|\;a_{\text{keep}}\,]
\]
\[
p_{\text{sev}\mid\text{sick}}=\sigma\!\left(H_{\text{sev}}(z_{\text{prog}})\right)
\]

> **Status:** Experimental extension. Risks include gate collapse and overfitting under small-N / noisy labels.  
> Report GOR results separately from the residual baseline.

---

## 7) Hierarchical/ordinal staging (Control/Mild/Severe)

We enforce “Severe ⊆ Sick” by composing probabilities:

\[
p(C)=1-p_{\text{sick}},\quad
p(M)=p_{\text{sick}}(1-p_{\text{sev}\mid\text{sick}}),\quad
p(S)=p_{\text{sick}}p_{\text{sev}\mid\text{sick}}
\]

This guarantees:
\[
p(S) \le p_{\text{sick}}
\]

### Optional: CORAL ordinal head
Instead of product-gating, we can treat C/M/S as an ordinal target and use **CORAL** (Consistent Rank Logits):

- Predict rank probabilities \(P(Y>0)\) and \(P(Y>1)\) with rank consistency.
- Convert them to class probabilities:
  \[
  P(Y=0)=1-p_1,\quad P(Y=1)=p_1-p_2,\quad P(Y=2)=p_2
  \]

> CORAL reference: Cao, Mirjalili, Raschka, *Rank consistent ordinal regression for neural networks with application to age estimation*, Pattern Recognition Letters (2020). :contentReference[oaicite:1]{index=1}

---

## 8) Cell-level evidence: attribution over attention (planned)

We do **not** treat attention weights as faithful explanations (**attention ≠ attribution**).
Attention indicates “where the model focuses” but not “what caused the logit”.

Therefore, cell-level contribution is quantified via attribution-style proxies tied to output logits:

- **Integrated Gradients (IG)** from cell embeddings \(h_i\) to:
  - sick logit, and/or
  - sev|sick logit (preferred for severe evidence)
- Alternative logit-contribution proxies:
  - gradient×input,
  - leave-one-out / logit-delta approximations (when feasible)

Planned outputs:
- ranked cells by attribution score,
- severe-enriched candidate subpopulations,
- agreement/disagreement between attention vs attribution rankings.

---

## 9) Repository contents (expected)

- `pipelines/scvi/`: scVI training/export scripts (optional; no data/weights shipped)
- `src/`: core modeling code
  - attention MIL branches (disease/severity)
  - disease prototypes \(W\), projector \(P=W^\top W\)
  - residualized severity classifier input \(\tilde z_s=(I-P)z_s\)
  - experimental GOR variants
  - ordinal composition + optional CORAL head
- `scripts/`: train/eval/diagnostics
- `docs/`: method + input format + reproduction notes
- Standardized run artifacts:
  - `metrics.json`, `predictions.csv`, confusion matrices
  - (optional) beta diagnostics, attribution summaries

---

## 10) Results (current status)

Observed behavior in the current experimental setup:
- validation performance can be strong, but **Val → Test drop** can occur in 3-class staging.
- conditional severity metrics may be more stable than 3-class accuracy, consistent with:
  - small sample size,
  - proxy/noisy severity labels (e.g., hospitalization-based definitions),
  - remaining confounding.

Interpretation:
- treat this framework primarily as a **mechanistic probe** for evidence discovery, not yet a final generalizable staging model.

---

## 11) Limitations

- **Small-N vs high-dimensional embeddings** (bag-level donors limited).
- **Label noise / proxy severity** can cap generalization.
- scVI reduces batch effects but may not remove all confounding.
- Evidence requires biological validation and stability checks across splits.

---

## 12) Next steps

- Stronger leakage audits (donor overlap, class balance reports, split sensitivity).
- Domain invariance objectives (adversarial site removal / MMD / calibration).
- External validation on datasets with clinically grounded severity labels.
- Evidence validation: enrichment for plausible severe-associated cell programs.

---

## 13) Citation / Acknowledgements

- Dataset: Ren et al. (COVID-19 scRNA-seq). Users must obtain data from the original source.
- Embedding: scVI / scvi-tools.
- MIL: Ilse, Tomczak, Welling (2018), Attention-based Deep Multiple Instance Learning. :contentReference[oaicite:2]{index=2}
- CORAL: Cao, Mirjalili, Raschka (2020), Rank consistent ordinal regression for neural networks. :contentReference[oaicite:3]{index=3}

---

## 14) Quickstart (dummy run)

> This will be enabled once scripts are added. Intended interface:

1) Generate dummy bags:
`python scripts/make_dummy_data.py --out data/dummy`

2) Train residual baseline:
`python scripts/train.py --config configs/residual_baseline.yaml --data data/dummy`

3) Evaluate:
`python scripts/evaluate.py --run runs/<EXP_ID>`
