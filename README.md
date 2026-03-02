# Hierarchical Residual Multi-Head MIL for Progressive Disease Staging (scRNA-seq)
**Residualized severity classification + (Experimental) Gated Orthogonal Residual (GOR)**

This repository implements a bag/donor-level framework for progressive disease staging (**Control / Mild / Severe**) from scRNA-seq, designed for **severe-specific subpopulation/state discovery** via cell-level evidence scoring.

> **Data and pretrained embeddings are NOT included.**  
> Users must obtain the dataset and generate embeddings (e.g., **scVI**) themselves.  
> This repo provides the **modeling + training + evaluation + evidence extraction** pipeline.

---

## 0) Problem

Progressive disease modeling with scRNA-seq is hard because:

- **Confounding / shortcut learning**: site/batch effects can dominate stage classification.
- **Heterogeneity**: multiple disease programs/subtypes coexist in patient cohorts.
- **Rare severe states**: severe-specific cellular programs are sparse and diluted by donor-level averaging.
- We want both:
  1) **staging performance** (C/M/S), and
  2) **cell-level evidence** for severe progression.

---

## 1) Hypothesis: Severe = amplified mild-shared + de novo severe-specific

We model the severe-stage signal as:

- **Amplified mild-shared signal** (shared disease program with larger magnitude), plus
- **De novo severe-specific signal** (additional program not explainable by mild-shared components).

This motivates two complementary objectives:

- **Objective A (de novo discovery)**: isolate severity evidence orthogonal to mild/disease-common structure.
- **Objective B (amplification-aware progression)**: allow partial reuse of mild/disease-aligned structure when it reflects amplification rather than confounding.

---

## 2) System overview (two-layer pipeline)

### Layer 1 — Representation (cell → embedding)
We assume each cell has an embedding $h_i \in \mathbb{R}^d$.  
Embeddings can be produced by scVI (recommended) or any comparable encoder.

- Input: raw scRNA-seq counts (+ batch/site covariates if available)
- Output: per-cell embeddings $h_i$

> This repo may include scVI scripts under `pipelines/scvi/`, but ships no data/weights.

### Layer 2 — Modeling (embedding bags → staging + evidence)
Given a bag (sample/donor) $X=\{h_i\}_{i=1}^{n}$, we predict C/M/S and compute cell-level evidence.

This layer contains:
- Dual-branch attention MIL (cell → bag summaries)
- Disease-subspace learning (K disease programs)
- Residualized severity classification (classifier sees residual, not raw severity summary)
- Ordinal staging structure (Severe ⊆ Sick)
- Evidence extraction (attribution-based; planned)

---

## 3) Modeling core: Dual-branch attention MIL + residualized severity classifier

### 3.1 Attention-based MIL (cell → bag)
We build a bag representation using attention-based MIL:

$$
z = \sum_{i=1}^{n}\alpha_i h_i,\quad \alpha = \mathrm{softmax}(\cdot)
$$

We use two independent branches on the same bag:

- **Disease branch (D)**: produces $z_d$ (onset / disease-common evidence)
- **Severity branch (S)**: produces $z_s$ (progression evidence candidate)

MIL reference: Ilse, Tomczak, Welling, *Attention-based Deep Multiple Instance Learning* (2018). :contentReference[oaicite:2]{index=2}

### 3.2 Non-negotiable constraint: what the severity classifier is allowed to see
The severity branch aggregates cells into a bag vector $z_s$.  
But the **severity classifier does NOT receive $z_s$**.

Instead it receives a residualized vector:

$$
\tilde z_s = (I-P)\,z_s
$$

where $P$ is a projector derived from the disease branch prototypes.

---

## 4) Module A — Disease branch: learn K disease programs and compute $p_{\text{sick}}$

### 4.1 Disease aggregation
$$
z_d=\sum_i \alpha^{(d)}_i h_i
$$

### 4.2 K disease programs (prototypes)
To represent heterogeneity, define $K$ prototype weight vectors:

$$
W=[w_1;\dots;w_K]\in\mathbb{R}^{K\times d},\quad w_k\in\mathbb{R}^{d}
$$

### 4.3 K logits → K probabilities → pool to $p_{\text{sick}}$
Compute per-program logits and probabilities:

$$
s_k = w_k^\top z_d + b_k,\qquad p_k=\sigma(s_k),\quad k=1,\dots,K
$$

Pool across programs to obtain “sick” probability (OR over subtypes):

- **Max pooling (default)**

$$
p_{\text{sick}}=\max_{k} p_k
$$

Interpretation: a bag is sick if it strongly matches *any* disease program.

- **Noisy-OR (optional)**

$$
p_{\text{sick}} = 1 - \prod_{k=1}^{K}(1-p_k)
$$

Interpretation: multiple moderately active programs can accumulate evidence smoothly.

### 4.4 Hard orthonormality (recommended) and projection operator
To make projection well-defined, construct $W$ via differentiable QR:

- Learn $V \in \mathbb{R}^{d\times K}$
- $Q,R=\mathrm{QR}(V)$, $Q\in\mathbb{R}^{d\times K}$
- $W = Q^\top \in\mathbb{R}^{K\times d}$

Then define the projector:

$$
P = W^\top W,\quad P^2=P,\quad P^\top=P
$$

---

## 5) Module B — Severity branch: aggregate, residualize, then classify

### 5.1 Severity aggregation (cell → bag)
$$
z_s=\sum_i \alpha^{(s)}_i h_i
$$

### 5.2 Residualization occurs at the classifier input
Remove disease/mild-aligned components from the bag vector $z_s$:

$$
\tilde z_s = (I-P)z_s = z_s - Pz_s
$$

Severity classifier input is **$\tilde z_s$** (not $z_s$):

$$
p_{\text{sev}\mid\text{sick}} = \sigma\!\left(H_{\text{sev}}(\tilde z_s)\right)
$$

Interpretation: $p_{\text{sev}\mid\text{sick}}$ is driven by components not explainable by the disease programs, i.e., candidate **de novo severe-specific** evidence.

---

## 6) Module C — (Experimental) GOR: amplification-aware severity input

Residual baseline prioritizes de novo evidence by discarding aligned components $Pz_s$.  
If severe includes amplification of mild-shared programs, discarding $Pz_s$ may reduce sensitivity.

GOR augments the severity classifier input with a gated portion of the aligned component.

Decompose:

$$
z_{\parallel}=Pz_s,\quad \tilde z_s=(I-P)z_s
$$

Use coefficients on disease programs:

$$
a = z_s W^\top \in \mathbb{R}^K,\quad z_{\parallel}=aW
$$

Gate:

$$
\beta=\sigma(\beta_{\text{logit}})
$$

Keep:

$$
a_{\text{keep}} = a\odot (1-\beta)
$$

Classifier input:

$$
z_{\text{prog}}=[\,\tilde z_s\;\|\;a_{\text{keep}}\,]
$$

$$
p_{\text{sev}\mid\text{sick}}=\sigma\!\left(H_{\text{sev}}(z_{\text{prog}})\right)
$$

> **Status:** Experimental. Risks include gate collapse and overfitting under small-N / noisy labels.  
> Report GOR separately from the residual baseline.

---

## 7) Module D — Staging head: Control/Mild/Severe with ordinal structure

We enforce “Severe ⊆ Sick” by composing probabilities:

$$
p(C)=1-p_{\text{sick}},\quad
p(M)=p_{\text{sick}}(1-p_{\text{sev}\mid\text{sick}}),\quad
p(S)=p_{\text{sick}}p_{\text{sev}\mid\text{sick}}
$$

Guarantee:

$$
p(S)\le p_{\text{sick}}
$$

### Optional: CORAL ordinal head (alternative to product composition)
Instead of product composition, treat C/M/S as ordinal and use **CORAL** (Consistent Rank Logits):

- Predict rank probabilities $p_1=P(Y>0)$ and $p_2=P(Y>1)$ with rank consistency.
- Convert to class probabilities:

$$
P(Y=0)=1-p_1,\quad P(Y=1)=p_1-p_2,\quad P(Y=2)=p_2
$$

CORAL reference: Cao, Mirjalili, Raschka, *Rank consistent ordinal regression for neural networks with application to age estimation*, Pattern Recognition Letters (2020).

---

## 8) Cell-level evidence: attribution over attention (planned)

We do **not** treat attention weights as faithful explanations (**attention ≠ attribution**).  
Attention indicates where the model focuses, not what caused the logit.

Planned evidence measures (logit-tied attribution proxies):

- **Integrated Gradients (IG)** from cell embeddings $h_i$ to:
  - sick logit, and/or
  - sev|sick logit (preferred for severe evidence)
- Alternative contribution proxies:
  - gradient×input,
  - leave-one-out / logit-delta approximations (when feasible)

Planned outputs:
- ranked cells by attribution score,
- severe-enriched candidate subpopulations,
- agreement/disagreement between attention ranking and attribution ranking.

---

## 9) Results (current status)

Observed behavior in the current setup:
- validation performance can be strong, but **Val → Test drop** can occur in 3-class staging.
- conditional severity metrics may be more stable than 3-class accuracy, consistent with small sample size, proxy/noisy severity labels, and residual confounding.

Interpretation: treat this framework primarily as a **mechanistic probe** for evidence discovery, not yet a final generalizable staging model.

---

## 10) Limitations

- Small-N vs high-dimensional embeddings (bag-level donors limited).
- Label noise / proxy severity can cap generalization.
- scVI reduces batch effects but may not remove all confounding.
- Evidence requires biological validation and stability checks across splits.

---

## 11) Next steps

- Stronger leakage audits (donor overlap, class balance reports, split sensitivity).
- Domain invariance objectives (adversarial site removal / MMD / calibration).
- External validation on datasets with clinically grounded severity labels.
- Evidence validation: enrichment for plausible severe-associated cell programs.

---

## 12) Citation / Acknowledgements

- Dataset: Ren et al. (COVID-19 scRNA-seq). Users must obtain data from the original source.
- Embedding: scVI / scvi-tools.
- MIL: Ilse, Tomczak, Welling (2018), Attention-based Deep Multiple Instance Learning.
- CORAL: Cao, Mirjalili, Raschka (2020), Rank consistent ordinal regression for neural networks.

---

## 13) Quickstart (dummy run)

> Enabled once scripts are added. Intended interface:

1) Generate dummy bags:  
`python scripts/make_dummy_data.py --out data/dummy`

2) Train residual baseline:  
`python scripts/train.py --config configs/residual_baseline.yaml --data data/dummy`

3) Evaluate:  
`python scripts/evaluate.py --run runs/<EXP_ID>`
