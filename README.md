# TinyStories_smallmodels

Small transformer training and synchronization-analysis experiments on the TinyStories dataset.

This repository studies whether transformer representations develop low-dimensional collective structure that can be characterized using ideas from **Kuramoto synchronization**, **Daido order parameters**, **hyperspherical harmonics**, and **information geometry**.

The core hypothesis is that normalization layers place token representations on a constrained geometric manifold, while attention supplies a heterogeneous coupling network between those token representations. Training may then produce structured collective modes that can be measured as a synchronization spectrum and potentially used for dimensionality reduction.

The repository is designed around small models trained from scratch on TinyStories so that the full training trajectory can be inspected.

---

## Research questions

The project is organized around several progressively stronger questions.

### 1. Do trained transformer representations exhibit synchronization structure?

For a normalized token representation

\[
u_i \in S^{d-1},
\]

we treat each token as a generalized oscillator whose state lies on a hypersphere.

The simplest collective order parameter is

\[
R_1
=
\left\|
\frac{1}{N}
\sum_i u_i
\right\|.
\]

More generally, the empirical distribution of normalized token states can be decomposed into hyperspherical harmonics,

\[
\rho(u)
=
\sum_{n,a}
c_{n,a}Y_{n,a}(u),
\]

with spectral power

\[
P_n
=
\sum_a |c_{n,a}|^2.
\]

The first question is whether these spectra are structured and reproducible in trained language models.

---

### 2. Does synchronization structure emerge during training?

Let

- \(\ell\) denote transformer layer,
- \(\tau\) denote training checkpoint,
- \(n\) denote harmonic degree.

We study

\[
P_n(\ell,\tau)
\]

and derived measures such as an effective synchronization dimension,

\[
D_{\text{sync}}
=
\frac{
\left(\sum_n P_n\right)^2
}{
\sum_n P_n^2
}.
\]

The goal is to determine whether training systematically reorganizes the synchronization spectrum.

This allows layer depth and training time to remain conceptually distinct:

\[
\ell
=
\text{where synchronization is represented},
\]

\[
\tau
=
\text{how synchronization develops during learning}.
\]

---

### 3. Is synchronization related to meaningful language learning?

Synchronization structure observed in a trained network may arise trivially from:

- RMS normalization,
- architectural constraints,
- optimization,
- token-frequency statistics,
- or generic low-rank structure.

Controls therefore include:

- randomly initialized models,
- normally trained TinyStories models,
- token-shuffled TinyStories,
- architecture variants,
- multiple random seeds.

The hypothesis is not simply that synchronization exists, but that its structure differs systematically when coherent language is learned.

---

### 4. Does attention act as a coupling network?

For attention head \(h\),

\[
A_{ij}^{(\ell,h)}
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}
{\sqrt{d_h}}
\right).
\]

We interpret \(A_{ij}\) as an empirical, directed, heterogeneous analogue of a Kuramoto coupling matrix

\[
K_{ij}.
\]

A hyperspherical Kuramoto-like interaction would have the form

\[
\dot u_i
=
\Omega_i u_i
+
\sum_j K_{ij}
P_{u_i}u_j,
\]

where

\[
P_{u_i}
=
I-u_i u_i^\top
\]

projects coupling forces onto the tangent space of the hypersphere.

The repository tests whether properties of the learned attention graph predict changes in synchronization structure across attention blocks.

---

### 5. Does attention causally alter synchronization?

Synchronization is measured immediately before and after attention.

For harmonic degree \(n\),

\[
\Delta P_n^{\text{attn}}
=
P_n^{\text{post-attn}}
-
P_n^{\text{pre-attn}}.
\]

Attention is then perturbed during inference using interventions such as:

- attention-temperature scaling,
- uniform attention,
- interpolation with identity coupling,
- shuffled attention edges,
- shuffled attention weights,
- head removal,
- low-rank attention approximations.

These interventions test whether modifying the coupling network predictably modifies the synchronization spectrum.

---

### 6. What role does RMSNorm geometry play?

Before learned RMSNorm scaling, normalized representations lie approximately on a hypersphere:

\[
u \in S^{d-1}.
\]

After learned per-dimension scaling,

\[
y=\Gamma u,
\]

where

\[
\Gamma
=
\operatorname{diag}
(\gamma_1,\ldots,\gamma_d),
\]

the representation lies on an ellipsoid:

\[
y^\top\Gamma^{-2}y
=
\text{constant}.
\]

We therefore compare synchronization measured:

1. intrinsically on the normalized hypersphere,
2. after learned \(\gamma\)-scaling,
3. in models where \(\gamma\) is fixed to one.

The goal is to determine whether learned anisotropic scaling reshapes synchronization structure.

---

### 7. Can synchronization structure be used for dimensionality reduction?

The long-term goal is to determine whether a small set of synchronization modes can approximate a much larger hidden representation.

The intended pipeline is

\[
H
\rightarrow
\text{synchronization transform}
\rightarrow
\text{top modes}
\rightarrow
\hat H.
\]

The reconstructed representation is evaluated using both geometric and functional metrics.

Geometric metrics include:

\[
\frac{
\|H-\hat H\|^2
}{
\|H\|^2
},
\]

cosine similarity, and reconstruction variance.

Functional metrics include:

- next-token cross-entropy,
- KL divergence between original and reconstructed logits,
- top-\(k\) prediction agreement,
- downstream generation quality.

Synchronization-based compression is compared against:

- PCA / truncated SVD,
- random projection,
- low-rank linear autoencoders,
- standard Fourier-like baselines where appropriate.

---

# Dataset

## TinyStories

The primary dataset is:

`roneneldan/TinyStories`

TinyStories contains short synthetic stories generated with deliberately simple vocabulary and narrative structure. It was designed to allow very small language models to learn coherent language behavior.

The dataset is useful here because:

- small models can be trained from scratch,
- many checkpoints can be saved,
- multiple random seeds are computationally feasible,
- model representations remain relatively interpretable,
- controlled corruption experiments are straightforward.

Example:

```python
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

train = dataset["train"]
validation = dataset["validation"]
```

A fixed validation subset should be created once and reused across every checkpoint and model.

---

# Model family

The initial model family is a small Llama-like causal decoder.

Recommended starting configuration:

```yaml
model:
  n_layers: 6
  hidden_size: 256
  n_heads: 4
  intermediate_size: 768

  normalization: rmsnorm
  positional_encoding: rope
  activation: swiglu

training:
  context_length: 256
  tokenizer_vocab_size: 8000
```

The exact architecture is intentionally small enough for repeated training runs.

The first objective is not state-of-the-art TinyStories performance.

The objective is experimental control.

---

# Representation convention

For sequence sample \(s\), layer \(\ell\), and token \(i\), let

\[
x_{s,\ell,i}
\in
\mathbb R^d
\]

denote the residual-stream representation.

After RMS normalization,

\[
u_{s,\ell,i}
=
\frac{
x_{s,\ell,i}
}{
\sqrt{
\frac{1}{d}
\sum_k x_{s,\ell,i,k}^2
}
}.
\]

Define

\[
\hat u_{s,\ell,i}
=
\frac{u_{s,\ell,i}}{\sqrt d}.
\]

Then

\[
\hat u_{s,\ell,i}
\in S^{d-1}.
\]

The working interpretation is:

```text
token                  -> oscillator
hidden dimensions      -> coordinates of oscillator state
RMSNorm                -> hyperspherical constraint
attention matrix       -> coupling matrix
training               -> evolution of learned synchronization structure
layer depth             -> sequence of distinct representation geometries
```

---

# Experiments

## Experiment 0 — Baseline model training

Train a small TinyStories model and save checkpoints throughout training.

Important checkpoints should include:

```text
step 0
step 10
step 30
step 100
step 300
step 1,000
step 3,000
...
final
```

Log:

- training loss,
- validation loss,
- gradient norm,
- learning rate,
- RMSNorm weights,
- attention statistics.

The random initialization checkpoint is required for all later comparisons.

---

## Experiment 1 — Static synchronization spectrum

Objective:

> Determine whether trained hidden states contain structured hyperspherical synchronization modes.

For each:

```text
sample
layer
token
checkpoint
```

extract normalized token states.

Compute low-order synchronization statistics.

### First-order synchronization

\[
R_1
=
\left\|
\frac1T
\sum_i u_i
\right\|.
\]

### Higher-order spectrum

Use hyperspherical harmonic degree \(n\).

Rather than explicitly constructing all harmonics, estimate degree power using the hyperspherical addition theorem.

For

\[
\alpha
=
\frac{d-2}{2},
\]

spectral degree power can be estimated using Gegenbauer polynomials:

\[
P_n
\propto
\frac{1}{T^2}
\sum_{i,j}
C_n^{(\alpha)}
(
u_i^\top u_j
).
\]

Initial range:

```text
n = 1 ... 8
```

or

```text
n = 1 ... 16
```

depending on numerical stability.

Outputs:

```text
P_n[sample, layer, checkpoint]
```

---

## Experiment 2 — Synchronization through training

Objective:

> Determine whether synchronization structure evolves systematically during training.

Compute

\[
P_n(\ell,\tau)
\]

for all saved checkpoints.

Derive normalized power:

\[
p_n
=
\frac{P_n}
{\sum_r P_r}.
\]

Compute effective synchronization dimension:

\[
D_{\text{sync}}
=
\frac{1}
{\sum_n p_n^2}.
\]

Primary plots:

```text
P_n vs training step
P_n vs layer
D_sync vs training step
D_sync vs layer
D_sync(layer, training_step)
```

Also compare against validation loss.

---

## Experiment 3 — Language-learning controls

Train models on:

### A. Original TinyStories

Normal training corpus.

### B. Token-shuffled TinyStories

Preserve approximate unigram distributions while destroying syntax and narrative structure.

### C. Random initialization

No training.

Optional additional controls:

- sentence-shuffled stories,
- story-level token permutation,
- random-token corpora matched by vocabulary,
- character-level corruption.

Compare:

\[
P_n^{\text{real}}
\]

against

\[
P_n^{\text{shuffled}}
\]

and

\[
P_n^{\text{random}}.
\]

---

## Experiment 4 — Synchronization inside transformer blocks

Measure synchronization at multiple points inside each transformer block.

Suggested hooks:

```text
residual_pre_attention
rmsnorm_pre_attention
attention_output
residual_post_attention
rmsnorm_pre_mlp
mlp_output
residual_post_mlp
```

For each harmonic:

\[
\Delta P_n^{\text{attn}}
=
P_n^{\text{post-attn}}
-
P_n^{\text{pre-attn}},
\]

and

\[
\Delta P_n^{\text{MLP}}
=
P_n^{\text{post-MLP}}
-
P_n^{\text{pre-MLP}}.
\]

Questions:

- Does attention increase particular synchronization modes?
- Does the MLP undo or reinforce them?
- Do different layers behave differently?
- Does this change during training?

---

## Experiment 5 — Attention as coupling

Store for every head:

```text
attention probabilities
pre-softmax attention logits
query vectors
key vectors
value vectors
```

Compute attention statistics including:

- entropy,
- effective rank,
- spectral norm,
- singular-value spectrum,
- graph spectral gap,
- sparsity,
- asymmetry,
- concentration.

Test whether attention structure predicts

\[
\Delta P_n^{\text{attn}}.
\]

Primary hypothesis:

\[
\boxed{
\text{attention coupling structure predicts synchronization reorganization}
}
\]

---

## Experiment 6 — Attention intervention

Modify attention during inference without retraining.

### Temperature

\[
A(T)
=
\operatorname{softmax}(L/T).
\]

### Identity interpolation

\[
A_\lambda
=
(1-\lambda)I+\lambda A.
\]

### Uniform coupling

\[
A_{ij}
=
\frac1T.
\]

### Edge shuffle

Preserve weights but randomize target tokens.

### Weight shuffle

Preserve topology but scramble coupling strengths.

### Head ablation

Disable one or more heads.

For each intervention measure:

\[
\Delta P_n,
\]

\[
\Delta D_{\text{sync}},
\]

and language-model performance.

---

## Experiment 7 — RMSNorm gamma and ellipsoidal geometry

For pre-\(\gamma\) state

\[
u_i
\]

and post-\(\gamma\) state

\[
y_i
=
\Gamma u_i,
\]

compute synchronization before and after anisotropic scaling.

Define unit Euclidean direction after scaling:

\[
\tilde y_i
=
\frac{
\Gamma u_i
}{
\|\Gamma u_i\|
}.
\]

Compare:

\[
P_n^{\text{sphere}}
\]

and

\[
P_n^{\text{ellipse}}.
\]

Define:

\[
\Delta P_n^\Gamma
=
P_n^{\text{ellipse}}
-
P_n^{\text{sphere}}.
\]

Train an additional model with

```text
RMSNorm gamma = 1
gamma_trainable = false
```

to test whether learned ellipsoidal geometry affects synchronization.

---

## Experiment 8 — Semantic synchronization

Test whether synchronization structure corresponds to linguistic or semantic categories.

Possible comparisons:

```text
nouns vs verbs
dialogue vs narration
story beginning vs ending
named entities vs function words
positive vs negative events
causal transitions
repeated concepts
paraphrased stories
```

Potential analyses:

- synchronization-spectrum similarity,
- clustering,
- nearest neighbors,
- linear probes on synchronization features,
- mutual information with linguistic labels.

The goal is not primarily classification performance.

The goal is to determine whether synchronization modes encode interpretable structure.

---

## Experiment 9 — Synchronization dimensionality reduction

The harmonic power spectrum alone is not sufficient for reconstruction.

For reconstruction, preserve full synchronization coefficients

\[
c_{n,a}
\]

rather than only degree power

\[
P_n
=
\sum_a |c_{n,a}|^2.
\]

Perform:

\[
H
\rightarrow
\{c_{n,a}\}
\rightarrow
\text{top synchronization modes}
\rightarrow
\hat H.
\]

Compare against:

```text
PCA
truncated SVD
random projection
linear autoencoder
```

Evaluate:

### Representation reconstruction

\[
\text{MSE}
=
\frac{
\|H-\hat H\|^2
}{
\|H\|^2
}.
\]

### Cosine similarity

\[
\frac{
H^\top\hat H
}{
\|H\|
\|\hat H\|
}.
\]

### Functional reconstruction

Inject \(\hat H\) back into the frozen transformer.

Measure:

```text
next-token loss
KL divergence of logits
top-k agreement
perplexity
generation quality
```

---

## Experiment 10 — Synchronization dimension vs functional dimension

For every layer estimate:

\[
D_{\text{sync}}^{(\ell)}.
\]

Separately determine the number of retained dimensions required to preserve model behavior:

\[
d_{95}^{(\ell)}.
\]

Test whether

\[
D_{\text{sync}}^{(\ell)}
\]

predicts

\[
d_{95}^{(\ell)}.
\]

This tests the strongest dimensionality-reduction hypothesis:

> Synchronization dimension reflects the effective computational dimensionality of a transformer representation.

---

# Repository structure

```text
TinyStories_smallmodels/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── model/
│   │   ├── tiny_5m.yaml
│   │   ├── tiny_15m.yaml
│   │   └── tiny_gamma_fixed.yaml
│   │
│   ├── training/
│   │   └── tinystories.yaml
│   │
│   └── experiments/
│       ├── synchronization.yaml
│       ├── attention_intervention.yaml
│       └── compression.yaml
│
├── src/
│   └── tinystories_smallmodels/
│       │
│       ├── data/
│       │   ├── tinystories.py
│       │   ├── corruption.py
│       │   └── tokenizer.py
│       │
│       ├── models/
│       │   ├── transformer.py
│       │   ├── attention.py
│       │   └── rmsnorm.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   └── checkpoints.py
│       │
│       ├── hooks/
│       │   ├── activations.py
│       │   ├── attention.py
│       │   └── rmsnorm.py
│       │
│       ├── synchronization/
│       │   ├── order_parameters.py
│       │   ├── gegenbauer.py
│       │   ├── hyperspherical.py
│       │   ├── spectra.py
│       │   └── effective_dimension.py
│       │
│       ├── interventions/
│       │   ├── attention_temperature.py
│       │   ├── attention_shuffle.py
│       │   └── head_ablation.py
│       │
│       └── compression/
│           ├── synchronization_transform.py
│           ├── reconstruction.py
│           └── baselines.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_activations.py
│   ├── compute_synchronization.py
│   ├── analyze_training_dynamics.py
│   ├── run_attention_interventions.py
│   ├── analyze_gamma.py
│   └── run_compression.py
│
├── notebooks/
│   ├── 01_training_dynamics.ipynb
│   ├── 02_synchronization_spectra.ipynb
│   ├── 03_attention_coupling.ipynb
│   ├── 04_gamma_geometry.ipynb
│   └── 05_compression.ipynb
│
├── results/
│   ├── spectra/
│   ├── interventions/
│   ├── compression/
│   └── figures/
│
├── checkpoints/
│
└── tests/
    ├── test_rmsnorm_geometry.py
    ├── test_order_parameters.py
    ├── test_gegenbauer.py
    └── test_reconstruction.py
```

---

# Suggested development order

The repository should be developed incrementally.

## Phase 1 — Training infrastructure

Implement:

```text
TinyStories loading
tokenization
small transformer
RMSNorm
training loop
checkpointing
validation
```

Deliverable:

```text
one successfully trained TinyStories model
```

---

## Phase 2 — Activation extraction

Add hooks for:

```text
pre/post RMSNorm
pre/post attention
pre/post MLP
attention matrices
attention logits
```

Deliverable:

```text
fixed validation corpus -> activation dataset
```

---

## Phase 3 — Synchronization spectrum

Implement:

```text
first-order order parameter
Gegenbauer estimator
low-degree hyperspherical spectrum
effective synchronization dimension
```

Deliverable:

```text
P_n(layer, checkpoint)
D_sync(layer, checkpoint)
```

This is the first major research checkpoint.

---

## Phase 4 — Controls

Run:

```text
random initialization
normal TinyStories
shuffled TinyStories
multiple random seeds
```

Deliverable:

```text
evidence that observed synchronization is not trivial
```

---

## Phase 5 — Attention mechanism

Analyze:

```text
attention graph statistics
pre/post attention synchronization
attention interventions
```

Deliverable:

```text
evidence connecting attention coupling to synchronization changes
```

---

## Phase 6 — Ellipsoidal geometry

Analyze:

```text
pre-gamma spectrum
post-gamma spectrum
fixed-gamma models
```

Deliverable:

```text
measurement of the effect of learned anisotropic normalization geometry
```

---

## Phase 7 — Compression

Implement full synchronization coefficients and inverse reconstruction.

Compare against PCA/SVD and evaluate reconstructed activations inside the model.

Deliverable:

```text
synchronization-based dimensionality-reduction benchmark
```

---

# Initial minimum viable experiment

Before implementing the full research program, answer one question:

> Does synchronization structure systematically change during TinyStories training?

Train one small RMSNorm transformer.

Use one random seed.

Save checkpoints.

For a fixed validation set, compute:

\[
P_n(s,\ell,\tau),
\qquad
n=1,\ldots,8.
\]

Then plot:

\[
P_n(\ell,\tau)
\]

and

\[
D_{\text{sync}}(\ell,\tau).
\]

If training produces a strong and reproducible restructuring of these quantities, proceed to the attention and compression experiments.

If not, revisit the synchronization representation before expanding the project.

---

# Working hypotheses

### H1

Transformer training reorganizes the hyperspherical synchronization spectrum of normalized token representations.

### H2

Synchronization structure differs between coherent language training and distribution-matched corrupted training.

### H3

Attention coupling predicts changes in synchronization structure.

### H4

Causal modification of attention coupling causes predictable changes in synchronization structure.

### H5

Learned RMSNorm scaling modifies synchronization through anisotropic ellipsoidal geometry.

### H6

The synchronization spectrum is lower-dimensional than the raw hidden representation.

### H7

Synchronization-based dimensionality reduction preserves model behavior with competitive or superior efficiency relative to conventional linear methods.

---

# Terminology

### Oscillator

A normalized token representation

\[
u_i\in S^{d-1}.
\]

### Coupling

The attention relationship between token oscillators, approximated by

\[
K_{ij}\sim A_{ij}.
\]

### Synchronization

Collective geometric organization of normalized token states.

### Synchronization spectrum

The distribution of representation power across hyperspherical harmonic degrees or modes.

### Synchronization dimension

An effective number of synchronization modes contributing substantially to the representation.

### Ellipsoidal synchronization

Synchronization analyzed after learned RMSNorm \(\gamma\)-scaling.

---

# Status

This repository is intended as an experimental research codebase.

Current priorities:

```text
[ ] TinyStories dataset pipeline
[ ] tokenizer
[ ] small RMSNorm transformer
[ ] training + checkpointing
[ ] fixed validation corpus
[ ] activation hooks
[ ] first-order synchronization metric
[ ] hyperspherical / Gegenbauer spectrum
[ ] training-time synchronization analysis
[ ] shuffled-data control
[ ] attention coupling analysis
[ ] RMSNorm gamma analysis
[ ] synchronization compression
```

---

# Longer-term directions

Potential follow-up work includes:

- larger language models,
- pretrained Llama-family models,
- reasoning datasets,
- synchronization during chain-of-thought generation,
- synchronization spectra across semantic tasks,
- comparison with information-geometric order parameters,
- wrapped Cauchy / Poisson-kernel formulations,
- learned synchronization transforms,
- synchronization-based neural architectures,
- ellipsoidal Kuramoto networks,
- compression of trained models using synchronization modes.

---

# Central idea

The project can be summarized as:

\[
\boxed{
\text{high-dimensional transformer state}
\rightarrow
\text{collective synchronization modes}
\rightarrow
\text{low-dimensional representation}
}
\]

Rather than asking only which Euclidean directions explain transformer representations, this project asks:

> **Which collective synchronization modes explain them?**