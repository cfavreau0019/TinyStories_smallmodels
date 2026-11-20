# Experiments

This document defines the experimental program for `TinyStories_smallmodels`.

The project studies whether normalized transformer representations exhibit measurable synchronization structure, whether that structure emerges during language-model training, whether attention acts as a coupling mechanism that reorganizes synchronization, and whether synchronization modes provide a useful basis for dimensionality reduction.

The experiments are intentionally ordered from weaker to stronger claims. Later experiments should be motivated by evidence from earlier stages rather than pursued simultaneously.

---

# Experimental conventions

## Indices

Throughout the experiments:

- $s$ = dataset sample
- $\tau$ = training checkpoint
- $\ell$ = transformer layer
- $i,j$ = token positions
- $h$ = attention head
- $n$ = hyperspherical harmonic degree

---

## Oscillator state

For sample $s$, layer $\ell$, token $i$, and checkpoint $\tau$, let

$$
x_{s,\ell,i,\tau} \in \mathbb{R}^d
$$

denote the hidden representation before RMSNorm.

Define the normalized state

$$
u_{s,\ell,i,\tau}
=
\frac{
x_{s,\ell,i,\tau}
}{
\sqrt{
\frac{1}{d}
\sum_k x_{s,\ell,i,\tau,k}^2
}
}.
$$

For unit-radius analysis define

$$
\hat u_{s,\ell,i,\tau}
=
\frac{u_{s,\ell,i,\tau}}{\sqrt d}.
$$

Then

$$
\hat u_{s,\ell,i,\tau}
\in
S^{d-1}.
$$

The working interpretation is:

- **Token representation** → generalized oscillator
- **Hidden dimensions** → coordinates of the oscillator state
- **RMSNorm** → hyperspherical constraint
- **Attention** → coupling network
- **Training step** → learning-time evolution
- **Layer depth** → sequence of learned representation spaces

---

# Shared evaluation dataset

Create one fixed validation subset before running synchronization experiments.

Recommended initial size:

- 2,000–5,000 TinyStories examples

All models and checkpoints should be evaluated on the same examples.

Save:

- `sample_id`
- `raw_text`
- `token_ids`
- `attention_mask`
- `sequence_length`

Do not resample the validation corpus between checkpoints.

---

# Shared checkpoints

At minimum save checkpoints near:

- step 0
- step 10
- step 30
- step 100
- step 300
- step 1,000
- step 3,000
- step 10,000
- later regular intervals
- final checkpoint

Early checkpoints should be sampled more densely because some of the largest representational changes may occur early in training.

Every checkpoint should preserve:

- model weights
- optimizer state
- training step
- validation loss
- training loss
- random seed
- configuration hash

---

# Shared synchronization quantities

## First-order synchronization

For a sequence of $T$ normalized token states,

$$
\hat u_1,\ldots,\hat u_T,
$$

define the mean direction

$$
\mathbf R_1
=
\frac{1}{T}
\sum_i \hat u_i.
$$

The synchronization magnitude is

$$
R_1
=
\|\mathbf R_1\|.
$$

---

## Hyperspherical degree power

Let $Y_{n,a}$ denote hyperspherical harmonics of degree $n$.

The corresponding empirical coefficients are

$$
c_{n,a}
=
\frac{1}{T}
\sum_i
Y_{n,a}^{*}(\hat u_i).
$$

Degree power is

$$
P_n
=
\sum_a
|c_{n,a}|^2.
$$

Where practical, estimate $P_n$ using the hyperspherical addition theorem:

$$
P_n
\propto
\frac{1}{T^2}
\sum_{i,j}
C_n^{(\alpha)}
\left(
\hat u_i^\top \hat u_j
\right),
$$

where

$$
\alpha
=
\frac{d-2}{2}.
$$

The initial degree range should be

$$
n=1,\ldots,8.
$$

Extend to higher degrees only after numerical stability is established.

---

## Normalized synchronization spectrum

Define

$$
p_n
=
\frac{P_n}
{\sum_r P_r}.
$$

---

## Effective synchronization dimension

Define

$$
D_{\mathrm{sync}}
=
\frac{1}
{\sum_n p_n^2}.
$$

Equivalently,

$$
D_{\mathrm{sync}}
=
\frac{
\left(\sum_n P_n\right)^2
}{
\sum_n P_n^2
}.
$$

Interpretation:

- Low $D_{\mathrm{sync}}$: synchronization power is concentrated into relatively few modes.
- High $D_{\mathrm{sync}}$: synchronization power is distributed across many modes.

---

# Experiment 0 — Train the baseline model

## Question

Can a small RMSNorm transformer learn TinyStories well enough to support later representation analysis?

## Input

- TinyStories training split
- TinyStories validation split
- small Llama-like causal decoder

Suggested starting architecture:

```yaml
n_layers: 6
hidden_size: 256
n_heads: 4
intermediate_size: 768
normalization: rmsnorm
activation: swiglu
position_encoding: rope
context_length: 256
```

## Procedure

Train until validation loss clearly plateaus or reaches acceptable TinyStories performance.

Save all predefined checkpoints.

## Outputs

- training loss by step
- validation loss by step
- checkpoints
- model configuration
- tokenizer

## Success criterion

The model should produce locally coherent TinyStories-like text and show clear improvement over initialization.

This experiment establishes infrastructure rather than evidence for synchronization.

## Figure

**Figure 0:** Training and validation loss versus optimization step.

---

# Experiment 1 — Static synchronization spectrum

## Question

Do trained transformer representations exhibit nontrivial hyperspherical synchronization structure?

## Input

- final trained checkpoint
- fixed validation set
- normalized token states

## Procedure

For each sample and layer:

1. Extract normalized token states.
2. Compute $R_1$.
3. Compute $P_n$ for $n=1,\ldots,8$.
4. Compute $D_{\mathrm{sync}}$.
5. Aggregate across samples.

## Saved tensors

Conceptually:

$$
P_n[s,\ell,n]
$$

along with

$$
R_1[s,\ell]
$$

and

$$
D_{\mathrm{sync}}[s,\ell].
$$

In code, use unambiguous field names rather than reusing $n$ for both the tensor name and harmonic index.

## Primary metrics

Mean spectral power across samples:

$$
\mathbb E_s[P_n].
$$

Variance across samples:

$$
\operatorname{Var}_s(P_n).
$$

Mean effective synchronization dimension:

$$
\mathbb E_s[D_{\mathrm{sync}}].
$$

## Null hypothesis

**H0:** The normalized hidden states exhibit no reproducible low-order synchronization structure beyond finite-sample and geometric effects expected from generic points on $S^{d-1}$.

## Alternative hypothesis

**H1:** Trained hidden representations show reproducible concentration in specific synchronization modes.

## Controls

Compare against synthetic random unit vectors with:

- the same hidden dimension
- the same sequence lengths
- the same number of samples

## Required figures

**Figure 1A:** Mean $P_n$ versus harmonic degree for each layer.

**Figure 1B:** Heatmap of layer × harmonic degree, colored by normalized synchronization power.

**Figure 1C:** $D_{\mathrm{sync}}$ versus layer.

---

# Experiment 2 — Synchronization through training

## Question

Does the synchronization spectrum evolve systematically during model training?

## Input

- all saved checkpoints
- the same fixed validation samples

## Procedure

Repeat Experiment 1 independently at each checkpoint.

Construct

$$
P_n(\ell,\tau)
$$

and

$$
D_{\mathrm{sync}}(\ell,\tau).
$$

## Primary metrics

For each degree and layer, define

$$
\Delta P_n(\ell,\tau)
=
P_n(\ell,\tau)
-
P_n(\ell,0).
$$

Also calculate

$$
\Delta D_{\mathrm{sync}}(\ell,\tau)
=
D_{\mathrm{sync}}(\ell,\tau)
-
D_{\mathrm{sync}}(\ell,0).
$$

Compare these trajectories with validation loss:

$$
\mathcal L_{\mathrm{val}}(\tau).
$$

## Null hypothesis

**H0:** Training does not systematically reorganize the synchronization spectrum beyond random checkpoint variation.

## Alternative hypothesis

**H1:** Training produces reproducible changes in synchronization structure associated with learned language performance.

## Important note

Do not assume that synchronization dimension must decrease.

Possible outcomes include:

- spectral concentration
- spectral broadening
- redistribution between harmonic degrees
- layer-specific transitions
- non-monotonic evolution

Any stable deviation from the random-initialization trajectory may be meaningful.

## Required figures

**Figure 2A:** Heatmap of layer × training checkpoint for $D_{\mathrm{sync}}$.

**Figure 2B:** Selected $P_n$ values versus training step.

**Figure 2C:** Validation loss and $D_{\mathrm{sync}}$ shown on aligned training axes.

**Figure 2D:** Synchronization spectra at initialization, early training, mid training, and the final checkpoint for selected layers.

---

# Experiment 3 — Language-learning control

## Question

Is synchronization reorganization specifically associated with learning coherent language?

## Models

Train identical architectures under at least two conditions.

### Condition A — Original TinyStories

Train normally on coherent stories.

### Condition B — Token-shuffled TinyStories

Preserve approximate unigram statistics while destroying syntax and narrative structure.

Optional later controls include:

- sentence-shuffled TinyStories
- random-token corpora
- frequency-matched synthetic corpora

## Matching requirements

Hold constant:

- architecture
- tokenizer
- optimizer
- learning-rate schedule
- number of optimization steps
- batch size
- random seed where possible

## Metrics

Compare

$$
P_n^{\mathrm{real}}
$$

against

$$
P_n^{\mathrm{shuffle}}
$$

at matched checkpoints.

Also compare

$$
D_{\mathrm{sync}}^{\mathrm{real}}
$$

against

$$
D_{\mathrm{sync}}^{\mathrm{shuffle}}.
$$

## Null hypothesis

**H0:** Synchronization changes are generic consequences of optimization or normalization and therefore appear similarly under shuffled training.

## Alternative hypothesis

**H1:** Coherent language training produces a distinct synchronization trajectory.

## Statistical analysis

Use:

- multiple random seeds
- bootstrap confidence intervals across stories
- paired comparisons at matched layers and checkpoints
- effect sizes

Avoid treating individual token observations as independent replicates.

## Required figures

**Figure 3A:** $D_{\mathrm{sync}}$ during real versus shuffled training.

**Figure 3B:** Difference spectrum

$$
P_n^{\mathrm{real}}
-
P_n^{\mathrm{shuffle}}.
$$

**Figure 3C:** Layerwise effect size at the final checkpoint.

---

# Experiment 4 — Within-block synchronization changes

## Question

Which components of a transformer block reorganize synchronization?

## Hook locations

Capture representations at:

- residual stream before attention
- RMSNorm before attention
- attention output
- residual stream after attention
- RMSNorm before MLP
- MLP output
- residual stream after MLP

## Procedure

Calculate synchronization spectra at each hook.

Define

$$
\Delta P_n^{\mathrm{attn}}
=
P_n^{\mathrm{post-attn}}
-
P_n^{\mathrm{pre-attn}},
$$

and

$$
\Delta P_n^{\mathrm{MLP}}
=
P_n^{\mathrm{post-MLP}}
-
P_n^{\mathrm{pre-MLP}}.
$$

## Null hypothesis

**H0:** Attention and MLP components do not produce systematically distinguishable synchronization changes.

## Alternative hypothesis

**H1:** Different transformer components produce characteristic changes in synchronization structure.

## Required figures

**Figure 4A:** Synchronization spectrum through the internal stages of a representative transformer block.

**Figure 4B:** Mean $\Delta P_n^{\mathrm{attn}}$ versus layer.

**Figure 4C:** Mean $\Delta P_n^{\mathrm{MLP}}$ versus layer.

**Figure 4D:** Attention versus MLP synchronization effect size.

---

# Experiment 5 — Attention as coupling

## Question

Does the attention matrix behave like a coupling structure whose properties predict synchronization changes?

## Inputs

For each attention head save:

- attention probabilities $A$
- pre-softmax attention logits $L$
- query vectors
- key vectors
- value vectors

## Coupling interpretation

Use

$$
K_{ij}
\sim
A_{ij}
$$

as the first empirical coupling approximation.

Analyze the pre-softmax logits independently as well, because softmax changes both scale and topology of the effective coupling.

## Attention statistics

Calculate quantities such as:

- row entropy
- effective rank
- spectral norm
- singular-value spectrum
- leading singular-vector concentration
- graph spectral gap
- sparsity
- asymmetry
- mean diagonal mass
- distance dependence

## Target variables

Test whether these quantities predict

$$
\Delta P_n^{\mathrm{attn}}
$$

and

$$
\Delta D_{\mathrm{sync}}^{\mathrm{attn}}.
$$

## Null hypothesis

**H0:** Attention-graph structure does not predict synchronization reorganization after controlling for layer and checkpoint.

## Alternative hypothesis

**H1:** Specific coupling properties systematically predict changes in synchronization modes.

## Analysis

Use:

- within-layer comparisons
- between-head comparisons
- between-sample comparisons
- training-time comparisons

Mixed-effects models may eventually be useful because observations are nested within:

- sample
- layer
- head
- checkpoint
- training seed

## Required figures

**Figure 5A:** Attention effective rank versus synchronization change.

**Figure 5B:** Attention entropy versus synchronization change.

**Figure 5C:** Head-level coupling statistics and synchronization response.

---

# Experiment 6 — Causal attention interventions

## Question

Does changing the coupling graph causally change synchronization?

## Interventions

### Temperature scaling

Define

$$
A(T)
=
\operatorname{softmax}(L/T).
$$

Suggested temperatures:

- $T=0.5$
- $T=0.75$
- $T=1.0$
- $T=1.5$
- $T=2.0$

### Uniform coupling

Replace attention with

$$
A_{ij}
=
\frac{1}{T_{\mathrm{seq}}},
$$

where $T_{\mathrm{seq}}$ denotes the number of causally available token positions.

### Identity interpolation

Define

$$
A_\lambda
=
(1-\lambda)I
+
\lambda A.
$$

Suggested values:

$$
\lambda
\in
\{0,\;0.25,\;0.5,\;0.75,\;1\}.
$$

When causal masking prevents a literal identity interpretation for some implementation details, preserve the causal structure appropriately.

### Edge permutation

Preserve row-wise attention weight distributions while changing which token receives each coupling.

### Weight permutation

Preserve the allowed topology but scramble coupling magnitudes.

### Head ablation

Disable selected attention heads.

## Outputs

For each intervention measure:

$$
\Delta P_n^{\mathrm{intervention}},
$$

$$
\Delta D_{\mathrm{sync}}^{\mathrm{intervention}},
$$

and

$$
\Delta\mathcal L_{\mathrm{LM}}.
$$

## Null hypothesis

**H0:** Synchronization statistics do not respond systematically to controlled modifications of attention coupling.

## Alternative hypothesis

**H1:** Changing coupling strength or topology produces predictable synchronization changes.

## Strong result

A particularly strong result would support the chain

$$
\text{coupling intervention}
\rightarrow
\text{synchronization change}
\rightarrow
\text{functional performance change}.
$$

## Required figures

**Figure 6A:** Attention temperature versus $D_{\mathrm{sync}}$.

**Figure 6B:** Attention temperature versus language-model loss.

**Figure 6C:** Synchronization change versus performance change.

**Figure 6D:** Natural attention versus shuffled-edge attention.

---

# Experiment 7 — RMSNorm gamma and ellipsoidal geometry

## Question

Does learned RMSNorm anisotropy modify synchronization structure?

## States

Before learned RMSNorm scaling:

$$
u_i.
$$

After learned per-dimension scaling:

$$
y_i
=
\Gamma_\ell u_i.
$$

For directional comparison define

$$
\tilde y_i
=
\frac{
\Gamma_\ell u_i
}{
\|\Gamma_\ell u_i\|
}.
$$

The scaled representation satisfies an ellipsoidal constraint of the form

$$
y_i^\top
\Gamma_\ell^{-2}
y_i
=
\text{constant}.
$$

## Spectra

Compute

$$
P_n^{\mathrm{sphere}}
$$

from $u_i$,

and

$$
P_n^{\mathrm{ellipse}}
$$

from $\tilde y_i$.

Define

$$
\Delta P_n^\Gamma
=
P_n^{\mathrm{ellipse}}
-
P_n^{\mathrm{sphere}}.
$$

## Model control

Train an otherwise identical model with:

```yaml
rmsnorm:
  gamma_init: 1.0
  gamma_trainable: false
```

## Null hypothesis

**H0:** Learned gamma scaling has no systematic effect on synchronization structure beyond trivial coordinate scaling.

## Alternative hypothesis

**H1:** Learned anisotropic geometry systematically reshapes the synchronization spectrum.

## Additional analyses

Track

$$
\gamma_{\ell,k}(\tau)
$$

through training.

Compare gamma anisotropy with

$$
D_{\mathrm{sync}}(\ell,\tau).
$$

Potential anisotropy metrics include

$$
\frac{
\max_k |\gamma_k|
}{
\min_k |\gamma_k|
},
$$

subject to numerical safeguards when the denominator is small, and

$$
\operatorname{Var}_k(\gamma_k).
$$

Other matrix-condition or log-scale dispersion measures may ultimately be more stable.

## Required figures

**Figure 7A:** Sphere versus ellipsoid synchronization spectra.

**Figure 7B:** $\Delta P_n^\Gamma$ versus layer.

**Figure 7C:** Gamma anisotropy through training.

**Figure 7D:** Trainable-gamma versus fixed-gamma models.

---

# Experiment 8 — Semantic synchronization

## Question

Does synchronization structure encode linguistically or semantically meaningful information?

## Candidate partitions

Compare synchronization conditioned on:

- token type
- syntactic class
- story location
- dialogue versus narration
- subject/object role
- repeated concepts
- semantic similarity
- paraphrase
- event type

## First analyses

Avoid immediately training complex nonlinear probes.

Start with:

- within-category spectrum similarity
- between-category spectrum similarity
- clustering
- nearest-neighbor retrieval
- simple linear probes

## Null hypothesis

**H0:** Synchronization spectra contain no additional semantic organization beyond generic activation statistics.

## Alternative hypothesis

**H1:** Different semantic or linguistic conditions produce reproducible synchronization signatures.

## Required figures

**Figure 8A:** Synchronization-spectrum similarity matrix across categories.

**Figure 8B:** Low-dimensional visualization of synchronization features.

**Figure 8C:** Within-category versus between-category similarity.

---

# Experiment 9 — Synchronization-based dimensionality reduction

## Question

Can synchronization modes reconstruct hidden representations efficiently?

## Important distinction

Degree power

$$
P_n
$$

is insufficient for reconstruction because it discards orientation within each harmonic subspace.

Reconstruction requires the complete coefficients

$$
c_{n,a}.
$$

## Procedure

Conceptually:

$$
H
\rightarrow
\{c_{n,a}\}
\rightarrow
\text{select synchronization modes}
\rightarrow
\hat H.
$$

Possible selection strategies include:

- largest coefficient magnitude
- largest variance across the dataset
- largest functional importance
- lowest harmonic degree
- greedy reconstruction improvement

## Baselines

Compare at matched retained dimension against:

- PCA
- truncated SVD
- random projection
- linear autoencoder

Optional later baseline:

- nonlinear autoencoder

## Geometric metrics

### Normalized reconstruction error

$$
E_{\mathrm{recon}}
=
\frac{
\|H-\hat H\|^2
}{
\|H\|^2
}.
$$

### Cosine similarity

$$
\cos(H,\hat H)
=
\frac{
H^\top \hat H
}{
\|H\|\|\hat H\|
}.
$$

### Explained variance

Use the same definition across synchronization and baseline methods.

## Functional metrics

Replace the original hidden state with the reconstructed state $\hat H$.

Measure

$$
D_{\mathrm{KL}}
\left(
p_{\mathrm{original}}
\|
p_{\mathrm{compressed}}
\right),
$$

plus:

- next-token cross entropy
- perplexity
- top-1 agreement
- top-$k$ agreement
- generation divergence

## Null hypothesis

**H0:** Synchronization modes provide no compression advantage over standard linear dimensionality-reduction methods.

## Alternative hypothesis

**H1:** A small number of synchronization modes preserve model behavior competitively or more efficiently.

## Required figures

**Figure 9A:** Reconstruction error versus retained dimensions.

**Figure 9B:** KL divergence versus retained dimensions.

**Figure 9C:** Synchronization compression versus PCA/SVD.

**Figure 9D:** Layerwise compression curves.

---

# Experiment 10 — Synchronization dimension predicts functional dimension

## Question

Does the measured synchronization dimension predict how compressible a transformer layer is?

## Definitions

For layer $\ell$, measure

$$
D_{\mathrm{sync}}^{(\ell)}.
$$

Separately determine the number of retained dimensions required to preserve model behavior.

Define

$$
d_{95}^{(\ell)}
$$

as the smallest retained representation dimension satisfying a predefined functional criterion.

Possible criteria include:

$$
D_{\mathrm{KL}}
<
\epsilon
$$

or preservation of at least 95% of a chosen model-quality metric.

## Primary test

Evaluate

$$
\operatorname{corr}
\left(
D_{\mathrm{sync}}^{(\ell)},
d_{95}^{(\ell)}
\right).
$$

## Null hypothesis

**H0:** Synchronization dimension does not predict functional compression dimension.

## Alternative hypothesis

**H1:** Layers with lower synchronization dimension require fewer retained modes to preserve model behavior.

## Strong interpretation

If supported, synchronization becomes not merely a descriptive statistic but a probe of effective computational dimensionality.

## Required figure

**Figure 10:** Scatterplot of $D_{\mathrm{sync}}$ against functional compression dimension, with layer identity annotated.

---

# Replication strategy

The exploratory phase may use:

- one model
- one random seed
- one architecture

No major scientific conclusion should rely on this phase.

Before paper-level claims, repeat key experiments across:

- 3–5 random seeds
- at least two model sizes
- multiple held-out text subsets

Priority replication experiments:

- Experiment 2
- Experiment 3
- Experiment 5
- Experiment 6
- Experiment 9

---

# Statistical strategy

The dataset contains hierarchical dependence.

Measurements are nested within:

- tokens
- stories
- layers
- heads
- checkpoints
- models
- random seeds

Avoid treating every token as an independent statistical sample.

Preferred approaches include:

- story-level aggregation
- bootstrap resampling over stories
- bootstrap or replication over seeds
- paired checkpoint comparisons
- mixed-effects models where appropriate
- effect sizes with confidence intervals

Report both statistical significance and effect magnitude.

---

# Numerical validation

Before interpreting any synchronization spectrum, validate the estimator on synthetic distributions.

Required synthetic cases include the following.

## Uniform hypersphere

Expected low-order power should match the finite-sample null.

## Concentrated unimodal distribution

The estimator should produce strong low-order synchronization.

## Mixture of clusters

The estimator should produce characteristic higher-order structure.

## Increasing noise

Synchronization power should degrade smoothly.

## Increasing sample size

Estimator variance should decrease as expected.

These tests should be implemented in something like:

```text
tests/test_synchronization_nulls.py
```

and should also produce diagnostic plots or notebooks.

---

# Ablation matrix

| Variable | Levels |
| --- | --- |
| Training | random / trained |
| Data structure | real / shuffled |
| Gamma | learned / fixed |
| Attention | natural / perturbed |
| Model size | small / medium |
| Seed | multiple |
| Layer | all |
| Checkpoint | early → late |
| Harmonic degree | low → higher |
| Compression method | synchronization / PCA / SVD / random |

---

# Paper-level claims and evidence

## Claim A — Training reorganizes hyperspherical synchronization structure

Required evidence:

- Experiment 2
- multiple seeds
- random-initialization comparison

---

## Claim B — The effect is associated with coherent language learning

Required evidence:

- Experiment 3

---

## Claim C — Attention acts as a coupling mechanism that reorganizes synchronization

Required evidence:

- Experiment 4
- Experiment 5
- Experiment 6

---

## Claim D — Learned RMSNorm anisotropy modifies synchronization geometry

Required evidence:

- Experiment 7

---

## Claim E — Synchronization structure reflects meaningful representation organization

Required evidence:

- Experiment 8

---

## Claim F — Synchronization modes provide a useful low-dimensional representation

Required evidence:

- Experiment 9
- Experiment 10

---

# Stop/go criteria

The project should use explicit decision points.

## Gate 1 — After Experiments 1–2

**Continue if:**

- synchronization spectra are stable across examples
- meaningful changes occur during training
- changes exceed random-hypersphere baselines

**Reconsider the representation if:**

- spectra are dominated by numerical artifacts
- results are indistinguishable from random normalized vectors
- no checkpoint evolution is visible

---

## Gate 2 — After Experiment 3

Continue toward language-specific claims if

$$
P_n^{\mathrm{real}}
\neq
P_n^{\mathrm{shuffled}}
$$

in a reproducible manner.

If not, synchronization may primarily reflect architecture or optimization rather than learned language structure.

---

## Gate 3 — After Experiments 5–6

Continue toward Kuramoto-style coupling claims if attention statistics and interventions produce reproducible synchronization changes.

Otherwise, retain synchronization as a descriptive representation analysis without claiming that attention implements an oscillator coupling mechanism.

---

## Gate 4 — After Experiment 9

Continue toward compression claims if synchronization-based reconstruction is competitive with standard baselines or reveals clearly distinct functional structure.

If not, the paper may still succeed as an analysis of synchronization during learning.

---

# Minimum viable paper

The smallest coherent paper does not require every experiment.

A strong minimal version would contain:

- Experiment 0 — model training
- Experiment 1 — synchronization spectrum
- Experiment 2 — training evolution
- Experiment 3 — language-learning controls
- Experiment 4 — pre/post-attention analysis
- Experiment 5 — coupling correlation
- Experiment 6 — causal attention intervention

This supports the central question:

> Does language-model training produce structured hyperspherical synchronization, and does attention causally organize it?

Compression and ellipsoidal geometry can then become extensions or follow-up work.

---

# Ideal full paper

The complete proposed story is:

$$
\text{training}
\rightarrow
\text{synchronization structure}
\rightarrow
\text{attention coupling}
\rightarrow
\text{collective modes}
\rightarrow
\text{effective dimensionality}
\rightarrow
\text{compression}.
$$

RMSNorm anisotropy provides the geometric bridge from hyperspherical to ellipsoidal synchronization.

---

# Immediate next experiment

Do not begin with all experiments simultaneously.

The first implementation target should be:

- one TinyStories model
- one random seed
- eight harmonic degrees
- all layers
- roughly ten training checkpoints
- one fixed validation set

Produce

$$
P_n(\ell,\tau)
$$

and

$$
D_{\mathrm{sync}}(\ell,\tau).
$$

The first research question is simply:

> **Does TinyStories training create a systematic synchronization-spectrum trajectory?**

Everything else follows from the answer.