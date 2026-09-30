# ADR 0007 — LoRA design evolution (v1 → v2 → v3)

- **Status:** Accepted
- **Date:** M2.6 (2026-09-30)
- **Deciders:** project owner
- **Related:** ADR 0002 (base model), ADR 0005 (domain restriction), ADR 0006 (FP32 on T4)

## Context

Our first two QLoRA fine-tunes **both failed the eval gate**:

| Metric | Baseline | Target (min) | v1 | v2 |
|---|---|---|---|---|
| Math accuracy | 0.960 | >= 0.960 | 0.730 FAIL | 0.660 FAIL |
| Physics accuracy | 0.725 | >= 0.800 | 0.441 FAIL | 0.471 FAIL |
| OOD refusal rate | 0.000 | >= 0.950 | 0.960 OK | 0.000 FAIL |
| False refusal rate | 0.000 | <= 0.020 | 0.260 FAIL | 0.000 OK |

Both runs used the same recipe: LoRA on **all linear layers**
(`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), training
targets built by hand (reasoning + `\boxed{X}` for math, `\boxed{X}` alone
for physics, exact refusal string for OOD).

Two distinct failure modes appeared:

- **v1 (lr=1e-4, r=16, 3 epochs):** refusal learned strongly (0.96), but
  math and physics crashed by 23 / 28 points. False-refusal rate hit 26% —
  the model refused legitimate math questions phrased conversationally.
- **v2 (lr=2e-5, r=8, 2 epochs):** refusal signal too weak to learn at all
  (0.000), math still crashed (0.660), physics only marginally better than v1.

The problem was **not** hyperparameter tuning. It was the design of the
LoRA adapter and the training objective.

## Decision

v3 changes four independent design choices at once, each with a specific
mechanistic justification. The combination is the working hypothesis; we
document them separately so future readers can attribute wins/losses.

### 1. Rank-1 LoRA on MLP layers only

**Source:** "LoRA is All You Need for Safety Alignment of Reasoning LLMs"
(2025) — rank-1 updates suffice for safety alignment, and the `up_proj` /
`gate_proj` / `down_proj` modules are the most critical for behavior change.

**Rationale:**
- **Attention layers** (`q_proj, k_proj, v_proj, o_proj`) hold the base
  model's chain-of-thought reasoning capability. Touching them is what
  caused v1/v2 math crash.
- **MLP layers** hold factual and behavioral associations. Safer to
  modify for refusal behavior.
- **Rank-1** minimizes the update's effective dimension. If the base model
  already reasons correctly, we should only *nudge* — not rewrite.

**Config change:**

```yaml
lora:
  r: 1                       # was 16 (v1), 8 (v2)
  lora_alpha: 2              # was 32 (v1), 16 (v2)
  lora_dropout: 0.0          # was 0.05
  target_modules:            # was: all linear
    - gate_proj
    - up_proj
    - down_proj
```

### 2. Completion-only loss masking

**Source:** TRL's `completion_only_loss=True` in `SFTConfig` (v1.0+);
supported by "safe default" guidance in the TRL docs and by practical
experience in the community.

**Rationale:**
- Before v3, loss was computed over **every token** — system prompt, user
  prompt, and assistant response. Two-thirds of the loss signal was spent
  re-teaching the model how to *read*, not how to *answer*.
- With completion-only masking, loss is computed only on the assistant
  response. The model preserves its base reasoning capability intact.

**Config change:**

```yaml
training:
  completion_only_loss: true
```

Dataset changes from a single `text` column to `prompt` + `completion`
columns; `training/data.py:build_hf_dataset()` was rewritten accordingly.

### 3. Rejection Sampling Fine-Tuning (RFT)

**Source:** Yuan et al., "Self-Rewarding Language Models" (2024) and RFT
usage in Llama-2 and DeepSeek-Math training recipes.

**Rationale:**
- Hand-crafted training targets (`reasoning + \boxed{X}`) force the model
  into a *style* it may not naturally produce, causing drift.
- RFT instead **samples the base model at temperature=0.7**, keeps only the
  outputs whose `\boxed{X}` matches ground truth, and trains on those.
- This is self-distillation: the model learns from outputs it *already
  produced correctly*, so no stylistic forcing.

**Data change:**
- For each math/physics training item, generate N=2 samples.
- Keep only correct ones (~60% for math, ~40% for physics).
- Rejected items are dropped from the training set.

### 4. Curriculum: SFT, then ORPO for refusal

**Source:** Hong et al., "ORPO: Monolithic Preference Optimization without
Reference Model" (2024).

**Rationale:**
- Refusal is a *behavior*, not a *target string*. Forcing a fixed string via
  SFT (as v1/v2 did) causes over-generalization: the model starts refusing
  legitimate in-domain queries.
- ORPO learns refusal from **preference pairs** (chosen = correct math
  answer, rejected = refusal on a legit math question). This teaches the
  boundary without binary forcing.

**Staging:**

| Stage | Data | Method | Purpose |
|---|---|---|---|
| 1 | math + physics (RFT output) | SFT, rank-1 MLP-only | Preserve + sharpen reasoning |
| 2 | chosen/rejected pairs, ~5% refusal signal | ORPO | Learn refusal *boundary* |

### 5. Lower LR, fewer epochs, longer warmup

**Source:** community consensus on LoRA fine-tuning (low LR = 1e-5 for
reasoning preservation).

**Config change:**

```yaml
training:
  learning_rate: 1.0e-5     # was 1e-4 (v1), 2e-5 (v2)
  num_train_epochs: 1       # was 3 (v1), 2 (v2)
  warmup_steps: 200         # was 100
```

### 6. Refusal data ratio lowered to 3-5%

**Source:** "The False Refusal Rate rises from 63% to 84% as safety data
increases from 0% to 40%" (multiple safety-alignment papers, 2024).

**Rationale:** More refusal data does not mean safer behavior. It means
over-refusal. The signal must be **diverse**, not **voluminous**.

**Data change (v3 vs v2):**

| Domain | v2 share | v3 share | Direction |
|---|---|---|---|
| math | 46% | ~47% | same |
| physics | 45% | ~45% | same |
| ood | 5% | ~3% | ↓ fewer |
| adversarial_ood | 2% | ~1% | ↓ fewer |
| false_refusal | 3% | ~4% | ↑ more (anti over-refusal) |

### 7. PiSSA initialization (deferred)

**Source:** Meng et al., "PiSSA: Principal Singular Values and Singular
Vectors Adaptation of Large Language Models" (2024).

**Rationale:** Init LoRA weights from the principal singular vectors of the
base weight matrix. The LoRA update lives in the *null space* of the base
matrix and does not perturb original weights at step 0.

**Status:** **Deferred.** PiSSA requires non-quantized weights at init time
(fp16/fp32). Our QLoRA pipeline quantizes the base model to 4-bit NF4
*before* LoRA init, so PiSSA cannot run. On a 24 GB GPU (L4/A10G) this
becomes feasible by loading base fp16 first, initializing PiSSA, then
quantizing.

Tracked as a v4 candidate in `docs/limitations.md`.

### 8. Additional platform / precision changes (v3 target)

These are hardware-driven and tracked in ADR 0006, not in this ADR:
- **bf16 + tf32** — only on GPUs with hardware bf16 (T4 does not).
- **Flash Attention 2** — partial on T4, full on L4.
- **Sequence length 4096** — cannot fit on T4 16 GB with 4-bit QLoRA.
- **Batch size 4** — cannot fit on T4.

On T4, v3 runs with FP32 + seq 2048 + batch 1 (same as v2). The 8 CPU-side
techniques above are what v3 changes.

## Consequences

### Positive

- Each design change has a distinct mechanism → failures become
  attributable (rather than 'the run got worse, we don't know why').
- v3 is the first attempt in the project to **separate** reasoning
  preservation from behavior change.
- If v3 succeeds, we have a recipe for *future* behavior edits (tone,
  format, safety) without risking reasoning regression.

### Negative / risks

| Risk | Mitigation |
|---|---|
| Rank-1 too weak to learn refusal at all | Keep false_refusal data; eval OOD refusal rate directly |
| Completion-only masking hurts some target formats | Verify eval on all 4 domains |
| RFT filters out hard items (only easy ones remain) | Sample N=2; monitor rejection rate per domain |
| ORPO stage 2 destabilizes stage 1 | Evaluate stage 1 before launching stage 2 |

### Open question

The **interaction** between rank-1 MLP-only + completion-only loss masking
is untested. Individually both are well-documented; jointly they may either
compound (both preserve reasoning) or conflict (too little capacity to
learn refusal). v3 result answers this.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Sweep hyperparameters on v1 recipe | Already proven that recipe is wrong at any LR |
| Larger rank (r=32) | More capacity → more overwrite risk; opposite of goal |
| Full fine-tune (no LoRA) | 7B × 4 bytes = 28 GB, does not fit free-tier GPU |
| DPO instead of ORPO | DPO needs a reference model → 2× VRAM |
| QLoRA + DoRA | DoRA is orthogonal; could be added in v4 |
| Llama-3.1-8B base instead of Qwen | Weaker math prior (see ADR 0002) |

## References

- ADR 0002 — why Qwen2.5-Math-7B
- ADR 0005 — domain-restricted behavior (refusal spec)
- ADR 0006 — FP32 training on T4
- ADR 0004 — eval methodology (how v3 will be measured)
- `training/model.py` — PiSSA fallback logic
- `training/data.py` — completion-only dataset construction
- `docs/limitations.md` — current gaps including PiSSA deferral
