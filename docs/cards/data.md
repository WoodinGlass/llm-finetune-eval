# Data Card — math + physics fine-tuning dataset

**Version:** 1.0 · **Last updated:** M1.8 · **Owner:** WoodinGlass
**License of curated artifact:** MIT (derived works follow source licenses below)

---

## 1. Overview

This dataset is used to fine-tune `Qwen2.5-Math-7B-Instruct` for a
**math-and-physics-only** assistant that:

1. Answers math and physics questions with a `\boxed{...}` final answer.
2. Refuses out-of-domain (OOD) requests with an exact, fixed string
   (see ADR 0005).
3. Never refuses legitimate math/physics questions (false-refusal rate <= 2%).

---

## 2. Composition

**Total training records:** 3935

| Domain | Rows | Share | Source |
|---|---:|---:|---|
| math | 1800 | 45.7% | GSM8K (1200) + Hendrycks MATH (600) |
| physics | 1735 | 44.1% | cais/mmlu (612) + MMLU-Pro physics (1188) minus 64 decontaminated |
| ood | 320 | 8.1% | manual (190) + template (130) |
| adversarial_ood | 80 | 2.0% | manual + composition |
| **Total** | **3935** | 100% | |

**Test set (frozen):** 302 records — SHA256 `f10f4893220d2d283cf6b5c4f35807f5...`

| Split | Rows | Purpose |
|---|---:|---|
| math test | 100 | exact-match evaluation |
| physics test | 102 | A/B/C/D letter match |
| ood test | 50 | refusal string exact match |
| false_refusal test | 50 | answered-not-refused check |

---

## 3. Provenance (source datasets)

| Source | HF ID | License | Split used |
|---|---|---|---|
| GSM8K | `openai/gsm8k` | MIT | train + test |
| Hendrycks MATH | `EleutherAI/hendrycks_math` | MIT | train |
| MMLU | `cais/mmlu` | MIT | test+validation+dev per config |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | MIT | test, validation |
| SciQ | `allenai/sciq` | CC-BY-NC-3.0 | **dropped** (bio contamination, see section 7) |

All sources are **public** and used under their respective licenses. The
curated artifact (deduped, scrubbed, decontaminated) is a derivative work
distributed under MIT with attribution to the sources above.

**Leak avoidance:**
- `cais/mmlu` **college_physics** is **never** used for training — it is
  reserved as frozen test set (M0.5).
- Test prompts are checked against training prompts via 13-gram overlap (section 5).

---

## 4. Cleaning pipeline

```
data/raw/train/*.jsonl          (4,000 rows)
        |
        v  M1.5  dedup + PII scrub
data/processed/train/*.jsonl    (3999 rows, removed 1)
        |
        v  M1.6  13-gram decontamination
data/processed/train/*.jsonl    (3935 rows, removed 64)
        |
        v  M1.7  DVC tracking
data/processed/train.dvc        (pointer file, content in DVC remote)
```

### 4.1 Deduplication (MinHash + LSH)

- **Method:** MinHash LSH, threshold 0.85 Jaccard, 5-word shingles, num_perm=128.
- **Effect:** 4000 -> 3999 rows (removed 1, 0.03%).
- **Idempotent:** running twice on cleaned input yields the same output.

### 4.2 PII scrubbing (domain-aware)

- **math / physics:** conservative regex only — email, phone (Indonesia + intl),
  URL with embedded token. **Numbers are preserved** because they are legitimate
  math content (e.g. 1234567890123456 is not a NIK here, it's a number).
- **ood / adversarial_ood:** aggressive regex (NIK, NPWP, passport, CC-like, IP v4)
  **plus** Presidio NER (PERSON, LOCATION, GPE, NRP, DATE_TIME, score >= 0.6).
- **False-positive guarantee:** tested — 27 unit tests including specific cases
  where numeric content must survive.

### 4.3 Decontamination (13-gram overlap)

- **Method:** whitespace-word 13-gram overlap against all 302 test prompts.
- **Effect:** 3999 -> 3935 (removed 64).
- **Where removed:** physics only (64 rows) — because MMLU and
  MMLU-Pro occasionally share verbatim question stems.

---

## 5. Data schema

Validated by `src/llm_ft/schema.py` (Pandera). Any row violating the schema
is a hard failure (see `docs/slo.md` section 4).

| Field | Type | Constraints |
|---|---|---|
| `id` | str | unique, regex `^[a-z0-9][a-z0-9_\-]{2,63}$` |
| `domain` | str | one of `math`, `physics`, `ood`, `adversarial_ood` |
| `prompt` | str | 4–8000 chars |
| `answer` | str | non-empty |
| `reasoning` | str or null | optional chain-of-thought |
| `meta` | dict | source, split, idx, etc. |

**Domain-specific rules:**
- `physics` -> `answer` in {A, B, C, D}
- `ood` / `adversarial_ood` -> `answer` == REFUSAL_STRING (exact match)
- `math` -> `answer` must contain digit or symbolic char

---

## 6. Known biases and limitations

1. **GSM8K is grade-school level.** The math training set skews toward
   arithmetic word problems, not competition mathematics. Hendrycks MATH
   mitigates this but is only 600/1800 of the math partition.
2. **MMLU-Pro physics is US-undergrad level.** It is easier than olympiad
   physics. Expect the model to plateau at ~85–90% on our physics test set.
3. **OOD prompts are English-only.** Refusal behavior will likely not
   generalize to Indonesian prompts at inference time.
4. **Adversarial OOD is curated, not scraped.** Coverage of real-world
   jailbreaks is limited — see ADR 0005 section "Non-goal".
5. **Answer distribution:** physics A/B/C/D approx balanced (423/464/446/467),
   but minor skew exists. Post-fine-tune eval should watch for letter bias.
6. **Refusal rate floor:** the base model never refuses today (0%), so
   achieving >=95% OOD refusal requires the fine-tune to *learn* a new
   behavior — check false-refusal rate to avoid over-correction.

---

## 7. What was deliberately excluded

| Excluded | Reason |
|---|---|
| `allenai/sciq` | Keyword-based physics filter leaked biology ("thyroid", "hormone") — dropped in favor of MMLU-Pro |
| `camel-ai/physics` | 20k synthetic GPT-4 pairs, zip archive parsing impractical on free tier |
| `cais/mmlu college_physics` | Reserved as frozen test set |
| Any dataset requiring paid license | Free-tier constraint |
| Non-English math/physics | Out of scope for v1 |

---

## 8. Reproducing this dataset

Run the following from a fresh clone:

- `make setup`         — uv sync --extra dev --extra data
- `make data-pull`     — dvc pull from configured remote
- `make test-data`     — schema + unit tests

Raw files are DVC-tracked: `data/raw/train.dvc`, `data/processed/train.dvc`,
`data/processed/test.dvc`. The remote is a local Drive path in dev;
see ADR 0003 for production configuration.

---

## 9. Revision history

| Date | Change |
|---|---|
| M1.8 | Initial card. 4,000 raw -> 3935 final (64 decontam, 1 dedup) |
| (M0.5) | Frozen test set (302 rows) locked via SHA256 |
