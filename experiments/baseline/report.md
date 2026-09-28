# Baseline Report — Qwen2.5-Math-7B-Instruct (M0.5)

- Model: `Qwen/Qwen2.5-Math-7B-Instruct`
- Quantization: 4-bit nf4 double-quant, float16 compute
- Decoding: greedy
- Batch size: 4
- Items: 102
- Wall: 1904.3s
- Test lock SHA256: `f10f4893220d2d283cf6b5c4f35807f51158eca06f5a2ac39dac6f33a7d37f5a`
- Git SHA: `a0db5533ec9b1845266eb43d91e600d6eaae20f3`

## Results (95% bootstrap CI, 1000 resamples)

| Kind | N | Correct | Accuracy | 95% CI |
|---|---:|---:|---:|---|
| false_refusal | 50 | 22 | 0.440 | [0.300, 0.580] |
| math | 100 | 96 | 0.960 | [0.920, 0.990] |
| ood | 50 | 0 | 0.000 | [0.000, 0.000] |
| physics | 102 | 74 | 0.725 | [0.627, 0.814] |
| **combined** | 252 | — | 0.762 | [0.710, 0.813] |
