# Threat Model

> **Status:** stub - full mapping in **M10**. OWASP LLM Top 10 (2025) mapping is
> already summarized in README.md (Security & Threat Model section).

## Scope (planned)

| OWASP ID | Threat | Mitigation (planned / partial) |
|---|---|---|
| LLM01 | Prompt injection | Llama Guard, input sanitization (M5) |
| LLM02 | Insecure output | PII redaction, output validation (M5) |
| LLM03 | Training data poisoning | Data provenance, dedup, decontamination (M1 done) |
| LLM04 | Model DoS | Rate limit, max tokens, timeout (M5) |
| LLM05 | Supply chain | pip-audit, Trivy, pinned deps (M6) |
| LLM06 | Sensitive info disclosure | Presidio scrub (deferred - regex only so far) |
| LLM07 | Insecure plugin | n/a (no tool use in v1) |
| LLM08 | Excessive agency | n/a (no agent actions in v1) |
| LLM09 | Overreliance | Disclaimer, eval gate (M3/M6) |
| LLM10 | Model theft | API key rotation, no raw weight download (M5/M7) |

Full write-up with attack scenarios, test cases, and residual-risk assessment
lands in M10.
