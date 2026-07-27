# AGENTS.md

## Scope

This repository is exclusively for RWKV-7 on MetaX C500. Keep Hugging Face,
vLLM and SGLang as independent adapters sharing only environment probes,
checkpoint metadata, mathematical oracles and evidence schemas.

## Rules

- Never commit credentials, SSH endpoints, hostnames, private model paths or
  unredacted environment dumps.
- Pin every upstream revision in `upstreams.lock.json`.
- Do not call a track supported from import-only or CPU-only tests.
- Exact C500 evidence must include `mx-smi`, MXMACA, PyTorch, model, dtype,
  batch/prompt/decode shape, correctness, speed and peak memory.
- Preserve the official RWKV-7 recurrence and fp32 recurrent-state semantics.
- Keep unsupported kernels fail-closed and provide a pure-PyTorch fallback.
- Do not describe state-slot reuse as prefix-cache reuse. Cache-hit claims need
  a state-aware shared-prefix test and measured hit-rate telemetry.
- W8/W4 speed claims require lower footprint, end-to-end speed no slower than
  W16 on the exact shape, and logits/greedy quality gates.
- Treat `acceptance_matrix.json` as the authoritative completion contract. All
  six model sizes and all required shapes must have evidence; never aggregate a
  partial model or batch result into a completed track.
- Performance acceptance requires same-card, same-checkpoint, same-dtype and
  same-shape RWKV-LM/Albatross comparisons. Missing or unavailable baselines are
  `not_validated`, never inferred passes.
- All commits and pull requests require a DCO sign-off.
