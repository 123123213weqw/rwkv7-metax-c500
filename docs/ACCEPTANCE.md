# MetaX C500 acceptance

The machine-readable source of truth is
[`acceptance_matrix.json`](../acceptance_matrix.json). A track is complete only
when every required model and matrix cell has committed evidence. The required
model inventory is 0.1B, 0.4B, 1.5B, 2.9B, 7.2B and 13.3B. A pass on one model,
one batch size or one smoke test never represents another cell.

Every claimed backend must provide committed, machine-readable artifacts for
the exact C500 environment. A passing import is not a backend acceptance.

## Common gates

- Environment probe passes fp16 and bf16 matmul with cosine at least `0.999`.
- Official RWKV-7 numpy/torch oracle matches forward and recurrent state.
- Checkpoint save/reload and deterministic greedy generation pass.
- Errors fail closed without silently selecting an NVIDIA-only kernel.
- Peak memory and elapsed time come from synchronized device measurements.
- Every result is one of `pass`, `fail`, `not_run` or `not_validated`; missing
  hardware, a missing baseline or missing evidence cannot be counted as pass.
- Compare the same checkpoint, card, dtype, batch size, prompt length, decode
  length and cache policy against current RWKV-LM and Albatross wherever those
  baselines run. The target is at least `1.0x` their throughput and no more than
  `1.0x` their peak memory, while preserving logits, tokens and recurrent state.

## Hugging Face

- `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM` and cached `generate`.
- Batch `1/2/4/8`, chunked prefill and dynamic cache select/reorder/drop.
- CPU fallback, C500 inference, Trainer and PEFT LoRA smoke.
- Trainer, PEFT LoRA save/load/merge, TRL SFT/DPO/GRPO, checkpoint resume and
  real DeepSpeed ZeRO-2/ZeRO-3 runs across the model inventory.
- W8/W4 footprint plus same-shape speed, logits and greedy comparisons.

## vLLM

- Public `vllm.LLM` engine, not a direct model call.
- Mixed prefill/decode scheduling, chunked prefill and per-request state slots.
- Shared-prefix state cache correctness and measured hit rate when claimed.
- TP/PP only after real multi-card rows.
- A controlled 80% repeated-prefix workload must reach at least 80% eligible
  state-cache hit rate after warmup, exactly match cache-disabled output and
  show no state leakage between requests.

## SGLang

- Public `sglang.Engine`, not a direct model call.
- Packed variable-length prefill and dynamic decode with independent states.
- State-aware radix/prefix caching must match radix-off greedy output.
- Quantization and speculative decoding need end-to-end gates.
- Radix lookup is not sufficient by itself: cached RWKV recurrent state must be
  keyed, restored and invalidated with the token prefix.

## Performance matrix

Run batch `1/2/4/8`, prompt `128/512/2048` and decode `1/128` for every model.
Report prefill tok/s, decode tok/s, end-to-end tok/s, p50/p95 latency, model
footprint, peak memory, state/cache telemetry and correctness. Resource-driven
fallbacks such as offload or gradient accumulation may be recorded, but they do
not remove the matrix cell.

## Quantization and speculation

- W8 and W4 must reduce model footprint and be at least as fast as W16 on every
  claimed card, model and inference shape. Logits and greedy output must pass;
  the quality target is llama.cpp `Q*_K_M` class rather than a memory-only path.
- Initial speculative decoding is required for every target model, using a
  smaller compatible RWKV draft or a reproducibly trained tiny draft. It must
  reproduce target greedy output and report acceptance rate and end-to-end
  speedup. DFlash remains a follow-up project.

## Hardware scope

Exact C500 evidence is mandatory in this repository. Cross-platform acceptance
also tracks NVIDIA Pascal, Volta, Turing, Ampere, Ada, Hopper and Blackwell plus
AMD. A conservative routing rule without exact-card evidence is recorded as
`not_validated`, not as support. Inference TP/PP and training ZeRO-2/ZeRO-3 need
real multi-device runs before they pass.
