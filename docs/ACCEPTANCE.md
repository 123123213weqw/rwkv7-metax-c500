# MetaX C500 acceptance

Every claimed backend must provide committed, machine-readable artifacts for
the exact C500 environment. A passing import is not a backend acceptance.

## Common gates

- Environment probe passes fp16 and bf16 matmul with cosine at least `0.999`.
- Official RWKV-7 numpy/torch oracle matches forward and recurrent state.
- Checkpoint save/reload and deterministic greedy generation pass.
- Errors fail closed without silently selecting an NVIDIA-only kernel.
- Peak memory and elapsed time come from synchronized device measurements.

## Hugging Face

- `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM` and cached `generate`.
- Batch `1/2/4/8`, chunked prefill and dynamic cache select/reorder/drop.
- CPU fallback, C500 inference, Trainer and PEFT LoRA smoke.
- W8/W4 footprint plus same-shape speed, logits and greedy comparisons.

## vLLM

- Public `vllm.LLM` engine, not a direct model call.
- Mixed prefill/decode scheduling, chunked prefill and per-request state slots.
- Shared-prefix state cache correctness and measured hit rate when claimed.
- TP/PP only after real multi-card rows.

## SGLang

- Public `sglang.Engine`, not a direct model call.
- Packed variable-length prefill and dynamic decode with independent states.
- State-aware radix/prefix caching must match radix-off greedy output.
- Quantization and speculative decoding need end-to-end gates.

## Performance matrix

Use at least batch `1/2/4/8`, prompt `128/512/2048` and decode `1/128` for
models that fit. Report prefill tok/s, decode tok/s, complete latency, peak
memory, state/cache telemetry and correctness. Compare with the same checkpoint
and shape; do not compare unrelated model parameter counts as raw speed claims.
