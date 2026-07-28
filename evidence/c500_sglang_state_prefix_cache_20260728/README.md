# C500 SGLang recurrent prefix-state cache gate

Date: 2026-07-28

This artifact records public `sglang.Engine` state-cache validation on one
MetaX C500. It uses the locked MetaX SGLang commit
`e55a35fbcd491eeb3743be9b021534512c6acf85`, FP16, top-k-1 deterministic
generation and the converted RWKV-7 0.4B checkpoint. Model hashes and the
validated source-diff hash are in `environment.txt`.

The defect was in scheduler cache selection. RWKV already allocated the hybrid
request/state pool, but `Scheduler.init_cache_with_memory_pool()` omitted
`rwkv7_config` from `is_hybrid_ssm`. The scheduler therefore selected ordinary
`RadixCache`: token prefixes reported hits while recurrent state remained zero.
The patch enables RWKV's Mamba cache contract in `ServerArgs` and selects
`MambaRadixCache` for the RWKV model runner.

The strict radix-off/radix-on A/B warms a 128-token prefix and submits four
shared-prefix requests plus one unrelated control. It reports cached tokens
`[128, 128, 128, 128, 0]`, request hit rate `0.8`, token hit rate `0.6818`, an
explicit control miss and exact equality for all generated tokens. The
shared-request median latency ratio is `1.0037`, so this row is approximately
neutral rather than a speed win. Used GPU memory is `3006.0 MiB` with radix
disabled and `23575.8 MiB` with the automatically sized state cache. That is a
real capacity cost, not a memory improvement.

Two ordinary-language checks exercise `chunked_prefill_size=64`. The five
request cold/warm matrix preserves all eight greedy tokens per request and the
warm batch takes `0.6326x` the cold batch time. The cold radix tree can still
share prefixes among requests in that same batch, so this timing is not the
strict cache-disabled comparison. A separate single-request check records a
cold miss followed by a 128-token hit; both generated tokens, token logprobs
and both top-10 distributions match exactly.

Reproduce the primary committed row after applying the locked SGLang patch
set. Timings vary; the acceptance conditions are exact output equality, four
128-token hits, the control miss and at least `0.8` request hit rate.

```bash
python adapters/sglang/accept_state_prefix_cache_c500.py \
  --model /path/to/rwkv7-g1d-0.4b-sglang \
  --chunked-prefill-size 8192 \
  --max-new-tokens 8 \
  --output result.json
```

Use `--chunked-prefill-size 64` for the forced chunk-boundary variant. The
committed `chunked_text_result.json` and `logprob_result.json` use ordinary
tokenized language; they supplement the deterministic synthetic-token primary
gate.

This exact cell establishes process-local state-aware prefix-cache correctness
and measured hit rate for 0.4B. It does not establish cache memory efficiency,
eviction stress, preemption restore, cross-process sharing, the full
model/shape matrix, TP/PP, RWKV-LM/Albatross parity, W8/W4 or speculative
decoding. Files are redacted of host, account, network, credential and private
filesystem information.
