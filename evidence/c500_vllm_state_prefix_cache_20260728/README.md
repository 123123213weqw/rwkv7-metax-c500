# C500 vLLM recurrent prefix-state cache gate

Date: 2026-07-28

This artifact records a public `vllm.LLM` cache-disabled/cache-enabled A/B on
one MetaX C500. It uses the locked RWKV vLLM source at
`7d16dbf88077834fbf93a0a65a882286319d7bd6`, the MetaX plugin source at
`e9f17cef56f43ce9ee6b3426819842307b739c28`, FP16, eager execution, chunked
prefill, top-k-1 deterministic generation and the official RWKV-7 2.9B
checkpoint with SHA256
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`.

The workload warms a 128-token common prefix, then submits four requests that
reuse it and one unrelated control request. The recurrent cache stores complete
64-token block boundaries. The accepted row reports cached-token counts
`[128, 128, 128, 128, 0]`, an `0.8` request hit rate, an approximately `0.6818`
token hit rate, an explicit control miss and exact cache-on/cache-off greedy
output equality for all five requests.

Median first-token latency for the four shared-prefix requests changed from
`0.103727 s` without the cache to `0.072677 s` with the cache, a ratio of
`0.700663`. The complete five-request mixed batch changed from `0.243564 s` to
`0.256037 s`. The latter includes cold snapshot writes and the unrelated
control request, so this artifact claims a measured hit-latency improvement,
not aggregate throughput acceleration. Peak allocated memory was `6004.8 MiB`
without the cache and `5995.9 MiB` with it; that small difference is telemetry,
not a memory-reduction claim.

The implementation is a bounded process-local LRU of RWKV shift/WKV/elapsed
state snapshots. It is bypassed for LoRA requests, prompt embeddings, cache
salts and KV-transfer configurations. This exact cell does not establish
cross-process sharing, preemption restore, eviction under sustained load, the
full model/shape matrix, TP/PP, RWKV-LM/Albatross parity, W8/W4 or speculative
decoding.

Reproduce after applying and installing both locked vLLM patch sets:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_RAPID_SAMPLER=1
export VLLM_RWKV7_DISABLE_CMIX_NOFC=1
python adapters/vllm/accept_state_prefix_cache_c500.py \
  --model /path/to/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --cache-capacity 16 \
  --cache-block-size 64 \
  --max-new-tokens 8 \
  --output result.json
```

`result.json`, `run.log` and `environment.txt` are redacted of host, account,
network and credential data. `exit_code.txt` records the process exit status.
