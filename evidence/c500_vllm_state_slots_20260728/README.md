# C500 vLLM per-request state-slot isolation gate

Date: 2026-07-28

This artifact records a public `vllm.LLM` state-isolation run on one MetaX
C500 using the locked RWKV vLLM source at
`7d16dbf88077834fbf93a0a65a882286319d7bd6` and MetaX plugin source at
`e9f17cef56f43ce9ee6b3426819842307b739c28`. The official RWKV-7 2.9B
checkpoint SHA256 is
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`.

Both workers used FP16, top-k-1 deterministic sampling, eight output tokens,
eager execution, a 128-token scheduler budget, and the C500-safe dense
channel-mix route. The solo worker generated request A and then request B. A
second worker generated A/B in one batch and then B/A in the same Engine.
Prompt lengths were 33 and 47 tokens.

The keyed output for each request matched its solo reference in both batch
orders. Request A produced `[82, 116, 84, 89, 86, 88, 90, 91]`; request B
produced `[120, 119, 123, 125, 126, 127, 128, 127]`. The repeated reverse-order
batch is also a post-completion slot-reuse smoke. Solo and batched generation
took `1.783` and `1.665` seconds respectively; peak allocated memory was
`5956.63` and `5960.05 MiB`. These one-run values are diagnostic telemetry,
not throughput or memory-performance claims.

This closes per-request recurrent-state isolation and completed-slot reuse for
this exact 2.9B, FP16, two-request cell. It does not establish prefix-cache
reuse or hit rate, continuously arriving requests, scheduler preemption and
restore, the required model/shape matrix, RWKV-LM/Albatross parity, TP/PP,
W8/W4, or speculative decoding.

Reproduce after applying and installing both locked vLLM patch sets:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_RAPID_SAMPLER=1
python adapters/vllm/accept_state_slots_c500.py \
  --model /path/to/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --max-new-tokens 8 \
  --output result.json
```

`result.json` contains no host name, account name, network address, credential,
or private absolute path.
