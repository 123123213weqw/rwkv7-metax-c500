# C500 vLLM continuously arriving dynamic-batch gate

Date: 2026-07-28

This artifact records a public `AsyncLLM` run on one MetaX C500 using the
locked RWKV vLLM source at
`7d16dbf88077834fbf93a0a65a882286319d7bd6` and MetaX plugin source at
`e9f17cef56f43ce9ee6b3426819842307b739c28`. The official RWKV-7 2.9B
checkpoint SHA256 is
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`.

The accepted run used FP16, top-k-1 deterministic sampling, 24 output tokens
per request, eager execution, asynchronous scheduling, a one-token stream
interval, a 128-token scheduler budget, and the C500-safe dense channel-mix
route. It first generated requests A and B separately as greedy references.
It then submitted A, waited for A's first streamed token, and only then
submitted B.

B was submitted at `0.035703 s`, produced its first token at `0.134086 s`, and
A completed at `0.379592 s`, all relative to the dynamic run start. Thus B was
admitted and scheduled while A was still active. Both 24-token dynamic outputs
matched their separate references exactly, and each request emitted 24 stream
events. The overlapping dynamic run completed in `0.468405 s`.

The MetaX reduced-profile patch defaults
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, while preserving explicit user override.
This is required for the public asynchronous library path on C500: the default
`fork` attempt failed during child device initialization before model
execution. The accepted run started without a manual multiprocessing-method
override and recorded the plugin-selected `spawn` value in `result.json`.

This closes one continuously arriving, overlapping, greedy-equivalent 2.9B
FP16 two-request cell. It does not establish arrival-rate capacity, latency
percentiles, the full model/batch/prompt matrix, prefix-cache hit rate,
preemption restore, RWKV-LM/Albatross parity, TP/PP, W8/W4, or speculative
decoding. Device memory is intentionally not reported by this gate because the
model runs in a separate EngineCore process; a later process-aware performance
matrix must measure it.

Reproduce after applying and installing both locked vLLM patch sets:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RAPID_SAMPLER=1
python adapters/vllm/accept_dynamic_batch_c500.py \
  --model /path/to/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --max-new-tokens 24 \
  --output result.json
```

`result.json` contains no host name, account name, network address, credential,
or private absolute path.
