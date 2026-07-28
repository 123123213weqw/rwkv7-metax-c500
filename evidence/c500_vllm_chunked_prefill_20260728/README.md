# C500 vLLM chunked-prefill equivalence gate

Date: 2026-07-28

This artifact records a public `vllm.LLM` A/B run on one MetaX C500 using the
locked RWKV vLLM source at
`7d16dbf88077834fbf93a0a65a882286319d7bd6` and MetaX plugin source at
`e9f17cef56f43ce9ee6b3426819842307b739c28`. The official RWKV-7 2.9B
checkpoint SHA256 is
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`.

Both runs used FP16, bsz=2, prompt lengths 129 and 193, top-k-1 deterministic
sampling, four output tokens per request, eager execution, and the C500-safe
dense channel-mix route. The baseline allowed 512 scheduled tokens, enough to
fit both prompts in one prefill step. The candidate allowed only 64 scheduled
tokens. Its 322 prompt tokens therefore required at least six scheduler steps
and both requests crossed a chunk boundary.

The forced-chunk candidate reproduced all eight baseline output tokens exactly.
Baseline and chunked generation took `1.473` and `1.693` seconds respectively;
peak allocated memory was `5985.28` and `5958.39 MiB`. These one-run values are
diagnostic telemetry and not throughput or memory-performance claims.

This closes chunk-boundary greedy equivalence for this exact 2.9B, FP16, bsz=2
cell. It does not establish continuously arriving dynamic requests, recurrent
state-cache reuse or hit rate, preemption restore, the required model/shape
matrix, RWKV-LM/Albatross parity, TP/PP, W8/W4, or speculative decoding.

Reproduce after applying and installing both locked vLLM patch sets:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_RAPID_SAMPLER=1
python adapters/vllm/accept_chunked_prefill_c500.py \
  --model /path/to/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --baseline-budget 512 \
  --chunk-size 64 \
  --output result.json
```

`result.json` contains no host name, account name, network address, credential,
or private absolute path.
