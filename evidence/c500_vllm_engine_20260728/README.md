# C500 vLLM public Engine gate

Date: 2026-07-28

This artifact records a real public `vllm.LLM` run on one MetaX C500 using the
locked RWKV vLLM source at
`7d16dbf88077834fbf93a0a65a882286319d7bd6` and MetaX plugin source at
`e9f17cef56f43ce9ee6b3426819842307b739c28`. The model is the official RWKV-7
2.9B checkpoint whose SHA256 is
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`.

Two tokenized requests with prompt lengths 4 and 5 were submitted together.
Each produced exactly four deterministic top-k-1 tokens. The run used FP16,
the V2 RWKV runner, eager execution, the compiled rapid sampler, and the C500
reduced-profile default that routes channel mix through the dense path instead
of the unvalidated CUDA-specific no-FC sparse kernels.

The public Engine initialized in `37.66` seconds, generated eight tokens in
`1.471` seconds, and reported `5952.82 MiB` peak allocated device memory. These
single short-run values are diagnostic telemetry, not production throughput or
latency claims.

This result establishes public Engine construction, real 2.9B loading,
mixed-length batch execution, and four decode steps per request. It does not
establish actual chunk-boundary prefill, continuously arriving dynamic
requests, recurrent state-cache reuse or hit rate, preemption restore,
RWKV-LM/Albatross parity, TP/PP, W8/W4, speculative decoding, or the complete
model/shape matrix.

Reproduce from the repository root after installing the vendor runtime and
both locked patch sets:

```bash
python scripts/apply_track_patches.py vllm /path/to/vllm-rwkv
python scripts/apply_track_patches.py vllm /path/to/vllm-metax \
  --manifest metax_patchset.json

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_RAPID_SAMPLER=1
python adapters/vllm/accept_engine_c500.py \
  --model /path/to/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --output result.json
```

`result.json` contains no host name, account name, network address, credential,
or private absolute path.
