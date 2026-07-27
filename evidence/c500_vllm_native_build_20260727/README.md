# C500 RWKV vLLM native build evidence

This artifact records the locked RWKV vLLM source at
`7d16dbf88077834fbf93a0a65a882286319d7bd6` after applying the repository's
three C500 patches. The reduced `rwkv` profile built and installed all three
native modules on one MetaX C500 64GB:

- `rwkv7_ops.cpython-310-x86_64-linux-gnu.so`
- `_rapid_sampling.cpython-310-x86_64-linux-gnu.so`
- `cumem_allocator.abi3.so`

The installed-operator probe uses two packed requests, five tokens, and four
recurrent-state slots. Both WKV precision modes pass against independent Torch
references. FP32-state/FP16-IO output and state cosine are `1.0`; FP16-state
output cosine is `0.9999994040` and state cosine is `0.9999999404`. The mapped
CPU-state view and rapid-sampling state allocation also execute successfully.

This closes native build, import, registration, and operator-correctness gates.
It does not by itself close public Engine, dynamic batching, chunked prefill,
state-cache reuse, model throughput, or peak-memory gates. Those are recorded
as separate real-checkpoint evidence.

Reproduce after applying the vLLM patchset and installing the reduced profile:

```bash
python adapters/vllm/accept_installed_ops_c500.py \
  --output evidence/local/vllm-installed-ops.json
```

The artifact excludes hostnames, network endpoints, account names, credentials,
and credential-bearing environment variables.
