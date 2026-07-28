# SGLang adapter

Target: port the RWKV-7 linear-attention backend onto the MetaX SGLang fork,
starting with pure PyTorch/MXMACA operations and public `sglang.Engine` tests.

The second patch routes RWKV through SGLang's `MambaRadixCache`. It enables
state-aware radix caching in `ServerArgs` and classifies RWKV as a hybrid SSM
when the scheduler selects its cache. This is required because token-prefix
hits alone are invalid unless the matching RWKV recurrent state is restored.

Apply the locked port to the exact MetaX SGLang revision:

```bash
python scripts/apply_track_patches.py sglang /path/to/metax-sglang
```

The first public gate uses `sglang.Engine` with two tokenized requests of
different prompt lengths. It keeps CUDA graph, overlap scheduling and radix
cache disabled so that the result establishes basic Engine and dynamic-batch
execution without implying state-cache reuse or production throughput. The
committed result is in `evidence/c500_sglang_engine_20260727`.

The public recurrent prefix-state cache gate is in
`evidence/c500_sglang_state_prefix_cache_20260728`. On the 0.4B checkpoint,
four requests restore the same 128-token prefix state, an unrelated control
request remains a cache miss, and all eight greedy output tokens exactly match
the radix-disabled run. A separate ordinary-language probe with
`chunked_prefill_size=64` also reproduces all outputs and exact two-step
top-logprob distributions. The primary A/B is a correctness and hit-rate gate:
its shared-request latency is approximately neutral (`1.0037x`), and the
automatically sized state pool increases used GPU memory substantially. It is
not a throughput or memory-efficiency claim.

Reproduce the strict gate after applying the locked patch set:

```bash
python adapters/sglang/accept_state_prefix_cache_c500.py \
  --model /path/to/rwkv7-g1d-0.4b-sglang \
  --chunked-prefill-size 8192 \
  --output result.json
```
