# SGLang adapter

Target: port the RWKV-7 linear-attention backend onto the MetaX SGLang fork,
starting with pure PyTorch/MXMACA operations and public `sglang.Engine` tests.

Radix caching stays disabled until a state-aware cache passes shared-prefix
greedy equivalence and records a real hit rate.

Apply the locked port to the exact MetaX SGLang revision:

```bash
python scripts/apply_track_patches.py sglang /path/to/metax-sglang
```

The first public gate uses `sglang.Engine` with two tokenized requests of
different prompt lengths. It keeps CUDA graph, overlap scheduling and radix
cache disabled so that the result establishes basic Engine and dynamic-batch
execution without implying state-cache reuse or production throughput. The
committed result is in `evidence/c500_sglang_engine_20260727`.
