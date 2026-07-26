# SGLang adapter

Target: port the RWKV-7 linear-attention backend onto the MetaX SGLang fork,
starting with pure PyTorch/MXMACA operations and public `sglang.Engine` tests.

Radix caching stays disabled until a state-aware cache passes shared-prefix
greedy equivalence and records a real hit rate.
