# vLLM adapter

Target: combine the RWKV-7 vLLM model/state scheduler with the official MetaX
hardware plugin. The initial route is eager and correctness-first. Graph capture
and C500 kernels follow only after public-engine dynamic batch and chunked
prefill pass.

State-slot reuse is required, but it is not reported as prefix-cache reuse.
