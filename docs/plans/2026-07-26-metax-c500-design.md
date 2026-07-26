# MetaX C500 adaptation design

## Architecture

The project is one evidence repository with three independent adapter surfaces.
The shared layer owns only exact upstream pins, environment detection, official
RWKV-7 mathematical oracles, checkpoint metadata and acceptance schemas. The HF
adapter starts from the native pure-PyTorch path and introduces C500 kernels only
after correctness. The vLLM adapter integrates the RWKV model and recurrent
state manager with the official `vLLM-metax` hardware plugin. The SGLang adapter
ports the RWKV linear-attention backend onto the official MetaX SGLang fork.

Complete upstream forks are kept in ignored worktrees. Deliverable changes are
small patch sets or installable plugins, which makes version drift explicit and
keeps ownership reviewable.

## Validation sequence

1. Inventory exact card, driver, SDK, PyTorch and framework versions.
2. Validate fp16/bf16 PyTorch operations and official RWKV recurrence on C500.
3. Close HF load, forward, cache, generate, training and quantized inference.
4. Close vLLM public-engine dynamic batch, chunked prefill and state lifecycle.
5. Close SGLang public-engine dynamic batch, chunked prefill and state lifecycle.
6. Add exact-card speed, memory, quality, cache-hit and soak artifacts.

Optimized kernels are disabled until an exact-card A/B passes. C500 has a
64-thread warp, so NVIDIA warp-32 assumptions in CUDA/Triton code require an
explicit audit. Slot reuse, state-prefix reuse and cache hit rate are recorded
as separate metrics.
