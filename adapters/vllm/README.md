# vLLM adapter

Target: combine the RWKV-7 vLLM model/state scheduler with the official MetaX
hardware plugin. The initial route is eager and correctness-first. Graph capture
and C500 kernels follow only after public-engine dynamic batch and chunked
prefill pass.

State-slot reuse is required, but it is not reported as prefix-cache reuse.

Apply the locked C500 patch set to the exact RWKV vLLM upstream before building:

```bash
python scripts/apply_track_patches.py vllm /path/to/vllm-rwkv
python scripts/apply_track_patches.py vllm /path/to/vLLM-metax \
  --manifest metax_patchset.json
VLLM_BUILD_PROFILE=rwkv python -m pip install --no-deps --no-build-isolation -e /path/to/vllm-rwkv
```

The first patch selects the vendor CUDA-compatible compiler and separates the
real `cucc` version probe from the mock compiler used by `cmake_maca`. The
second exposes the native C++/CUDA language standards as CMake cache settings;
the C500 CUDA 11.6-compatible toolchain uses
`-DVLLM_CXX_STANDARD=17 -DVLLM_CUDA_STANDARD=17`. The third enables the MACA
compile contract, uses the traditional Torch registration API available in the
vendor PyTorch 2.8 build, and provides a C500-safe FP16 WKV copy path when
NVIDIA `cp.async` assembly is unavailable. The fourth restores the fake-op
registration import omitted by the reduced RWKV branch and guards an optional
FP4 overload that the MetaX operator library does not publish. The fifth lets
the compiled rapid sampler run on an out-of-tree CUDA-compatible platform, and
the sixth keeps JIT monitoring operational with vendor Triton builds that do
not expose the newer `triton.knobs` API. The ninth adds an opt-in RWKV-aware
prefix cache behind the public `enable_prefix_caching=True` setting. It hashes
complete token blocks in the scheduler and stores bounded GPU snapshots of the
corresponding recurrent shift/WKV state in the worker. Generic KV block caching
remains disabled for RWKV; unsupported LoRA, prompt-embedding, salted-cache and
KV-transfer combinations bypass or fail closed instead of reusing an unsafe
state. The seventh adapts a Triton pointer
helper to the vendor compiler's constexpr representation. The eighth adds a
registered RWKV channel-mix safety switch so a hardware plugin can select the
dense path when the CUDA-specific no-FC sparse kernels are not validated.

The separate MetaX plugin patch lets an explicit V2 RWKV runner opt in, loads
the reduced RWKV extension at the normal platform-kernel lifecycle point,
fills the missing vendor `torch.accelerator` memory methods from `torch.cuda`,
and skips unrelated MLA, model and quant registrations removed by the reduced
build. On the reduced C500 profile it defaults the no-FC channel-mix switch to
the dense path while preserving explicit user override and all non-MetaX
defaults. A second MetaX patch defaults the reduced profile to the `spawn`
worker start method required by the C500 asynchronous EngineCore path while
preserving explicit override. None of the patches modifies the host SDK
installation.

The first native WKV compile and numerical gate is recorded in
`evidence/c500_vllm_native_build_20260727`. It is kernel evidence, not a
public-engine performance claim.

The public `vllm.LLM` 2.9B mixed-length batch gate is recorded in
`evidence/c500_vllm_engine_20260728`. It proves basic public Engine loading and
multi-step generation on C500. Chunk-boundary prefill, continuous batching,
state-cache, performance, parallel, quantization, and speculative decoding are
separate gates.

The public 2.9B forced chunk-boundary A/B gate is recorded in
`evidence/c500_vllm_chunked_prefill_20260728`. With prompt lengths 129/193, the
64-token scheduler budget requires at least six prefill steps and reproduces
the 512-token-budget baseline output exactly. This closes one exact
chunked-prefill correctness cell, not continuous batching, state-cache reuse,
or the full model/shape matrix.

The public 2.9B per-request state-slot gate is recorded in
`evidence/c500_vllm_state_slots_20260728`. Two prompts reproduce their solo
greedy outputs when submitted in A/B order and again in B/A order on the same
Engine. This closes one exact recurrent-state isolation and completed-slot
reuse cell. It is not prefix-cache reuse or hit-rate evidence and does not
close continuously arriving dynamic batching or scheduler preemption.

The public 2.9B continuously arriving gate is recorded in
`evidence/c500_vllm_dynamic_batch_20260728`. It uses `AsyncLLM`, starts request
A, submits B only after A's first streamed token, verifies that B produces a
token before A completes, and matches both requests against their solo greedy
references. This closes one exact overlapping-arrival correctness cell, not
the full arrival-rate/model/shape matrix, prefix-cache hit rate, preemption, or
performance parity.

The public 2.9B recurrent prefix-state cache gate is recorded in
`evidence/c500_vllm_state_prefix_cache_20260728`. Four requests each restore a
128-token cached state while an unrelated control request reports zero cached
tokens. The request hit rate is `0.8`, every cache-enabled greedy output exactly
matches cache-disabled generation, and median shared-prefix TTFT is `0.7007x`
the cache-disabled row. The complete mixed batch is `1.0512x` slower because it
also measures cache writes and the cold control, so this is a state-cache
correctness and hit-latency cell rather than an aggregate throughput claim.
Cross-process sharing, preemption restore, eviction stress and the full
model/shape matrix remain separate gates.
