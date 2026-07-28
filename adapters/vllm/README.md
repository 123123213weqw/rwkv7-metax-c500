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
not expose the newer `triton.knobs` API. The seventh adapts a Triton pointer
helper to the vendor compiler's constexpr representation. The eighth adds a
registered RWKV channel-mix safety switch so a hardware plugin can select the
dense path when the CUDA-specific no-FC sparse kernels are not validated.

The separate MetaX plugin patch lets an explicit V2 RWKV runner opt in, loads
the reduced RWKV extension at the normal platform-kernel lifecycle point,
fills the missing vendor `torch.accelerator` memory methods from `torch.cuda`,
and skips unrelated MLA, model and quant registrations removed by the reduced
build. On the reduced C500 profile it defaults the no-FC channel-mix switch to
the dense path while preserving explicit user override and all non-MetaX
defaults. None of the patches modifies the host SDK installation.

The first native WKV compile and numerical gate is recorded in
`evidence/c500_vllm_native_build_20260727`. It is kernel evidence, not a
public-engine performance claim.

The public `vllm.LLM` 2.9B mixed-length batch gate is recorded in
`evidence/c500_vllm_engine_20260728`. It proves basic public Engine loading and
multi-step generation on C500, not the remaining chunk-boundary prefill,
continuous batching, state-cache, performance, parallel, quantization, or
speculative-decoding gates.
