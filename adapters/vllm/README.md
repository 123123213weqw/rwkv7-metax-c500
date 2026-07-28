# vLLM adapter

Target: combine the RWKV-7 vLLM model/state scheduler with the official MetaX
hardware plugin. The initial route is eager and correctness-first. Graph capture
and C500 kernels follow only after public-engine dynamic batch and chunked
prefill pass.

State-slot reuse is required, but it is not reported as prefix-cache reuse.

Apply the locked C500 patch set to the exact RWKV vLLM upstream before building:

```bash
python scripts/apply_track_patches.py vllm /path/to/vllm-rwkv
VLLM_BUILD_PROFILE=rwkv python -m pip install --no-deps --no-build-isolation -e /path/to/vllm-rwkv
```

The first patch selects the vendor CUDA-compatible compiler and separates the
real `cucc` version probe from the mock compiler used by `cmake_maca`. The
second exposes the native C++/CUDA language standards as CMake cache settings;
the C500 CUDA 11.6-compatible toolchain uses
`-DVLLM_CXX_STANDARD=17 -DVLLM_CUDA_STANDARD=17`. The third enables the MACA
compile contract, uses the traditional Torch registration API available in the
vendor PyTorch 2.8 build, and provides a C500-safe FP16 WKV copy path when
NVIDIA `cp.async` assembly is unavailable. None of the patches modifies the
host SDK installation.

The first native WKV compile and numerical gate is recorded in
`evidence/c500_vllm_wkv_extension_20260727`. It is kernel evidence, not a
public-engine performance claim.
