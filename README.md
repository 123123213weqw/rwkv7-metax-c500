# RWKV-7 on MetaX C500

This repository carries the MetaX C500 adaptation and reproducible acceptance
for three independent RWKV-7 tracks:

- Hugging Face / Transformers
- vLLM
- SGLang

The repository uses the MXMACA CUDA-compatible software stack, but treats every
CUDA extension and optimized route as unverified until it passes on an exact
C500 machine. It does not copy complete upstream repositories. Each track is
maintained as a small plugin or patch set against a pinned upstream revision.

## First machine probe

Run this inside the C500 host or its official MXMACA container:

```bash
python3 scripts/probe_c500.py \
  --run-smoke \
  --output evidence/local/c500-environment.json
python3 scripts/verify_probe.py evidence/local/c500-environment.json
```

The probe records driver, SDK, framework and device information plus small
fp16/bf16 matrix operations. It deliberately excludes usernames, hostnames,
network addresses, environment secrets and command-line credentials.

## Layout

```text
adapters/hf/       Transformers adapter integration
adapters/vllm/     vLLM MetaX hardware-plugin integration
adapters/sglang/   SGLang MetaX integration
docs/              design and acceptance contracts
evidence/          committed machine-readable acceptance artifacts
scripts/           environment, bootstrap and verification commands
src/               shared probe and evidence utilities
tests/             CPU-runnable contract tests
```

See [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) and the machine-readable
[`acceptance_matrix.json`](acceptance_matrix.json) before making a support or
performance claim. Completion is evaluated across all six released model sizes
(0.1B through 13.3B), all required batch/prompt/decode cells and each capability
gate; smoke evidence is reported separately from production acceptance.

## Status

The exact-card environment gate passes on one MetaX C500 64GB with MXMACA
3.5.3.20 and MetaX PyTorch 2.8.0. The redacted artifact and checksum are in
[`evidence/c500_environment_20260726`](evidence/c500_environment_20260726/README.md).
The HF native/no-FLA tiny-model gate passes FP32, FP16 and BF16 inference,
cache, generation and backward after the locked FP32-normalization patch; see
[`evidence/c500_hf_native_20260726`](evidence/c500_hf_native_20260726/README.md).
The real 0.4B HF gate also passes CPU-oracle FP16 inference, split/chunked
prefill, B8 dynamic cache selection, save/reload, BF16 Trainer + LoRA, and a
strict FP32 PEFT merge roundtrip; see
[`evidence/c500_hf_real_0_4b_20260726`](evidence/c500_hf_real_0_4b_20260726/README.md).
The locked RWKV vLLM reduced profile now builds and installs its native WKV,
rapid-sampling and allocator modules on C500. Both FP32-state/FP16-IO and
FP16-state packed WKV operators pass numerical gates; see
[`evidence/c500_vllm_native_build_20260727`](evidence/c500_vllm_native_build_20260727/README.md).
Public-engine vLLM, optimized HF and SGLang acceptance remain tracked
independently from these completed smoke and operator gates.

## License

Apache-2.0.
