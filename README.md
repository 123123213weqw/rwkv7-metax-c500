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

See [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) before making a support or
performance claim.

## Status

The exact-card environment gate passes on one MetaX C500 64GB with MXMACA
3.5.3.20 and MetaX PyTorch 2.8.0. The redacted artifact and checksum are in
[`evidence/c500_environment_20260726`](evidence/c500_environment_20260726/README.md).
HF, vLLM and SGLang model-level acceptance remains tracked independently.

## License

Apache-2.0.
