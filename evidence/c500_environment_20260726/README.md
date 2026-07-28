# MetaX C500 environment evidence

Exact-card inventory captured on 2026-07-26:

- MetaX C500, 64GB, one card
- MX-SMI 2.2.12
- kernel-mode driver 3.8.30
- MXMACA 3.5.3.20
- Ubuntu 22.04, x86_64
- Python 3.10.10
- PyTorch 2.8.0+metax3.5.3.9
- Triton 3.0.0+metax3.5.3.9
- Transformers 4.57.6
- vLLM 0.17.0
- flash-linear-attention 0.4.0+metax3.5.3.9torch2.8

`environment.json` passes `scripts/verify_probe.py` including fp16 and bf16
matrix correctness. The recorded elapsed values are synchronized single-call
smoke telemetry. They include first-use initialization and are not performance
benchmarks.

The artifact intentionally omits the SSH endpoint, username, hostname and all
credential-bearing environment variables.
