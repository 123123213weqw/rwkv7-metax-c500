# C500 HF native tiny-model acceptance

This artifact records a deterministic two-layer RWKV-7 native/no-FLA run on
one MetaX C500 64GB using the HF adapter at
`9d7dbc9213f0fb6b021d8dd0e3a828dad5fcd4af` plus this repository's locked
FP32-normalization patch.

The same model state and token batch were exercised in FP32, FP16 and BF16.
All three rows pass full forward, recurrent-cache handoff, dynamic cache batch
selection, four-token greedy generation, causal loss, backward, and finite
gradient checks. FP16 forward cosine against the CPU FP32 oracle is
`0.9999995828`; BF16 is `0.9999823570`.

This is a synthetic compatibility and numerical-correctness gate. The elapsed
times include eager tiny-model overhead and are not a production throughput
claim. Real-checkpoint and matrix evidence are tracked separately.

Reproduce from the repository root after applying the HF patchset:

```bash
python adapters/hf/smoke_native_c500.py \
  --output evidence/local/hf-native-tiny.json
```

The JSON deliberately excludes hostnames, addresses, credentials and user
paths.
