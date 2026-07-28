# C500 HF real-checkpoint acceptance

This directory records the HF native/no-FLA adapter at commit
`9d7dbc9213f0fb6b021d8dd0e3a828dad5fcd4af` on one MetaX C500 64GB with the
real `rwkv7-g1d-0.4b-hf` checkpoint.

## Passing gates

- FP16 `AutoModelForCausalLM` forward matches a CPU FP32 oracle at cosine
  `0.9999972582`.
- Split and three-token chunked prefill both match the single-pass final logits
  at cosine `0.9999954700`.
- B8 recurrent cache creation and B8-to-B3 dynamic selection pass.
- Eight-token greedy generation produces `Hello! How can I assist you today`.
- `save_pretrained` followed by a fresh reload matches at cosine
  `0.9999954700`.
- BF16 HF Trainer + PEFT LoRA runs three steps; loss falls from `3.0154` to
  `1.6506`, and 144/144 trainable parameter tensors update.
- FP32 PEFT adapter save/load, merge/unmerge, merge-and-unload and fresh greedy
  generation pass. The merged logits max-abs difference is `4.5776e-05` under
  the strict `1e-4` gate.

## FP16 merge diagnostic

The FP16 PEFT row is retained as a negative precision probe. Adapter reload is
exact and all three greedy tails agree, but merging LoRA weights into the FP16
base changes logits by up to `0.234375`, above the declared `0.1` max-abs gate.
Users that need strict merge/reload equivalence should merge in FP32 and cast
the resulting model afterward. This does not invalidate BF16 Trainer or
unmerged FP16 adapter inference.

The eager elapsed time in `hf-real-0.4b.json` is a compatibility measurement,
not an optimized throughput claim. The files exclude credentials, hostnames
and network addresses.
