# Hugging Face adapter

Target: run the native/no-FLA RWKV-7 Transformers implementation through the
MXMACA PyTorch CUDA-compatible API, then add C500-specific fused kernels where
exact-card A/B evidence justifies them.

First implementation checkpoint: load a converted 0.1B model, match the CPU
oracle, run cached greedy generation, and execute one Trainer + PEFT LoRA step.

Apply the C500 patchset to the exact locked HF source before running the GPU
acceptance:

```bash
python scripts/apply_track_patches.py hf worktrees/hf_adapter
python adapters/hf/smoke_native_c500.py \
  --output evidence/local/hf-native-tiny.json
python adapters/hf/accept_real_checkpoint.py \
  --model /path/to/rwkv7-g1d-0.4b-hf \
  --output evidence/local/hf-real-0.4b.json \
  --reload-dir /tmp/hf-real-0.4b-reload
```

The first patch keeps RWKV-7 key normalization in FP32. This is required
because FP16 cannot represent PyTorch's default `F.normalize` epsilon; without
it, a zero or near-zero key norm can produce NaNs on the C500 native path.
