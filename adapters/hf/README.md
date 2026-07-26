# Hugging Face adapter

Target: run the native/no-FLA RWKV-7 Transformers implementation through the
MXMACA PyTorch CUDA-compatible API, then add C500-specific fused kernels where
exact-card A/B evidence justifies them.

First implementation checkpoint: load a converted 0.1B model, match the CPU
oracle, run cached greedy generation, and execute one Trainer + PEFT LoRA step.
