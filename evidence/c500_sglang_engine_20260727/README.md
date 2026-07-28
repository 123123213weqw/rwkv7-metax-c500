# C500 SGLang public Engine gate

Date: 2026-07-27

This artifact records a real `sglang.Engine` run on one MetaX C500 using the
locked MetaX SGLang patch set and a real RWKV-7 0.4B checkpoint view. Two
tokenized requests with prompt lengths 4 and 5 were submitted together, and
each request produced exactly four greedy tokens.

The correctness-first gate uses FP16, the PyTorch sampler, eager execution,
overlap scheduling disabled, CUDA graph disabled and radix cache disabled. Its
28.51 second elapsed value includes Engine startup, model loading, generation
and shutdown, so it is not a throughput measurement.

The result establishes public Engine construction and mixed-length batch
execution on C500. It does not establish production throughput, chunked
prefill, recurrent state-cache reuse, radix-cache hit rate, quantized
inference, speculative decoding or output quality. Those remain separate
acceptance cells in `acceptance_matrix.json`.

Reproduce from the repository root after installing the vendor runtime:

```bash
python scripts/apply_track_patches.py sglang /path/to/metax-sglang
python adapters/sglang/accept_engine_c500.py \
  --model /path/to/rwkv7-g1d-0.4b-sglang \
  --output result.json
```

`result.json` contains no host name, account name, network address or private
absolute path.
