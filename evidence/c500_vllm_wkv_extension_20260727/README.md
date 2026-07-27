# C500 vLLM WKV extension evidence

This artifact records the RWKV vLLM WKV CUDA-compatible extension at commit
`7d16dbf88077834fbf93a0a65a882286319d7bd6` compiled and executed on one
MetaX C500 64GB with PyTorch `2.8.0+metax3.5.3.9`.

The probe packs two requests and five tokens while using four recurrent-state
slots. The extension output matches the PyTorch oracle with cosine `1.0` and
maximum absolute error `2.7939677e-09`; the updated state reaches cosine
`0.9999997616` and maximum absolute error `3.7252903e-09`.

This is a native-kernel compile and numerical-correctness gate. It does not by
itself prove that the public vLLM Engine, dynamic batching, chunked prefill,
state-cache reuse, throughput, or memory acceptance gates pass. Those require
separate engine-level evidence.

Reproduce from the locked RWKV vLLM source tree with the C500 environment
loaded:

```bash
python adapters/vllm/probe_wkv_extension.py \
  --source /path/to/vllm-rwkv \
  --output /path/to/result.json
```

The artifact omits the SSH endpoint, account name, hostname, credentials, and
credential-bearing environment variables.
