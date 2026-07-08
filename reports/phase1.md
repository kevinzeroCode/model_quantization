# Phase 1 Report - BF16 Baseline + Frozen Prompts

## Status

Phase 1 completed on 2026-07-08 (Asia/Taipei) on RTX 4090. Results are sourced from `results/perf.csv`, `results/quality.csv`, and `results/raw/` only.

Acceptance summary:
- `results/perf.csv`: 11 BF16+YaRN vLLM speed rows.
- `results/quality.csv`: 60 BF16+YaRN NIAH cells, 8 LongBench subset rows, 1 vLLM PPL row, plus 15 norope comparison cells.
- Prompt set frozen: 3 languages x 60 prompts = 180 prompts.

## Frozen Prompts

| file | n | sha256 |
|---|---:|---|
| prompts/niah_code.jsonl | 60 | `68c6031405dcf5dae33658a3088047a1ec994d413802c73ac1d47294684dd4de` |
| prompts/niah_en.jsonl | 60 | `eb86377becb0e41eda842c0193d07da445107b2a7634f4c319f5537dda2a6f3d` |
| prompts/niah_zh.jsonl | 60 | `f6430c5533dd5b9fa52c3c4e80b370e3cb4f7a00148311c612a0ac182a19f0e4` |

English corpus uses `wikitext-103` by default because PG-19 download exceeded the Phase 1 timebox; `USE_PG19=1` remains available in `scripts/gen_prompts.py` if a PG-19 variant is required later.

## KV Budget

```text
FP16 KV = 56.0 KiB/token
| context | fp16 | fp8 | int4.5 | hybrid3.3 | int2.5 |
|---|---|---|---|---|---|
| 4K | 0.22 | 0.11 | 0.06 | 0.05 | 0.03 | (GiB)
| 16K | 0.88 | 0.44 | 0.25 | 0.18 | 0.14 | (GiB)
| 32K | 1.75 | 0.88 | 0.49 | 0.36 | 0.27 | (GiB)
| 64K | 3.50 | 1.75 | 0.98 | 0.72 | 0.55 | (GiB)
| 128K | 7.00 | 3.50 | 1.97 | 1.44 | 1.09 | (GiB)
| 256K | 14.00 | 7.00 | 3.94 | 2.89 | 2.19 | (GiB)
| 1M | 56.00 | 28.00 | 15.75 | 11.55 | 8.75 | (GiB)
```

vLLM server capacity observed for `models/Qwen2.5-7B-Instruct-yarn64k`:
- BF16 weights: 14.2717 GiB.
- Available KV cache memory: 5.92 GiB.
- GPU KV cache size: 110,896 tokens.
- vLLM reported maximum concurrency for 66,560 tokens/request: 1.67x.

Conclusion for 64K: practical resident batch ceiling is 1 long request. `ctx=63,488,c=4` runs by queueing and shows the expected throughput/TTFT degradation; `c=8` was skipped by runbook.

## Speed Grid

| ctx | concurrency | n | mean TTFT ms | mean TPOT ms | output tok/s |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1 | 6 | 330.6 | 16.16 | 57.31 |
| 4,096 | 4 | 12 | 431.3 | 18.61 | 197.34 |
| 4,096 | 8 | 24 | 674.6 | 21.76 | 326.94 |
| 16,384 | 1 | 6 | 1568.8 | 16.95 | 43.45 |
| 16,384 | 4 | 12 | 1829.9 | 30.12 | 106.86 |
| 16,384 | 8 | 24 | 6814.7 | 57.30 | 92.51 |
| 32,768 | 1 | 6 | 3783.8 | 18.02 | 30.55 |
| 32,768 | 4 | 12 | 9347.5 | 56.65 | 41.58 |
| 32,768 | 8 | 24 | 31786.4 | 56.98 | 38.96 |
| 63,488 | 1 | 6 | 9600.3 | 19.98 | 17.42 |
| 63,488 | 4 | 12 | 46021.7 | 48.38 | 15.76 |

64K behavior:
- `63,488,c=1`: mean TTFT 9.6s, output throughput 17.42 tok/s.
- `63,488,c=4`: mean TTFT 46.0s, output throughput 15.76 tok/s, matching the KV pool limit rather than linear batching.

## NIAH

BF16+YaRN is all green across zh/en/code, 4K through 63.5K. Every ctx x depth cell is 3/3 correct.

| subset | 4K | 16K | 32K | 63.5K |
|---|---:|---:|---:|---:|
| niah_zh | 1.000 | 1.000 | 1.000 | 1.000 |
| niah_en | 1.000 | 1.000 | 1.000 | 1.000 |
| niah_code | 1.000 | 1.000 | 1.000 | 1.000 |

## YaRN vs None Appendix Check

Original `Qwen/Qwen2.5-7B-Instruct` with `max_model_len=32768` is valid for the current frozen 4K/16K prompts and overflows at frozen 32K because prompt overhead pushes request length above 32,768.

| ctx | mean acc | note |
|---:|---:|---|
| 4,096 | 1.000 | valid |
| 16,384 | 1.000 | valid |
| 32,768 | 0.000 | FAIL_CONTEXT_OVERFLOW: input 32853-32856 > max_model_len 32768 |

The 32K norope zeros should not be interpreted as retrieval misses; server logs show actual inputs of 32,853-32,856 tokens.

## LongBench

`limit=50` per subset, official LongBench metrics/config from commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`.

| subset | metric | value | n |
|---|---|---:|---:|
| multifieldqa_zh | qa_f1_zh_score | 0.6288 | 50 |
| dureader | rouge_zh_score | 0.2731 | 50 |
| passage_retrieval_zh | retrieval_zh_score | 0.8600 | 50 |
| hotpotqa | qa_f1_score | 0.5523 | 50 |
| 2wikimqa | qa_f1_score | 0.4859 | 50 |
| passage_retrieval_en | retrieval_score | 1.0000 | 50 |
| lcc | code_sim_score | 0.0926 | 50 |
| repobench-p | code_sim_score | 0.0534 | 50 |

## PPL

vLLM WikiText-2 teacher-forced PPL:
- model: `Qwen/Qwen2.5-7B-Instruct`
- chunk: 4096 tokens
- chunks: 40
- tokens scored: 163800
- PPL: 6.9572

The PPL script now runs `batch_size=1` and `gpu_memory_utilization=0.65`; earlier higher KV reservation OOMed during prompt-logprob `log_softmax`.

## Issues And Workarounds

- vLLM must run with `VLLM_USE_FLASHINFER_SAMPLER=0` on this host due flashinfer sampler JIT failure recorded in `ISSUES.md`.
- PG-19 was replaced by runbook-allowed `wikitext-103` fallback for frozen English prompts.
- norope 32K comparison overflows by prompt overhead; use 4K/16K for valid current comparison or create a separate approx 31K appendix prompt set later.
- vLLM PPL prompt-logprobs require reduced KV reservation and batch size 1 on RTX 4090.
