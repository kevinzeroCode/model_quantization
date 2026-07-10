# Phase 2 Report - AWQ / GPTQ weight-only baseline

Date: 2026-07-08
Host: RTX 4090 24GB
Runtime: vLLM 0.10.2
Baseline: Phase 1 BF16 + YaRN, fp16 KV

## Scope

Phase 2 repeated the Phase 1 serving path for two weight-only 4-bit models:

| run_id | model | weight_quant | vLLM kernel | KV quant |
|---|---|---:|---|---|
| p2-awq | models/Qwen2.5-7B-Instruct-AWQ-yarn64k | awq_w4 | awq_marlin | fp16 |
| p2-gptq | models/Qwen2.5-7B-Instruct-GPTQ-Int4-yarn64k | gptq_w4 | gptq_marlin | fp16 |

NIAH was run on zh/en only, as planned. Code NIAH is left for Phase 7.

## Capacity

All three configurations used `--max-model-len 66560` and `--gpu-memory-utilization 0.92`.

| config | weights_gb | KV pool tokens | KV ratio vs BF16 | max concurrency at 66,560 tokens |
|---|---:|---:|---:|---:|
| BF16 + fp16 KV | 14.2717 | 110,896 | 1.00x | 1.67x |
| AWQ W4 + fp16 KV | 5.2271 | 280,224 | 2.53x | 4.21x |
| GPTQ W4 + fp16 KV | 5.2033 | 280,704 | 2.53x | 4.22x |

AWQ freed about 9.04 GiB of model-weight memory relative to BF16 and converted that headroom into 280,224 KV tokens, or 2.53x the BF16 KV pool. GPTQ was nearly identical, freeing about 9.07 GiB and reaching 280,704 KV tokens.

## Serving Speed

Generated token throughput (`gen_tps`) improved across nearly all comparable points. Full values are in `reports/summary_tables.md`.

| ctx, concurrency | BF16 | AWQ | GPTQ |
|---|---:|---:|---:|
| 4,096 x 1 | 57.31 | 126.90 | 127.17 |
| 4,096 x 8 | 326.94 | 544.82 | 530.14 |
| 16,384 x 1 | 43.45 | 81.75 | 82.00 |
| 32,768 x 1 | 30.55 | 43.56 | 43.66 |
| 63,488 x 1 | 17.42 | 21.07 | 21.11 |
| 63,488 x 4 | 15.76 | 20.50 | 19.87 |

The largest gains are at short and medium context lengths, where lower weight bandwidth matters more. At 32K and 64K with high concurrency, both AWQ and GPTQ become queue/prefill constrained; the throughput still stays above BF16, but TTFT and TPOT grow sharply. This is expected for the pressure points and was recorded as the real serving behavior rather than tuned away.

## PPL

WikiText-2 PPL was computed through vLLM prompt logprobs over 163,800 tokens with 4,096-token chunks.

| config | PPL | delta vs BF16 | relative delta |
|---|---:|---:|---:|
| BF16 | 6.9572 | - | - |
| AWQ W4 | 7.3326 | +0.3754 | +5.40% |
| GPTQ W4 | 7.2823 | +0.3251 | +4.67% |

GPTQ stays under the runbook's expected +5% PPL increase. AWQ is slightly above that threshold at +5.40%, but this is not a PPL explosion and does not look like an accidental fp16 dequantized path or a broken quantization load. Both vLLM runs explicitly used the Marlin quantized kernels.

## NIAH

Both quantized models retained perfect NIAH retrieval on zh/en across 4K, 16K, 32K, and 63K contexts.

| run_id | subset | total | result |
|---|---|---:|---|
| p2-awq | niah_zh | 60 | 60/60 |
| p2-awq | niah_en | 60 | 60/60 |
| p2-gptq | niah_zh | 60 | 60/60 |
| p2-gptq | niah_en | 60 | 60/60 |

This means Phase 2 did not show a retrieval failure from weight-only quantization under YaRN for zh/en NIAH.

## LongBench

Limit: 50 examples per subset. Budget: 31,500 context tokens.

| subset | BF16 | AWQ | GPTQ |
|---|---:|---:|---:|
| multifieldqa_zh | 0.6288 | 0.6547 | 0.6315 |
| dureader | 0.2731 | 0.2747 | 0.2596 |
| passage_retrieval_zh | 0.8600 | 0.8200 | 0.8000 |
| hotpotqa | 0.5523 | 0.5206 | 0.5238 |
| 2wikimqa | 0.4859 | 0.4277 | 0.5123 |
| passage_retrieval_en | 1.0000 | 0.9600 | 1.0000 |
| lcc | 0.0926 | 0.1232 | 0.0748 |
| repobench-p | 0.0534 | 0.0494 | 0.0428 |

Mean over these 8 subsets:

| config | mean |
|---|---:|
| BF16 | 0.4933 |
| AWQ | 0.4788 |
| GPTQ | 0.4806 |

The LongBench deltas are mixed and small at this sample size. AWQ and GPTQ lose some retrieval score on `passage_retrieval_zh`, while GPTQ improves `2wikimqa` in this 50-example slice. Code subsets remain noisy and low overall; they should be interpreted cautiously until the larger Phase 7 code-focused run.

## Answer to Phase 2 Questions

AWQ reduces model memory from 14.2717 GiB to 5.2271 GiB and increases the vLLM KV pool from 110,896 tokens to 280,224 tokens, a 2.53x increase. GPTQ reduces model memory to 5.2033 GiB and increases the KV pool to 280,704 tokens, also 2.53x.

The quality cost is visible but not severe. PPL increases by +5.40% for AWQ and +4.67% for GPTQ. NIAH zh/en stays perfect for both. LongBench mean score drops from 0.4933 to 0.4788 for AWQ and 0.4806 for GPTQ on the Phase 2 limit-50 subset.

Phase 2 passes the acceptance criteria: perf.csv has 22 new AWQ/GPTQ rows, quality.csv has AWQ/GPTQ NIAH, LongBench, and PPL rows, and summary tables now compare BF16/AWQ/GPTQ directly.
