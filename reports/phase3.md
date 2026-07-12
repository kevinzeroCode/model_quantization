# Phase 3 Report - FP8 KV cache factorial

Date: 2026-07-08
Host: RTX 4090 24GB
Runtime: vLLM 0.10.2
Baseline: Phase 1 BF16 + fp16 KV and Phase 2 AWQ + fp16 KV

## Scope

Phase 3 completed the 2x2 weight/KV factorial for the two main configurations:

| run_id | model | weight_quant | KV quant | vLLM backend note |
|---|---|---:|---:|---|
| p1-bf16 | models/Qwen2.5-7B-Instruct-yarn64k | bf16 | fp16 | V1 / FlashAttention path |
| p2-awq | models/Qwen2.5-7B-Instruct-AWQ-yarn64k | awq_w4 | fp16 | V1 / awq_marlin path |
| p3-bf16-fp8 | models/Qwen2.5-7B-Instruct-yarn64k | bf16 | fp8_e4m3 | V0 fallback / XFormers |
| p3-awq-fp8 | models/Qwen2.5-7B-Instruct-AWQ-yarn64k | awq_w4 | fp8_e4m3 | V0 fallback / XFormers / awq_marlin |

Both FP8 KV runs used `--kv-cache-dtype fp8_e4m3 --calculate-kv-scales`. vLLM accepted e4m3, so the e5m2 fallback was not used. vLLM 0.10.2 warned that `--kv-cache-dtype` is unsupported by the V1 engine and fell back to the V0 engine; FP8 KV also used the XFormers attention backend instead of FlashAttention.

## Capacity

All configurations used `--max-model-len 66560` and `--gpu-memory-utilization 0.92`.

| config | weights_gb | KV pool tokens | ratio vs BF16 fp16 | max concurrency at 66,560 tokens |
|---|---:|---:|---:|---:|
| BF16 + fp16 KV | 14.2717 | 110,896 | 1.00x | 1.67x |
| BF16 + fp8 KV | 14.2717 | 219,600 | 1.98x | 3.30x |
| AWQ W4 + fp16 KV | 5.2271 | 280,224 | 2.53x | 4.21x |
| AWQ W4 + fp8 KV | 5.2271 | 558,320 | 5.04x | 8.39x |

The capacity gain is close to multiplicative. AWQ alone produced 2.53x the BF16 fp16 KV pool, FP8 KV alone produced 1.98x, and AWQ plus FP8 KV produced 5.04x. The product of the individual gains is about 5.01x, which matches the measured combined result.

## Serving Speed

Generated token throughput (`gen_tps`) did not improve with FP8 KV in this vLLM version. The FP8 runs had higher capacity, but the V0/XFormers fallback made the hot path slower than the fp16 KV baselines. Full values are in `reports/summary_tables.md`.

| ctx, concurrency | BF16 fp16 | BF16 fp8 | AWQ fp16 | AWQ fp8 |
|---|---:|---:|---:|---:|
| 32,768 x 4 | 41.58 | 14.80 | 77.80 | 11.80 |
| 63,488 x 1 | 17.42 | 3.03 | 21.07 | 0.22 |
| 63,488 x 4 | 15.76 | 5.42 | 20.50 | 2.51 |

The long-context FP8 speed runs also showed early EOS or very short outputs. At 63,488 x 1, BF16 fp8 generated 295 output tokens across 6 trials and AWQ fp8 generated only 18 output tokens across 6 trials, while the fp16 KV baselines generated the full 1,536 output tokens at the same benchmark point.

## NIAH

NIAH was run on the frozen zh prompt set. Both fp16 KV baselines retained perfect retrieval through 63K. Both FP8 KV configurations failed the 63K retrieval target.

| run_id | KV quant | total zh NIAH | 4K | 16K | 32K | 63K |
|---|---:|---:|---:|---:|---:|---:|
| p1-bf16 | fp16 | 60/60 | 1.00 | 1.00 | 1.00 | 1.00 |
| p2-awq | fp16 | 60/60 | 1.00 | 1.00 | 1.00 | 1.00 |
| p3-bf16-fp8 | fp8_e4m3 | 2/60 | 0.00 | 0.13 | 0.00 | 0.00 |
| p3-awq-fp8 | fp8_e4m3 | 9/60 | 0.60 | 0.00 | 0.00 | 0.00 |

This is a functional quality failure for the current FP8 KV serving path. The largest KV pool is not useful for the 62K target if retrieval accuracy collapses at that length.

## LongBench

Limit: 50 examples per subset. Budget: 31,500 context tokens.

| subset | BF16 fp16 | BF16 fp8 | AWQ fp16 | AWQ fp8 |
|---|---:|---:|---:|---:|
| multifieldqa_zh | 0.6288 | 0.3713 | 0.6547 | 0.3550 |
| dureader | 0.2731 | 0.1406 | 0.2747 | 0.1278 |
| passage_retrieval_zh | 0.8600 | 0.2400 | 0.8200 | 0.1933 |
| hotpotqa | 0.5523 | 0.1496 | 0.5206 | 0.0784 |
| 2wikimqa | 0.4859 | 0.1573 | 0.4277 | 0.0871 |
| passage_retrieval_en | 1.0000 | 0.1383 | 0.9600 | 0.0340 |
| lcc | 0.0926 | 0.1834 | 0.1232 | 0.1422 |
| repobench-p | 0.0534 | 0.1886 | 0.0494 | 0.1530 |

Mean over these 8 subsets:

| config | mean |
|---|---:|
| BF16 fp16 | 0.4933 |
| BF16 fp8 | 0.1961 |
| AWQ fp16 | 0.4788 |
| AWQ fp8 | 0.1464 |

The code-subset scores for FP8 are higher than the fp16 baselines in this limit-50 slice, but they should not override the retrieval evidence. The QA and passage-retrieval subsets fall sharply, and NIAH at 32K/63K collapses.

## Answer to Phase 3 Questions

For best quality x capacity at the 62K target, the current winner is AWQ W4 with fp16 KV (`p2-awq`), not AWQ plus FP8 KV. AWQ fp16 has 280,224 KV tokens, perfect 63K zh/en NIAH from Phase 2, LongBench mean 0.4788, and 63K x 4 throughput of 20.50 gen_tps. BF16 fp16 preserves quality too, but its KV pool is only 110,896 tokens. AWQ fp8 has the largest KV pool at 558,320 tokens, but it fails 63K NIAH and drops LongBench mean to 0.1464.

The memory and capacity gains from weight quantization and KV quantization are additive in memory terms and almost multiplicative in KV-token capacity. AWQ weight-only frees model memory, FP8 KV halves KV storage, and the combined run reaches 5.04x the BF16 fp16 KV pool. The speed and quality effects are not additive: in vLLM 0.10.2, FP8 KV forces the V0/XFormers path and causes severe quality loss on long-context retrieval.

Phase 3 passes the data-completeness criteria but fails the practical acceptance criteria for using FP8 KV as the main 62K serving configuration. The next phases should treat FP8 KV as a diagnostic path unless a newer runtime or different KV scaling method fixes the retrieval collapse.
