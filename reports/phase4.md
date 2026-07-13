# Phase 4 Report - HF low-bit KV cache baseline

Date: 2026-07-08
Host: RTX 4090 24GB
Runtime: Hugging Face transformers 4.57.6 / torch 2.8.0+cu128
Model: models/Qwen2.5-7B-Instruct-yarn64k

## Scope

Phase 4 measured the HF research route for low-bit KV cache quality. The original runbook target was transformers `quanto` cache (`int4_quanto` / `int2_quanto`). During execution, `quanto4` passed the 0.5B smoke test after fixing the local cache API path, but produced unusable 7B NIAH output. Following the runbook fallback, HQQ was installed and used for the official low-bit baseline rows:

| run_id | backend | kv arg | kv_quant | effective KV bits | residual window |
|---|---|---:|---:|---:|---:|
| p4-fp16 | HF DynamicCache | fp16 | fp16 | 16.0 | n/a |
| p4-hqq4 | HF QuantizedCache / HQQ | hqq4 | int4_hqq | 4.5 | 128 |
| p4-hqq2 | HF QuantizedCache / HQQ | hqq2 | int2_hqq | 2.5 | 128 |

A diagnostic `p4-quanto4` zh NIAH run is retained in the raw results and summary table. It is not treated as the official Phase 4 baseline because it generated corrupted text at every tested context.

## Implementation Notes

HF long-context generation now uses chunked prefill through the same cache object instead of calling `model.generate()` on the full prompt. This avoided single-shot 16K/32K prefill OOM and keeps the KV cache path active. The HF benchmark script was also fixed to run under `torch.no_grad()`; without that, autograd graphs were retained and even the 0.5B smoke test OOMed.

Cached-PPL now defaults to `wikitext-103` unless `USE_PG19=1` is set. PG-19 attempted to download many small files and exceeded a reasonable Phase 4 setup window, matching the earlier Phase 1 PG-19 issue.

## NIAH

NIAH was run on frozen zh/en prompts up to 32,768 tokens. Each language has 45 examples: 3 context lengths x 5 depths x 3 seeds.

| run_id | kv_quant | zh | en | first failing point |
|---|---:|---:|---:|---|
| p4-fp16 | fp16 | 45/45 | 45/45 | none |
| p4-hqq4 | int4_hqq | 0/45 | 0/45 | 4K, depth 0.1 |
| p4-hqq2 | int2_hqq | 0/45 | 0/45 | 4K, depth 0.1 |
| p4-quanto4 diagnostic | int4_quanto | 0/45 | not run | 4K, depth 0.1 |

The low-bit failure is not a long-context-only failure. Both HQQ 4-bit and 2-bit fail from the first 4K cell. Raw outputs show repetitive or corrupted text rather than merely wrong but fluent answers.

## Cached-PPL

Cached-PPL used `wikitext-103`, `score_tail=1024`, `n_docs=1`, and chunked prefill with `prefill_chunk=512`.

| run_id | kv_quant | 16K PPL | 32K PPL |
|---|---:|---:|---:|
| p4-fp16 | fp16 | 5.4877 | 8.0138 |
| p4-hqq4 | int4_hqq | 8485.7470 | 26420.3876 |
| p4-hqq2 | int2_hqq | 39672.8659 | 188153.7391 |

The PPL result confirms the NIAH collapse. HQQ 4-bit is already unusable at 16K, and HQQ 2-bit is worse by 4.68x at 16K and 7.12x at 32K relative to HQQ 4-bit.

## HF Runtime Reference

These rows are HF-only references and must not be compared directly with vLLM throughput.

| run_id | ctx | TTFT ms | TPOT ms | gen_tps | peak VRAM GB |
|---|---:|---:|---:|---:|---:|
| p4-fp16 | 16,384 | 2582.4 | 19.73 | 50.7 | 15.33 |
| p4-fp16 | 32,768 | 7070.4 | 22.65 | 44.1 | 16.45 |
| p4-hqq4 | 16,384 | 7338.5 | 27.22 | 36.7 | 14.75 |
| p4-hqq4 | 32,768 | 27081.9 | 41.75 | 24.0 | 15.27 |
| p4-hqq2 | 16,384 | 8092.2 | 28.20 | 35.5 | 14.64 |
| p4-hqq2 | 32,768 | 31043.4 | 43.73 | 22.9 | 15.05 |

The measured HF peak VRAM reduction is small because model weights and runtime workspaces dominate the process peak. Phase 4 should therefore use `eff_kv_bits` as the analytic memory-budget signal and use the HF VRAM rows only as a local runtime reference.

## Answer to Phase 4 Questions

There is no useful int2-vs-int4 quality cliff in NIAH because both low-bit HQQ configurations collapse immediately at 4K/depth 0.1. The PPL cliff still distinguishes them: int2 is materially worse than int4, increasing cached-PPL from 8485.7470 to 39672.8659 at 16K and from 26420.3876 to 188153.7391 at 32K.

For Phase 6, this means the improvement target is not subtle. A successful Hybrid Polar-KIVI result must first avoid the all-context retrieval collapse seen in both transformers QuantizedCache backends. If hybrid main still fails at 4K NIAH, the likely issue is the residual/quantization protocol itself rather than bit allocation alone.

Phase 4 passes data completeness for the fallback baseline: quality.csv contains fp16/hqq4/hqq2 NIAH zh/en and cached-PPL 16K/32K rows, perf.csv contains HF 16K/32K rows for all three configurations, and the quanto4 diagnostic failure is preserved for traceability.
