# Phase 7 Report - Chinese / Code / RAG Scenario Evaluation

Date: 2026-07-15
Host: RTX 4090 24GB
Model: models/Qwen2.5-7B-Instruct-yarn64k

## Scope

Phase 7 answers RQ3: which task family is most sensitive to KV quantization, and whether the failure looks symmetric across Chinese and English.

Representative configurations:

| run_id | runtime | weight_quant | kv_quant | notes |
|---|---|---:|---:|---|
| p1-bf16 | vllm | bf16 | fp16 | Phase 1 baseline |
| p3-awq-fp8 | vllm | awq_w4 | fp8_e4m3 | Phase 3 best vLLM quantized-KV path |
| p7-int2-quanto | hf | bf16 | int2_quanto | Phase 7 HF low-bit run, `max-ctx=32768` |
| p7-main | hf | bf16 | hybrid_pk | Phase 7 HF main hybrid run, `max-ctx=32768` |

`p3-awq-fp8` got an additional Phase 7 `niah_code` fill-in run so the code retrieval gap is explicit in this phase. The HF runs only cover contexts up to 32K because that is the runbook limit for the low-bit transformer backends.

## Coverage

LongBench uses the frozen 50-example slices already established in the repo. NIAH rows are grouped by context length, 5 depths, and 3 seeds.

| run_id | zh NIAH | code NIAH | LongBench zh | LongBench code | LongBench RAG |
|---|---|---|---|---|---|
| p1-bf16 | 60/60 | 60/60 | 3/3 subsets | 2/2 subsets | 3/3 subsets |
| p3-awq-fp8 | 9/60 | 1/60 | 3/3 subsets | 1/1 subset | 3/3 subsets |
| p7-int2-quanto | 0/45 | 0/45 | 3/3 subsets | 2/2 subsets | 3/3 subsets |
| p7-main | 0/45 | 0/45 | 3/3 subsets | 2/2 subsets | 3/3 subsets |

## Scenario Means

The table below uses the bf16+fp16 baseline as the reference and averages the subsets inside each scenario family.

| scenario | p1-bf16 | p3-awq-fp8 | p7-int2-quanto | p7-main |
|---|---:|---:|---:|---:|
| Chinese LongBench mean | 0.5873 | 0.2254 | 0.0416 | 0.0364 |
| Code LongBench mean | 0.0730 | 0.1476 | 0.1831 | 0.1231 |
| RAG LongBench mean | 0.6794 | 0.0665 | 0.0136 | 0.0068 |

Relative change vs `p1-bf16`:

| scenario | p3-awq-fp8 | p7-int2-quanto | p7-main |
|---|---:|---:|---:|
| Chinese LongBench | -61.6% | -92.9% | -93.8% |
| Code LongBench | +102.2% | +150.8% | +68.6% |
| RAG LongBench | -90.2% | -98.0% | -99.0% |

The code LongBench metric is the odd one out: it improves under low-bit configs, but the exact code retrieval task does not. That makes the code similarity score useful as a generation heuristic, not as proof that long-context code retrieval is preserved.

## Retrieval Gate

Exact retrieval is the hard gate. NIAH shows the collapse directly.

| run_id | zh NIAH | code NIAH |
|---|---:|---:|
| p1-bf16 | 60/60 | 60/60 |
| p3-awq-fp8 | 9/60 | 1/60 |
| p7-int2-quanto | 0/45 | 0/45 |
| p7-main | 0/45 | 0/45 |

The HF low-bit paths fail immediately on both Chinese and code NIAH. `p3-awq-fp8` is slightly less bad at 4K Chinese NIAH, but it still collapses by 16K and remains unusable for code NIAH beyond the first 4K slice.

## Answer

The most sensitive scenario is RAG / multi-document retrieval. It is the first place where every low-bit configuration loses almost all utility: the Chinese and English passage-retrieval subsets both fall sharply, and the combined RAG mean drops by 90.2% to 99.0% relative to bf16+fp16.

Chinese is not protected by the low-bit path. `p3-awq-fp8` still has some 4K Chinese NIAH signal, but the HF low-bit routes lose all Chinese and code NIAH signal by 32K and below. English passage retrieval is usually at least as fragile as Chinese passage retrieval in these runs, so there is no consistent Chinese-vs-English advantage once KV quantization gets aggressive.

## Acceptance

Phase 7 is complete for the available matrix: the missing `p3-awq-fp8` code retrieval cell is filled, the two HF representative runs are present, `reports/summary_tables.md` has been regenerated, and the scenario conclusion is explicit:

- RAG / multi-document retrieval is the most brittle family.
- Exact retrieval collapses much earlier than code-sim LongBench would suggest.
- English is not materially more robust than Chinese under the low-bit KV paths.
