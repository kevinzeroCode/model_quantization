# Phase 6 Report - Hybrid Polar-KIVI KV cache

Date: 2026-07-14
Host: RTX 4090 24GB
Runtime: Hugging Face transformers 4.57.6 / torch 2.8.0+cu128
Model: models/Qwen2.5-7B-Instruct-yarn64k

## Scope

Phase 6 tested the independent Hybrid Polar-KIVI cache implementation. Keys use simplified polar quantization, values use per-token grouped quantization, and the residual window remains 128 tokens. The Phase 6 matrix used the same HF protocol as Phase 4: zh/en NIAH up to 32K and cached-PPL at 16K/32K on `wikitext-103`.

`pytest scripts/test_hybrid_cache.py -v` passed 7/7 before the 7B runs.

## Configurations

Qwen2.5-7B has 28 decoder layers. `skip01` keeps layers 0 and 1 in full precision, so its budget is reported as a layer-weighted effective bit estimate instead of the raw hybrid cache budget.

| run_id | config | theta bits | r bits | value bits | rotation | skip layers | base eff bits | weighted eff bits |
|---|---|---:|---:|---:|---|---:|---:|---:|
| p6-main | main | 4 | 4 | 2 | true | none | 3.31 | 3.31 |
| p6-theta3 | theta3 | 3 | 4 | 2 | true | none | 3.06 | 3.06 |
| p6-norot | norot | 4 | 4 | 2 | false | none | 3.31 | 3.31 |
| p6-skip01 | skip01 | 4 | 4 | 2 | true | 0, 1 | 3.31 | 4.22 |

## NIAH

Each language has 45 examples: 3 context lengths x 5 depths x 3 seeds.

| run_id | weighted eff bits | zh 4K | zh 16K | zh 32K | zh total | en 4K | en 16K | en 32K | en total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p6-main | 3.31 | 0/15 | 0/15 | 0/15 | 0/45 | 0/15 | 0/15 | 0/15 | 0/45 |
| p6-theta3 | 3.06 | 0/15 | 0/15 | 0/15 | 0/45 | 0/15 | 0/15 | 0/15 | 0/45 |
| p6-norot | 3.31 | 0/15 | 0/15 | 0/15 | 0/45 | 0/15 | 0/15 | 0/15 | 0/45 |
| p6-skip01 | 4.22 | 15/15 | 15/15 | 15/15 | 45/45 | 15/15 | 14/15 | 15/15 | 44/45 |

The main proposal does not pass the retrieval gate: it fails from the first 4K cells in both languages. Lowering theta to 3 bits and disabling rotation also fail. Keeping only the first two layers full precision changes the result from 0/90 to 89/90, so the failure is strongly localized to early-layer KV sensitivity rather than the whole hybrid scheme being uniformly unusable.

## Cached-PPL

PPL rows use `wikitext-103`, one document, and chunked prefill. Phase 4 baselines are included for budget context.

| run_id | kv_quant | weighted eff bits | 16K PPL | 32K PPL |
|---|---:|---:|---:|---:|
| p4-fp16 | fp16 | 16.00 | 5.4877 | 8.0138 |
| p4-hqq4 | int4_hqq | 4.50 | 8485.7470 | 26420.3876 |
| p4-hqq2 | int2_hqq | 2.50 | 39672.8659 | 188153.7391 |
| p6-main | hybrid_pk | 3.31 | 3613.8965 | 8218.2561 |
| p6-theta3 | hybrid_pk | 3.06 | 3485.3272 | 6465.5975 |
| p6-norot | hybrid_pk | 3.31 | 3138.1609 | 2679.7439 |
| p6-skip01 | hybrid_pk | 4.22 | 5.9968 | 8.7105 |

The no-skip hybrids have much better PPL than the transformers HQQ baselines, but still fail NIAH completely. `norot` gives the best no-skip PPL, suggesting the random rotation hurts language-modeling quality, but rotation is not the retrieval failure's root cause because `norot` still scores 0/90.

`skip01` is the first useful Phase 6 configuration: at an estimated 4.22 effective KV bits, it is below the HQQ int4 budget of 4.5 bits, recovers 89/90 NIAH, and has PPL close to fp16 at both 16K and 32K.

## Answer to Phase 6 Criteria

The main 3.31-bit proposal is only a partial success. It significantly improves cached-PPL over the Phase 4 low-bit HQQ baselines, but it does not satisfy the retrieval-quality requirement because zh/en NIAH are 0/90.

The `skip01` ablation gives a constructive positive result. It shows that preserving the first two layers' KV cache is enough to restore both retrieval and PPL under a sub-int4 weighted budget. The next research step should turn this into a principled mixed-precision policy: early-layer fp16 fallback, learned layer allocation, or outlier-channel fallback, then re-run Phase 7 tasks.

## Acceptance

Phase 6 passes data completeness: `results/quality.csv` has 4 hybrid configs x (zh NIAH + en NIAH + 16K PPL + 32K PPL) = 128 rows, raw NIAH outputs are saved under `results/raw/`, and the T6 quality-budget conclusion is explicit: main is partial, skip01 is the successful mixed-precision variant.
