# Phase 5 Report - TurboQuant probe and LMDeploy low-bit KV fallback

Date: 2026-07-14
Host: RTX 4090 24GB, NVIDIA driver 570.211.01
Model: models/Qwen2.5-7B-Instruct-yarn64k

## Scope

Phase 5 tested the deployment-oriented low-bit KV route. The preferred route was vLLM TurboQuant KV. The stable Phase 1/2/3 vLLM environment is vLLM 0.10.2 and does not expose TurboQuant KV cache dtypes. An isolated vLLM 0.25.0 probe environment does expose TurboQuant and int4/int8 KV flags, but it installs torch 2.11.0+cu130 and cannot initialize CUDA on the current driver. Evidence is saved in `env/probe_vllm_p5.json`:

| probe | result |
|---|---|
| vLLM version | 0.25.0 |
| torch | 2.11.0+cu130 |
| TurboQuant module | true |
| KV dtype help has TurboQuant | true |
| CUDA allocation | false, driver too old for cu130 |

Following the runbook route B, LMDeploy 0.14.0 was used as the fallback deployment runtime. LMDeploy speed numbers are only comparable with LMDeploy rows, not with vLLM or HF rows.

## LMDeploy Configuration

Both runs used `session_len=66560`, `cache_max_entry_count=0.7`, `max_batch_size=4`, and `max_prefill_token_num=4096`.

| run_id | quant_policy | kv_quant | block size | max block count | KV pool tokens | effective KV bits |
|---|---:|---:|---:|---:|---:|---:|
| p5-lmd-int8 | 8 | int8_lmd | 1.805 MB | 3,397 | 217,408 | 8 |
| p5-lmd-int4 | 4 | int4_lmd | 0.930 MB | 6,595 | 422,080 | 4 |

The first int8 server attempt with a more aggressive cache setting OOMed during warm-up. The conservative setting above started cleanly and was reused for int4.

## NIAH Quality

Frozen `niah_zh` prompts were run from 4,096 to 63,488 context tokens. Each context has 15 examples: 5 depths x 3 seeds.

| run_id | kv_quant | 4K | 16K | 32K | 63K | total |
|---|---:|---:|---:|---:|---:|---:|
| p5-lmd-int8 | int8_lmd | 7/15 | 1/15 | 1/15 | 0/15 | 9/60 |
| p5-lmd-int4 | int4_lmd | 0/15 | 0/15 | 0/15 | 0/15 | 0/60 |

Raw status counts: int8 had 59 OK responses and 1 FAIL response; int4 had 60 OK responses. The int4 result is a quality collapse from the first 4K cell, not a server stability failure.

## Serving Reference

The vLLM benchmark client sent request fields that LMDeploy rejected with HTTP 422, so `scripts/bench_openai_serve.py` was added for OpenAI-compatible streaming measurement. It records actual output tokens, TTFT, TPOT, and throughput directly to `results/perf.csv`.

| run_id | kv_quant | ctx | batch | trials | out_tok | TTFT ms | TPOT ms | gen_tps | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p5-lmd-int8 | int8_lmd | 32,768 | 1 | 4 | 135 | 4,282.1 | 17.37 | 6.97 | OK |
| p5-lmd-int8 | int8_lmd | 32,768 | 4 | 8 | 384 | 9,063.6 | 167.30 | 10.68 | OK |
| p5-lmd-int8 | int8_lmd | 63,488 | 1 | 3 | 189 | 10,768.5 | 18.17 | 5.30 | OK |
| p5-lmd-int8 | int8_lmd | 63,488 | 3 | 6 | 379 | 18,020.6 | 241.54 | 5.70 | OK |
| p5-lmd-int4 | int4_lmd | 32,768 | 1 | 4 | 256 | 4,263.5 | 16.58 | 12.06 | OK |
| p5-lmd-int4 | int4_lmd | 32,768 | 4 | 8 | 510 | 8,563.7 | 147.78 | 14.21 | OK |
| p5-lmd-int4 | int4_lmd | 63,488 | 1 | 3 | 187 | 10,750.0 | 17.39 | 5.28 | OK |
| p5-lmd-int4 | int4_lmd | 63,488 | 4 | 8 | 509 | 20,920.8 | 361.11 | 5.83 | OK |

The int4 KV pool was large enough to run 63K/c4 successfully; int8 was measured at 63K/c3 because the conservative int8 KV pool is 217,408 tokens.

## Conclusion

Phase 5 produces a deployment fallback result, not a usable low-bit KV recommendation. LMDeploy int8 and int4 both increase KV capacity relative to fp16 KV, and int4 roughly doubles the int8 KV pool under the same cache fraction. The quality tradeoff is unacceptable for the frozen Chinese retrieval task: int8 reaches only 9/60 and int4 reaches 0/60.

The result is consistent with Phase 3 and Phase 4: low-bit KV capacity gains must be validated with retrieval quality, not treated as usable context length by memory math alone. For the next phase, a hybrid method must first clear the basic 4K NIAH retrieval gate before longer-context capacity improvements matter.

## Acceptance

Phase 5 is complete by route B. `results/quality.csv` contains 40 Phase 5 NIAH rows, `results/perf.csv` contains 8 formal LMDeploy serving rows, raw artifacts are saved under `results/raw/`, and the probe/freeze artifacts are saved under `env/`.
