# Phase 8 Plan - DGX Spark Capacity Boundary

Date: 2026-07-15
Status: planning draft before Spark execution
Host target: DGX Spark / aarch64 container

## What This Phase Can Prove

Phase 8 answers RQ4: how far a single Spark box can push long-context serving before the memory or runtime boundary fails.

The only claims we want from this phase are:

- the largest context that each config can actually serve
- whether fp8 KV pushes the boundary farther than fp16 KV
- whether 1M is reachable for any config

This is a capacity experiment, not a throughput contest. TPOT on Spark is expected to be slower than the 4090 runs in earlier phases.

## Hard Limits

- Spark is aarch64. Do not reuse x86 wheels or the local `venv-*` directories from the 4090 machine.
- The container should come from the NVIDIA DGX Spark playbooks.
- If the container or vLLM build does not support 1M / dual-chunk attention, stop at 131072 and record that as the upper bound.
- If a configuration OOMs or fails at a given step, record that step as the boundary and do not keep retrying larger lengths first.
- The only new prompt file allowed in this phase is `prompts/niah_zh_128k.jsonl`.
- The phase is complete only if host=spark rows are written to `results/perf.csv` and the 128K zh NIAH sample is recorded for the configs that can reach it.

## Files To Bring Onto Spark

Copy these directories as-is:

- `scripts/`
- `prompts/`

If the Spark side needs local model weights, mirror only the model directories required by the three configs below.

## Configurations

Run these three serving configs:

- `bf16+fp16KV`
- `bf16+fp8KV`
- `awq+fp8KV`

Use the Qwen2.5 7B YaRN family as the runbook specifies:

- `Qwen/Qwen2.5-7B-Instruct` for the bf16 configs
- `Qwen/Qwen2.5-7B-Instruct-AWQ` or the matching AWQ checkpoint for the awq config
- `Qwen/Qwen2.5-7B-Instruct-1M` only if the container and vLLM help output confirm the 1M path is supported

## Execution Order

### 1. Probe the environment

Run the phase-0 probe inside the Spark container:

```bash
python scripts/probe_env.py --venv vllm
```

Save the JSON output under `env/` on Spark if the container has persistent storage.

### 2. Capacity ladder

For each of the three configs, try the context ladder in this order:

```text
131072 -> 262144 -> 524288 -> 1048576
```

At each step:

- serve the model
- run a single-request serving benchmark with input length = step - 2048 and output = 128
- record TTFT, decode tokens/s, and peak free memory
- stop at the first FAIL or OOM

Suggested benchmark command shape:

```bash
python scripts/bench_openai_serve.py   --run-id p8-<cfg>-<ctx>   --phase 8   --host spark   --runtime vllm   --weight-quant <weight_quant>   --kv-quant <kv_quant>   --rope-cfg yarn4   --base-url http://127.0.0.1:8000   --model-id <served-model-name>   --tokenizer Qwen/Qwen2.5-7B-Instruct   --ctx <step-minus-2048>   --batch 1   --num-prompts 3   --max-tokens 128   --eff-kv-bits <value>   --weights-gb <value>   --kv-pool-tokens <value>   --vram-peak-gb <value>
```

Use the serving logs and `nvidia-smi` to fill in the memory fields.

### 3. 128K zh NIAH sample

Generate the only additional prompt file allowed in this phase:

```bash
python scripts/gen_prompts.py --ctx 126976 --samples 2 --langs zh --suffix _128k
```

Then run NIAH only for the configs that can serve 128K:

```bash
python scripts/run_niah.py   --run-id p8-<cfg>   --phase 8   --host spark   --runtime vllm   --weight-quant <weight_quant>   --kv-quant <kv_quant>   --rope-cfg yarn4   --notes spark_128k   --prompts prompts/niah_zh_128k.jsonl   --backend openai   --base-url http://127.0.0.1:8000
```

## Expected Artifacts

When Phase 8 is finished on Spark, the repo should contain:

- `results/perf.csv` rows with `host=spark`
- `results/quality.csv` rows for the 128K zh NIAH sample
- `results/raw/` outputs for the capacity ladder and the 128K NIAH samples
- `reports/phase8.md` with the boundary summary

## Reporting Template

The report should answer these three questions:

- What is the maximum feasible context for each config?
- How far does fp8 KV move the boundary relative to fp16 KV?
- Does any config reach 1M, or does the run stop earlier at 131072 / 262144 / 524288?

If 1M does not work, that is still a valid result. Record the first failing step and the last successful step, then move on.
