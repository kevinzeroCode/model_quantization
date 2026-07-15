#!/usr/bin/env python3
"""彙整兩張 CSV 成論文用主表(markdown),輸出 reports/summary_tables.md。"""
import os

import pandas as pd

os.makedirs("reports", exist_ok=True)
out = ["# Summary Tables(自動生成,勿手改)\n"]

if os.path.exists("results/perf.csv"):
    p = pd.read_csv("results/perf.csv")
    out.append("## T1 速度/容量(vLLM runtime)\n")
    t = p[p.runtime == "vllm"].pivot_table(
        index=["weight_quant", "kv_quant"], columns=["ctx_len", "batch"],
        values="gen_tps", aggfunc="max")
    out.append(t.to_markdown() + "\n")
    out.append("## T1b HF runtime 參考(不得與 T1 互比)\n")
    hf = p[p.runtime == "hf"][["run_id", "kv_quant", "ctx_len", "ttft_ms_mean",
                               "tpot_ms_mean", "vram_peak_gb", "eff_kv_bits", "status"]]
    out.append(hf.to_markdown(index=False) + "\n")
    out.append("## T1c LMDeploy runtime 參考(不得與 T1/HF 互比)\n")
    lmd = p[p.runtime == "lmdeploy"][["run_id", "kv_quant", "ctx_len", "batch",
                                      "ttft_ms_mean", "tpot_ms_mean", "gen_tps",
                                      "kv_pool_tokens", "eff_kv_bits", "status"]]
    out.append(lmd.to_markdown(index=False) + "\n")

if os.path.exists("results/quality.csv"):
    q = pd.read_csv("results/quality.csv")
    niah = q[q.task == "niah"]
    if len(niah):
        out.append("## T2 NIAH 準確率(config × ctx,depth 取平均)\n")
        t = niah.pivot_table(index=["run_id", "runtime", "weight_quant",
                                      "kv_quant", "rope_cfg", "subset"],
                             columns="ctx_len", values="value", aggfunc="mean")
        out.append(t.round(3).to_markdown() + "\n")
    ppl = q[q.task.str.startswith("ppl")]
    if len(ppl):
        out.append("## T3 PPL\n")
        out.append(ppl[["run_id", "task", "subset", "ctx_len", "runtime", "weight_quant",
                        "kv_quant", "value", "n"]].to_markdown(index=False) + "\n")
    lb = q[q.task == "longbench"]
    if len(lb):
        out.append("## T4 LongBench(subset × config)\n")
        t = lb.pivot_table(index="subset",
                           columns=["runtime", "weight_quant", "kv_quant"],
                           values="value", aggfunc="mean")
        out.append(t.round(3).to_markdown() + "\n")

with open("reports/summary_tables.md", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote reports/summary_tables.md")
