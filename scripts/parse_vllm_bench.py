#!/usr/bin/env python3
"""把 `vllm bench serve --save-result` 的 JSON 轉成 perf.csv 一列。
weights_gb / kv_pool_tokens 無法從 bench JSON 取得 —— 用 --weights-gb / --kv-pool-tokens
手動帶入(值從 serve log grep,見 Phase 1.3),沒帶就留空。"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reslog import log_perf, meta_args


def main():
    ap = meta_args(argparse.ArgumentParser())
    ap.add_argument("--json", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--ctx-len", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)  # = max_concurrency
    ap.add_argument("--eff-kv-bits", default="")
    ap.add_argument("--weights-gb", default="")
    ap.add_argument("--kv-pool-tokens", default="")
    a = ap.parse_args()
    d = json.load(open(a.json, encoding="utf-8"))
    g = lambda *keys: next((d[k] for k in keys if k in d), "")
    log_perf({"run_id": a.run_id, "phase": a.phase, "host": a.host, "runtime": a.runtime,
              "model_id": a.model_id, "weight_quant": a.weight_quant,
              "kv_quant": a.kv_quant, "rope_cfg": a.rope_cfg, "ctx_len": a.ctx_len,
              "batch": a.batch,
              "in_tok": g("total_input_tokens"), "out_tok": g("total_output_tokens"),
              "n_trials": g("completed", "num_prompts"),
              "ttft_ms_mean": g("mean_ttft_ms"), "ttft_ms_std": g("std_ttft_ms"),
              "tpot_ms_mean": g("mean_tpot_ms"), "tpot_ms_std": g("std_tpot_ms"),
              "gen_tps": g("output_throughput"),
              "weights_gb": a.weights_gb, "kv_pool_tokens": a.kv_pool_tokens,
              "eff_kv_bits": a.eff_kv_bits, "status": "OK",
              "notes": a.notes + f" src={os.path.basename(a.json)}"})
    print(f"logged {a.json} -> results/perf.csv")


if __name__ == "__main__":
    main()
