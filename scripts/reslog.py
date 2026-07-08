"""統一的結果落盤:所有腳本只透過這兩個函式寫 CSV。"""
import csv, os

PCOLS = ["run_id","phase","host","runtime","model_id","weight_quant","kv_quant","rope_cfg",
         "ctx_len","batch","in_tok","out_tok","n_trials","ttft_ms_mean","ttft_ms_std",
         "tpot_ms_mean","tpot_ms_std","gen_tps","weights_gb","kv_pool_tokens","vram_peak_gb",
         "eff_kv_bits","status","notes"]
QCOLS = ["run_id","phase","host","runtime","model_id","weight_quant","kv_quant","rope_cfg",
         "task","subset","ctx_len","extra","metric","value","n","raw_path","notes"]

def _append(path, cols, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})

def log_perf(row, path="results/perf.csv"):
    _append(path, PCOLS, row)

def log_quality(row, path="results/quality.csv"):
    _append(path, QCOLS, row)

def meta_args(ap):
    """各腳本共用的 bookkeeping 參數。"""
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--host", default="4090")
    ap.add_argument("--runtime", required=True, choices=["vllm", "hf", "lmdeploy"])
    ap.add_argument("--weight-quant", required=True)
    ap.add_argument("--kv-quant", required=True)
    ap.add_argument("--rope-cfg", default="yarn4")
    ap.add_argument("--notes", default="")
    return ap
