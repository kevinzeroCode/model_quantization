#!/usr/bin/env python3
"""LongBench v1 子集評測(zh 3 + en 3 + code 2)。prompt 模板與 max_gen 讀
third_party/LongBench 的 config;context 依 token 預算做中段截斷(LongBench 官方做法);
計分用官方 metrics.py。backend=openai 或 hf(同 run_niah)。"""
import argparse, importlib.util, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests

from reslog import log_quality, meta_args

SUBSETS = {  # subset -> metrics.py 內的函式名
    "multifieldqa_zh": "qa_f1_zh_score", "dureader": "rouge_zh_score",
    "passage_retrieval_zh": "retrieval_zh_score",
    "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score",
    "passage_retrieval_en": "retrieval_score",
    "lcc": "code_sim_score", "repobench-p": "code_sim_score",
}


def load_repo(repo):
    spec = importlib.util.spec_from_file_location("lb_metrics",
                                                  os.path.join(repo, "metrics.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    p = json.load(open(os.path.join(repo, "config/dataset2prompt.json"), encoding="utf-8"))
    g = json.load(open(os.path.join(repo, "config/dataset2maxlen.json"), encoding="utf-8"))
    return m, p, g


def main():
    ap = meta_args(argparse.ArgumentParser())
    ap.add_argument("--backend", choices=["openai", "hf"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--hf-model", default="models/Qwen2.5-7B-Instruct-yarn64k")
    ap.add_argument("--kv", default="fp16")
    ap.add_argument("--repo", default="third_party/LongBench/LongBench")
    ap.add_argument("--budget", type=int, default=31500)  # context token 預算
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    a = ap.parse_args()
    metrics, d2p, d2g = load_repo(a.repo)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    if a.backend == "openai":
        served = requests.get(f"{a.base_url}/v1/models", timeout=30).json()["data"][0]["id"]
        model_id = served

        def gen(prompt, max_new):
            r = requests.post(f"{a.base_url}/v1/chat/completions", timeout=1800, json={
                "model": served, "temperature": 0, "max_tokens": max_new,
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    else:
        import torch
        from transformers import AutoModelForCausalLM

        from cachelib import make_cache
        model_id = a.hf_model
        model = AutoModelForCausalLM.from_pretrained(
            a.hf_model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")
        htok = AutoTokenizer.from_pretrained(a.hf_model)

        def gen(prompt, max_new):
            ids = htok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True,
                                           return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                     past_key_values=make_cache(a.kv))
            return htok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    os.makedirs("results/raw", exist_ok=True)
    for sub in a.subsets:
        try:
            ds = load_dataset("THUDM/LongBench", sub, split="test", trust_remote_code=True)
        except Exception:
            ds = load_dataset("zai-org/LongBench", sub, split="test", trust_remote_code=True)
        fn = getattr(metrics, SUBSETS[sub])
        tpl, max_new = d2p[sub], int(d2g[sub])
        raw_path = f"results/raw/longbench_{a.run_id}_{sub}.jsonl"
        scores = []
        with open(raw_path, "w", encoding="utf-8") as raw:
            for i, x in enumerate(ds):
                if i >= a.limit:
                    break
                ctx = x["context"]
                ids = tok(ctx, add_special_tokens=False).input_ids
                if len(ids) > a.budget:  # 官方式中段截斷
                    half = a.budget // 2
                    ctx = tok.decode(ids[:half]) + tok.decode(ids[-half:])
                prompt = tpl.format(context=ctx, input=x.get("input", ""))
                try:
                    pred = gen(prompt, max_new)
                    s = max((fn(pred, gt, all_classes=x.get("all_classes"))
                             if "all_classes" in fn.__code__.co_varnames
                             else fn(pred, gt)) for gt in x["answers"])
                    status = "OK"
                except Exception as e:  # noqa: BLE001
                    pred, s, status = f"ERROR: {e}", 0.0, "FAIL"
                scores.append(s)
                raw.write(json.dumps({"i": i, "score": s, "status": status,
                                      "pred": str(pred)[:300]}, ensure_ascii=False) + "\n")
        val = round(sum(scores) / max(len(scores), 1), 4)
        print(f"{sub}: {val} (n={len(scores)})")
        log_quality({"run_id": a.run_id, "phase": a.phase, "host": a.host,
                     "runtime": a.runtime, "model_id": model_id,
                     "weight_quant": a.weight_quant, "kv_quant": a.kv_quant,
                     "rope_cfg": a.rope_cfg, "task": "longbench", "subset": sub,
                     "ctx_len": a.budget, "extra": "", "metric": SUBSETS[sub],
                     "value": val, "n": len(scores), "raw_path": raw_path,
                     "notes": a.notes})


if __name__ == "__main__":
    main()
