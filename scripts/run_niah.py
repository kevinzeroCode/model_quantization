#!/usr/bin/env python3
"""對一個配置跑凍結 NIAH 集並計分(判準:needle code 是否出現在回覆,不分大小寫)。
backend=openai:打 vLLM/LMDeploy 的 OpenAI 相容端點。backend=hf:行程內生成,支援
quantized/hybrid cache。結果寫 results/quality.csv(每個 ctx×depth 一列)。"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests

from reslog import log_quality, meta_args


def main():
    ap = meta_args(argparse.ArgumentParser())
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--backend", choices=["openai", "hf"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--hf-model", default="models/Qwen2.5-7B-Instruct-yarn64k")
    ap.add_argument("--kv", default="fp16")  # hf 專用:fp16|quanto4|quanto2|hybrid:<cfg>
    ap.add_argument("--max-ctx", type=int, default=10**9)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    rows = [r for r in rows if r["ctx_tokens"] <= a.max_ctx]
    assert rows, "no prompts under --max-ctx"

    if a.backend == "openai":
        served = requests.get(f"{a.base_url}/v1/models", timeout=30).json()["data"][0]["id"]
        model_id = served

        def gen(prompt):
            r = requests.post(f"{a.base_url}/v1/chat/completions", timeout=1800, json={
                "model": served, "temperature": 0, "max_tokens": a.max_new, "seed": 42,
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from cachelib import make_cache
        model_id = a.hf_model
        tok = AutoTokenizer.from_pretrained(a.hf_model)
        model = AutoModelForCausalLM.from_pretrained(
            a.hf_model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")

        def gen(prompt):
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True,
                                          return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                                     past_key_values=make_cache(a.kv))
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    os.makedirs("results/raw", exist_ok=True)
    tag = os.path.basename(a.prompts).replace(".jsonl", "")
    raw_path = f"results/raw/niah_{a.run_id}_{tag}.jsonl"
    cells: dict = {}
    with open(raw_path, "w", encoding="utf-8") as raw:
        for i, r in enumerate(rows):
            t0 = time.time()
            try:
                resp = gen(r["prompt"])
                ok, status = r["needle"].lower() in resp.lower(), "OK"
            except Exception as e:  # noqa: BLE001
                resp, ok, status = f"ERROR: {e}", False, "FAIL"
            cells.setdefault((r["ctx_tokens"], r["depth"]), []).append(ok)
            raw.write(json.dumps({"id": r["id"], "ok": ok, "status": status,
                                  "latency_s": round(time.time() - t0, 2),
                                  "resp": str(resp)[:300]}, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(rows)}] {r['id']} -> {'HIT' if ok else 'MISS'} ({status})")

    for (ctx, depth), oks in sorted(cells.items()):
        log_quality({"run_id": a.run_id, "phase": a.phase, "host": a.host,
                     "runtime": a.runtime, "model_id": model_id,
                     "weight_quant": a.weight_quant, "kv_quant": a.kv_quant,
                     "rope_cfg": a.rope_cfg, "task": "niah", "subset": tag,
                     "ctx_len": ctx, "extra": f"depth={depth}", "metric": "acc",
                     "value": round(sum(oks) / len(oks), 4), "n": len(oks),
                     "raw_path": raw_path, "notes": a.notes})
    flat = [o for v in cells.values() for o in v]
    print(f"TOTAL acc = {sum(flat)}/{len(flat)} = {sum(flat) / len(flat):.3f}")


if __name__ == "__main__":
    main()
