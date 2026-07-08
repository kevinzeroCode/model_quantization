#!/usr/bin/env python3
"""PPL 評測(HF runtime)。
--mode standard:WikiText-2 滑動視窗 PPL(window 4096 / stride 2048;只量 weight 影響,
                 KV cache 路徑不參與 —— 不能用來評 KV 量化)。
--mode cached  :PG-19 長文,先把前段 chunked prefill 進 cache,再逐 token 計分最後
                 score_tail 個 token —— KV cache 路徑全程參與,是量 KV 量化品質的模式。"""
import argparse, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from cachelib import fwd, make_cache
from reslog import log_quality, meta_args


@torch.no_grad()
def ppl_standard(model, tok, window=4096, stride=2048):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    seq_len = ids.size(1)
    nll_sum, prev_end = 0.0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + window, seq_len)
        trg = end - prev_end
        chunk = ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-trg] = -100
        out = model(chunk, labels=labels)
        nll_sum += out.loss.float().item() * trg
        prev_end = end
        if end == seq_len:
            break
    return math.exp(nll_sum / prev_end)


@torch.no_grad()
def ppl_cached(model, tok, kv, ctx, score_tail=1024, n_docs=3, chunk=2048):
    from datasets import load_dataset
    try:
        docs = load_dataset("deepmind/pg19", split="test", trust_remote_code=True)["text"]
        src = "pg19"
    except Exception:
        docs = ["\n".join(load_dataset("wikitext", "wikitext-103-raw-v1",
                                       split="test")["text"])]
        src = "wikitext103"
    ppls = []
    for d in docs:
        ids = tok(d, return_tensors="pt").input_ids
        if ids.size(1) < ctx:
            continue
        ids = ids[:, :ctx].to(model.device)
        cache = make_cache(kv)
        S = ctx - score_tail
        out = None
        for b in range(0, S, chunk):
            out = fwd(model, ids[:, b:min(b + chunk, S)], cache)
        nll = 0.0
        for t in range(S, ctx):  # out.logits[:, -1] 預測位置 t 的 token
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            nll += -logp[ids[0, t]].item()
            if t < ctx - 1:
                out = fwd(model, ids[:, t:t + 1], cache)
        ppls.append(math.exp(nll / score_tail))
        del cache
        torch.cuda.empty_cache()
        if len(ppls) >= n_docs:
            break
    assert ppls, f"no doc >= {ctx} tokens in {src}"
    return sum(ppls) / len(ppls), src, len(ppls)


def main():
    ap = meta_args(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["standard", "cached"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv", default="fp16")
    ap.add_argument("--ctx", type=int, default=32768)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda",
                                                 attn_implementation="sdpa")
    if a.mode == "standard":
        val, subset, n, ctx = ppl_standard(model, tok), "wikitext2", 1, 4096
    else:
        (val, subset, n), ctx = ppl_cached(model, tok, a.kv, a.ctx), a.ctx
    print(f"PPL[{a.mode}] {a.model} kv={a.kv} ctx={ctx} = {val:.4f}")
    log_quality({"run_id": a.run_id, "phase": a.phase, "host": a.host,
                 "runtime": a.runtime, "model_id": a.model,
                 "weight_quant": a.weight_quant, "kv_quant": a.kv_quant,
                 "rope_cfg": a.rope_cfg, "task": f"ppl_{a.mode}", "subset": subset,
                 "ctx_len": ctx, "extra": "", "metric": "ppl", "value": round(val, 4),
                 "n": n, "raw_path": "", "notes": a.notes})


if __name__ == "__main__":
    main()
