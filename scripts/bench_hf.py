#!/usr/bin/env python3
"""HF runtime 的 TTFT/TPOT/VRAM 量測(batch=1;研究路線內部對照用,不與 vLLM 同表)。
方法:隨機 token ids 做 chunked prefill(計 TTFT=prefill+第一步 decode),再逐 token decode
128 步量 TPOT;1 次 warmup 後取 n=trials 的 mean/std;VRAM 用 torch.cuda.max_memory_allocated。"""
import argparse, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from cachelib import EFF_BITS, fwd, make_cache
from reslog import log_perf, meta_args


@torch.no_grad()
def once(model, ids, kv, out_tokens, prefill_chunk):
    torch.cuda.reset_peak_memory_stats()
    cache = make_cache(kv, model.config)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for b in range(0, ids.shape[1], prefill_chunk):
        out = fwd(model, ids[:, b:min(b + prefill_chunk, ids.shape[1])], cache)
    nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000
    steps = []
    for _ in range(out_tokens - 1):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = fwd(model, nxt, cache)
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        steps.append((time.perf_counter() - t1) * 1000)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del cache
    torch.cuda.empty_cache()
    return ttft, statistics.mean(steps), peak


def main():
    ap = meta_args(argparse.ArgumentParser())
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--kv", default="fp16")
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda",
                                                 attn_implementation="sdpa")
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(42)
    for ctx in a.ctx:
        ids = torch.randint(0, vocab - 1000, (1, ctx), generator=g).to("cuda")
        try:
            once(model, ids, a.kv, 16, a.prefill_chunk)  # warmup(丟棄)
            runs = [once(model, ids, a.kv, a.out_tokens, a.prefill_chunk)
                    for _ in range(a.trials)]
            ttfts, tpots, peaks = zip(*runs)
            row = {"ttft_ms_mean": round(statistics.mean(ttfts), 1),
                   "ttft_ms_std": round(statistics.stdev(ttfts), 1) if a.trials > 1 else 0,
                   "tpot_ms_mean": round(statistics.mean(tpots), 2),
                   "tpot_ms_std": round(statistics.stdev(tpots), 2) if a.trials > 1 else 0,
                   "gen_tps": round(1000 / statistics.mean(tpots), 1),
                   "vram_peak_gb": round(max(peaks), 2), "status": "OK"}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {"status": "OOM"}
        eff = EFF_BITS.get(a.kv)
        if eff is None and a.kv.startswith("hybrid:"):
            eff = make_cache(a.kv).cfg.eff_bits()
        print(f"ctx={ctx} kv={a.kv} -> {row}")
        if not a.no_log:
            log_perf({**row, "run_id": a.run_id, "phase": a.phase, "host": a.host,
                      "runtime": a.runtime, "model_id": a.model,
                      "weight_quant": a.weight_quant, "kv_quant": a.kv_quant,
                      "rope_cfg": a.rope_cfg, "ctx_len": ctx, "batch": 1, "in_tok": ctx,
                      "out_tok": a.out_tokens, "n_trials": a.trials,
                      "eff_kv_bits": eff, "notes": a.notes})


if __name__ == "__main__":
    main()
