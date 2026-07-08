#!/usr/bin/env python3
"""vLLM runtime 的 teacher-forced PPL(bf16/AWQ/GPTQ 用同一條路,免裝 autoawq/gptqmodel)。
約定:WikiText-2 test 串接後切 4096-token 不重疊 chunk,取前 40 chunk,經 prompt_logprobs
拿每個 token 的 logprob。此約定與 S11 standard 模式不同 —— 兩者數字不得互比,
只在各自腳本內部跨配置比較。"""
import argparse, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reslog import log_quality, meta_args


def main():
    ap = meta_args(argparse.ArgumentParser())
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    ap.add_argument("--batch-size", type=int, default=1)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text).input_ids
    chunks = [ids[i:i + a.chunk] for i in range(0, len(ids) - a.chunk, a.chunk)]
    chunks = chunks[:a.max_chunks]

    llm = LLM(model=a.model, max_model_len=a.chunk + 16,
              gpu_memory_utilization=a.gpu_memory_utilization,
              max_num_seqs=a.batch_size,
              max_num_batched_tokens=a.chunk * a.batch_size)
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)
    try:
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=c) for c in chunks]
    except ImportError:
        prompts = [{"prompt_token_ids": c} for c in chunks]
    outs = []
    for i in range(0, len(prompts), a.batch_size):
        batch = prompts[i:i + a.batch_size]
        print(f"ppl chunk {i + 1}-{i + len(batch)}/{len(prompts)}", flush=True)
        outs.extend(llm.generate(batch, sp, use_tqdm=False))

    nll, cnt = 0.0, 0
    for o in outs:
        for lp in o.prompt_logprobs[1:]:  # 第 0 個位置無 logprob
            nll -= next(iter(lp.values())).logprob
            cnt += 1
    val = math.exp(nll / cnt)
    print(f"PPL(vllm) {a.model} = {val:.4f} over {cnt} tokens")
    log_quality({"run_id": a.run_id, "phase": a.phase, "host": a.host, "runtime": a.runtime,
                 "model_id": a.model, "weight_quant": a.weight_quant,
                 "kv_quant": a.kv_quant, "rope_cfg": a.rope_cfg, "task": "ppl_vllm",
                 "subset": "wikitext2", "ctx_len": a.chunk, "extra": "",
                 "metric": "ppl", "value": round(val, 4), "n": cnt,
                 "raw_path": "", "notes": a.notes})


if __name__ == "__main__":
    main()
