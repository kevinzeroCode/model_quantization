#!/usr/bin/env python3
"""Small OpenAI-compatible serving benchmark for non-vLLM servers.

The vLLM benchmark client is the preferred path for vLLM servers. This helper is
for fallback servers such as LMDeploy when the vLLM client request schema is not
accepted by the server. It records one perf.csv row per ctx/concurrency setting.
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests
from transformers import AutoTokenizer

from reslog import log_perf, meta_args


def apply_chat_len(tok, content):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        return_tensors=None,
    )
    return len(ids)


def make_prompt(tok, ctx_len, index):
    empty_len = apply_chat_len(tok, "")
    suffix = (
        "\n\nTask: Output the word BENCHMARK separated by spaces for at least "
        "512 words. Do not explain or stop early."
    )
    suffix_ids = tok.encode(suffix, add_special_tokens=False)
    target_content = max(1, ctx_len - empty_len - len(suffix_ids))
    unit = (
        f"Record {index:04d}: long-context serving benchmark filler. "
        "Return exactly the requested short answer after reading all records. "
    )
    text = unit * max(1, (target_content // 18) + 32)
    ids = tok.encode(text, add_special_tokens=False)
    content = tok.decode(ids[:target_content], skip_special_tokens=True) + suffix
    actual = apply_chat_len(tok, content)
    while actual > ctx_len and len(content) > 32:
        keep = max(
            1,
            len(tok.encode(content, add_special_tokens=False)) - (actual - ctx_len) - 8,
        )
        content = tok.decode(
            tok.encode(content, add_special_tokens=False)[:keep],
            skip_special_tokens=True,
        )
        if suffix not in content:
            content = content + suffix
        actual = apply_chat_len(tok, content)
    return content, actual


def parse_stream_response(resp, tok, t0):
    first = None
    chunks = []
    finish_reason = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        data = json.loads(payload)
        choice = data.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        piece = delta.get("content") or choice.get("text") or ""
        if piece:
            if first is None:
                first = time.perf_counter()
            chunks.append(piece)
    t_end = time.perf_counter()
    text = "".join(chunks)
    out_tok = len(tok.encode(text, add_special_tokens=False))
    ttft_ms = ((first or t_end) - t0) * 1000
    if out_tok > 1:
        tpot_ms = (t_end - (first or t_end)) * 1000 / (out_tok - 1)
    else:
        tpot_ms = 0.0
    return {
        "status": "OK",
        "completion_tokens": out_tok,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "latency_ms": (t_end - t0) * 1000,
        "finish_reason": finish_reason,
        "response_preview": text[:160],
    }


def run_one(base_url, model_id, prompt, prompt_tokens, tok, max_tokens, timeout, index):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "seed": 42 + index,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            return {
                "status": "OOM" if "out of memory" in resp.text.lower() else "FAIL",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "ttft_ms": 0.0,
                "tpot_ms": 0.0,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
            }
        row = parse_stream_response(resp, tok, t0)
        row["prompt_tokens"] = prompt_tokens
        return row
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        return {
            "status": "OOM" if "out of memory" in text.lower() else "FAIL",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "error": text[:500],
        }


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    ap = meta_args(argparse.ArgumentParser())
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--num-prompts", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--raw-path", default="")
    ap.add_argument("--eff-kv-bits", default="")
    ap.add_argument("--weights-gb", default="")
    ap.add_argument("--kv-pool-tokens", default="")
    ap.add_argument("--vram-peak-gb", default="")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = [make_prompt(tok, a.ctx, i) for i in range(a.num_prompts)]
    raw_path = a.raw_path or (
        f"results/raw/openai_serve_{a.run_id}_{a.ctx}_c{a.batch}.jsonl"
    )
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)

    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=a.batch) as pool:
        futs = [
            pool.submit(
                run_one,
                a.base_url,
                a.model_id,
                prompt,
                prompt_tokens,
                tok,
                a.max_tokens,
                a.timeout,
                i,
            )
            for i, (prompt, prompt_tokens) in enumerate(prompts)
        ]
        for fut in as_completed(futs):
            rows.append(fut.result())
    wall_s = max(time.perf_counter() - started, 1e-9)

    with open(raw_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok_rows = [r for r in rows if r["status"] == "OK"]
    statuses = {r["status"] for r in rows}
    status = "OK" if statuses == {"OK"} else ("OOM" if "OOM" in statuses else "FAIL")
    ttfts = [r["ttft_ms"] for r in ok_rows]
    tpots = [r["tpot_ms"] for r in ok_rows if r["completion_tokens"] > 1]
    out_tok = sum(r["completion_tokens"] for r in ok_rows)
    in_tok = sum(r["prompt_tokens"] for r in rows)
    log_perf({
        "run_id": a.run_id,
        "phase": a.phase,
        "host": a.host,
        "runtime": a.runtime,
        "model_id": a.model_id,
        "weight_quant": a.weight_quant,
        "kv_quant": a.kv_quant,
        "rope_cfg": a.rope_cfg,
        "ctx_len": a.ctx,
        "batch": a.batch,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "n_trials": len(rows),
        "ttft_ms_mean": round(mean(ttfts), 1),
        "ttft_ms_std": round(stdev(ttfts), 1),
        "tpot_ms_mean": round(mean(tpots), 2),
        "tpot_ms_std": round(stdev(tpots), 2),
        "gen_tps": round(out_tok / wall_s, 2),
        "weights_gb": a.weights_gb,
        "kv_pool_tokens": a.kv_pool_tokens,
        "vram_peak_gb": a.vram_peak_gb,
        "eff_kv_bits": a.eff_kv_bits,
        "status": status,
        "notes": f"{a.notes} src={raw_path}".strip(),
    })
    print(
        f"ctx={a.ctx} c={a.batch} status={status} "
        f"ok={len(ok_rows)}/{len(rows)} out_tok={out_tok} gen_tps={out_tok / wall_s:.2f}"
    )


if __name__ == "__main__":
    main()
