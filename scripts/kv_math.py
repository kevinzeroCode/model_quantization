#!/usr/bin/env python3
"""GQA 模型的 KV cache 解析預算表(payload only,不含 paged-cache/metadata 開銷)。"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=28)
ap.add_argument("--kv-heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=128)
a = ap.parse_args()

per_tok_fp16 = 2 * a.layers * a.kv_heads * a.head_dim * 2  # K+V, bytes @ 16-bit
print(f"FP16 KV = {per_tok_fp16 / 1024:.1f} KiB/token")
fmts = {"fp16": 16, "fp8": 8, "int4.5": 4.5, "hybrid3.3": 3.3, "int2.5": 2.5}
print("| context | " + " | ".join(fmts) + " |")
print("|---|" + "---|" * len(fmts))
for c in [4096, 16384, 32768, 65536, 131072, 262144, 1048576]:
    row = [f"{per_tok_fp16 * c * b / 16 / 2**30:.2f}" for b in fmts.values()]
    label = f"{c // 1024}K" if c < 1048576 else "1M"
    print(f"| {label} | " + " | ".join(row) + " | (GiB)")
