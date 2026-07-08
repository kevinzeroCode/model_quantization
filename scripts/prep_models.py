#!/usr/bin/env python3
"""下載模型並建立 YaRN-64K config 變體。

Hugging Face snapshot 目錄中的檔案通常是指向 cache blobs 的 symlink。建立 models/
變體時必須 hardlink symlink 的真實目標，不能 hardlink symlink 本身，否則搬到
models/ 後相對 symlink 會斷掉。
"""
import json
import os
import shutil

from huggingface_hub import snapshot_download

IDS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-0.5B-Instruct",
]


def hardlink_real(src, dst):
    real = os.path.realpath(src)
    try:
        os.link(real, dst)
    except OSError:
        shutil.copy2(real, dst)


for mid in IDS:
    src = snapshot_download(mid)
    dst = "models/" + mid.split("/")[1] + "-yarn64k"
    cfg_path = os.path.join(dst, "config.json")

    if os.path.isdir(dst) and not os.path.exists(cfg_path):
        print(f"removing incomplete/broken model dir: {dst}")
        shutil.rmtree(dst)

    if not os.path.exists(dst):
        shutil.copytree(src, dst, copy_function=hardlink_real)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if os.path.islink(cfg_path):
        os.remove(cfg_path)

    cfg["rope_scaling"] = {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }
    cfg["max_position_embeddings"] = 131072

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"{mid} -> {dst}  rope_scaling={cfg['rope_scaling']}")
