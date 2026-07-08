#!/usr/bin/env python3
"""生成並凍結 NIAH 長上下文評測集(Phase 1.1 跑一次,之後不得重生成)。
輸出 prompts/niah_{zh,en,code}.jsonl 與 MANIFEST.json(sha256 鎖定)。
語料:zh=LongBench(multifieldqa_zh/dureader 的 context)、en=PG-19(fallback wikitext-103)、
code=codeparrot-clean-valid。全部以 Qwen tokenizer 控長。"""
import argparse, hashlib, json, os, random

from datasets import load_dataset
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

NEEDLE = {
    "zh": ("本文件的通關密語是 {code}。請務必記住這個密語。",
           "文件中藏有一個通關密語(格式 KEY-XXXXXX),請找出並只回答該密語。"),
    "en": ("The secret passcode of this document is {code}. Remember it carefully.",
           "A secret passcode (format KEY-XXXXXX) is hidden in the document. "
           "Reply with ONLY the passcode."),
    "code": ("# NOTE: deployment passcode is {code} (do not remove this line)",
             "A deployment passcode (format KEY-XXXXXX) is hidden in a code comment. "
             "Reply with ONLY the passcode."),
}


def load_corpus(lang, n_docs=300):
    docs = []
    if lang == "zh":
        for sub in ["multifieldqa_zh", "dureader"]:
            try:
                ds = load_dataset("THUDM/LongBench", sub, split="test",
                                  trust_remote_code=True)
            except Exception:  # org 更名 fallback
                ds = load_dataset("zai-org/LongBench", sub, split="test",
                                  trust_remote_code=True)
            docs += [x["context"] for x in ds]
    elif lang == "en":
        try:
            ds = load_dataset("deepmind/pg19", split="test", trust_remote_code=True)
            docs = [x["text"] for x in ds]
        except Exception:
            joined = "\n".join(load_dataset("wikitext", "wikitext-103-raw-v1",
                                            split="test")["text"])
            docs = [joined[i:i + 200_000] for i in range(0, len(joined), 200_000)]
    elif lang == "code":
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                          streaming=True)
        for i, x in enumerate(ds):
            if i >= n_docs * 2:
                break
            docs.append(x["content"])
    docs = [d for d in docs if len(d) > 2000]
    return docs[:n_docs]


def build_context(docs, target_tokens, rng):
    parts, total = [], 0
    while total < target_tokens + 1024:
        d = docs[rng.randrange(len(docs))]
        parts.append(d)
        total += len(TOK(d, add_special_tokens=False).input_ids)
    text = "\n\n===== 文件分隔 / DOCUMENT SPLIT =====\n\n".join(parts)
    ids = TOK(text, add_special_tokens=False).input_ids[:target_tokens]
    return TOK.decode(ids)


def make_one(docs, lang, ctx_tokens, depth, rng, idx):
    code = "KEY-" + "".join(rng.choice("ABCDEF0123456789") for _ in range(6))
    needle_tpl, question = NEEDLE[lang]
    needle = needle_tpl.format(code=code)
    ctx = build_context(docs, ctx_tokens, rng)
    pos = int(len(ctx) * depth)
    cut = ctx.find("。", pos) if lang == "zh" else ctx.find("\n", pos)
    pos = cut + 1 if cut > 0 else pos
    ctx = ctx[:pos] + "\n" + needle + "\n" + ctx[pos:]
    head = "以下是長文件內容:" if lang == "zh" else "Here is a long document:"
    prompt = f"{head}\n\n{ctx}\n\n{'問題:' if lang == 'zh' else 'Question: '}{question}"
    return {"id": f"niah-{lang}-{ctx_tokens}-{depth}-{idx}", "mode": "niah", "lang": lang,
            "ctx_tokens": ctx_tokens, "depth": depth, "needle": code,
            "question": question, "prompt": prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 32768, 63488])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--langs", nargs="+", default=["zh", "en", "code"])
    ap.add_argument("--suffix", default="")  # Phase 8 的 128K 增量檔用
    a = ap.parse_args()
    os.makedirs("prompts", exist_ok=True)
    mpath = "prompts/MANIFEST.json"
    manifest = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {}
    for lang in a.langs:
        rng = random.Random(f"{a.seed}-{lang}")  # 字串 seed:跨行程可重現
        docs = load_corpus(lang)
        assert docs, f"corpus empty for {lang}"
        rows = [make_one(docs, lang, c, d, rng, i)
                for c in a.ctx for d in a.depths for i in range(a.samples)]
        path = f"prompts/niah_{lang}{a.suffix}.jsonl"
        assert not os.path.exists(path), f"{path} 已存在:凍結集不得重生成(§A.6)"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path, "rb") as f:
            manifest[path] = {"n": len(rows),
                              "sha256": hashlib.sha256(f.read()).hexdigest()}
        print(f"{path}: {len(rows)} prompts")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
