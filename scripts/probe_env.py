#!/usr/bin/env python3
"""Phase-0 環境探查。分別在兩個 venv 各跑一次:python scripts/probe_env.py --venv hf|vllm"""
import argparse, importlib, json, os, shutil, subprocess, sys
from datetime import datetime, timezone


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", choices=["hf", "vllm"], required=True)
    a = ap.parse_args()
    info = {"time": datetime.now(timezone.utc).isoformat(), "venv": a.venv,
            "python": sys.version, "checks": {}}

    def check(name, passed, detail=""):
        info["checks"][name] = {"pass": bool(passed), "detail": str(detail)[:500]}
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {str(detail)[:200]}")

    smi = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    check("nvidia_smi", smi and "ERROR" not in smi, smi)
    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    check("disk_free_300gb", free_gb >= 300, f"{free_gb:.0f} GB free")
    try:
        import torch
        check("torch_cuda", torch.cuda.is_available(),
              f"torch {torch.__version__}, cuda {torch.version.cuda}, "
              f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no dev'}")
    except Exception as e:  # noqa: BLE001
        check("torch_cuda", False, e)

    if a.venv == "hf":
        for mod in ["transformers", "datasets", "optimum.quanto", "accelerate"]:
            try:
                m = importlib.import_module(mod)
                check(f"import_{mod}", True, getattr(m, "__version__", "ok"))
            except Exception as e:  # noqa: BLE001
                check(f"import_{mod}", False, e)
        try:  # quantized-cache smoke(首次會下載 0.5B,約 1 GB)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                                         device_map="cuda")
            ids = tok.apply_chat_template([{"role": "user", "content": "1+1=?"}],
                                          add_generation_prompt=True,
                                          return_tensors="pt").to("cuda")
            out = model.generate(ids, max_new_tokens=16, do_sample=False,
                                 cache_implementation="quantized",
                                 cache_config={"backend": "quanto", "nbits": 4})
            txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            check("quantized_cache_smoke", len(txt.strip()) > 0, txt[:80])
        except Exception as e:  # noqa: BLE001
            check("quantized_cache_smoke", False, e)
    else:
        try:
            import vllm
            check("vllm_import", True, vllm.__version__)
        except Exception as e:  # noqa: BLE001
            check("vllm_import", False, e)
        try:  # informational:TurboQuant 是否存在(假設 C5)
            importlib.import_module("vllm.model_executor.layers.quantization.turboquant")
            check("vllm_turboquant", True, "module present -> Phase 5 走 A 路線")
        except Exception as e:  # noqa: BLE001
            check("vllm_turboquant", False,
                  f"not found ({type(e).__name__}) -> Phase 5 走 B fallback")

    os.makedirs("env", exist_ok=True)
    path = f"env/probe_{a.venv}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    required_fail = [k for k, r in info["checks"].items()
                     if not r["pass"] and k != "vllm_turboquant"]
    print(("ALL REQUIRED PASS" if not required_fail
           else f"REQUIRED FAIL: {required_fail}") + f" -> {path}")
    sys.exit(1 if required_fail else 0)


if __name__ == "__main__":
    main()
