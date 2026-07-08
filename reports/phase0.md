# Phase 0 — 環境建立與驗證(2026-07-08, host=4090)

**結論**:[已驗證] 本機可作為 4090 實驗機使用。vLLM 與 Hugging Face 兩個 venv 都已建立，required probes 全部 PASS；四個 Qwen2.5 YaRN-64K 模型變體已建立；Hybrid cache pytest 7/7 通過；vLLM 0.5B OpenAI-compatible smoke 回傳非空中文內容。TurboQuant probe 為 FAIL，因此 Phase 5 走 B fallback。vLLM 需固定使用 `VLLM_USE_FLASHINFER_SAMPLER=0` 避開本機 CUDA/CUB 與 flashinfer sampling JIT 的相容性問題。

**驗收證據**:

| 項目 | 結果 | 證據 |
|---|---:|---|
| `env/probe_vllm.json` required checks | PASS | `torch_cuda=True`, `vllm_import=0.10.2`; `vllm_turboquant=False` 為非阻塞 |
| `env/probe_hf.json` required checks | PASS | `torch_cuda=True`, `transformers=4.57.6`, `quantized_cache_smoke=True` |
| scripts 編譯 | PASS | `python -m py_compile scripts/*.py` 無錯誤 |
| Hybrid cache pytest | 7/7 PASS | `pytest scripts/test_hybrid_cache.py -v` |
| vLLM 0.5B smoke | PASS | `env/vllm_smoke_models.json`, `env/vllm_smoke_response.json` |
| YaRN 模型變體 | 4/4 PASS | `models/Qwen2.5-{7B,7B-AWQ,7B-GPTQ,0.5B}*-yarn64k/config.json` 含 `rope_scaling.type=yarn` |
| 版本凍結 | PASS | `env/freeze_vllm.txt`, `env/freeze_hf.txt`, `env/versions.json` |

**主要數據**:

| 類別 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, driver 570.211.01 |
| vLLM env | vLLM 0.10.2, torch 2.8.0+cu128, transformers 4.57.6 |
| HF env | torch 2.8.0+cu128, transformers 4.57.6, datasets 2.21.0, optimum-quanto 0.2.7 |
| 模型目錄大小 | `models/` 約 26G |
| vLLM 0.5B smoke response | 「量子力學是一種描述原子、分子和費米納粒子等微观粒子相互作用的理論...」 |

**偏離與問題**:

- `phase0-vllm-cuda13-wheel-incompatible`:[已驗證] 未釘版本的 vLLM 解析到 CUDA 13 wheel，driver 570 無法初始化；已改用 vLLM 0.10.2 + torch 2.8.0+cu128。
- `phase0-hf-cuda13-wheel-incompatible`:[已驗證] HF venv 同樣需降到 torch 2.8.0+cu128。
- `phase0-prep-models-broken-symlink`:[已驗證] 修正 `prep_models.py`，hardlink HF cache symlink 的 realpath。
- `phase0-hybrid-e2e-length-assumption`:[已驗證] 低 bit 量化允許提早 EOS，測試改為檢查 pipeline 正常生成。
- `phase0-vllm-transformers5-tokenizer-incompatible`:[已驗證] vLLM 0.10.2 與 transformers 5.x tokenizer API 不相容，兩個 venv 均固定 transformers 4.57.6。
- `phase0-vllm-flashinfer-needs-ninja`:[已驗證] vLLM flashinfer JIT 需要 venv PATH 中的 `ninja`。
- `phase0-vllm-flashinfer-sampler-compile-fail`:[已驗證] flashinfer sampling JIT 與本機 CUDA/CUB 組合不相容；後續 vLLM 命令需設定 `VLLM_USE_FLASHINFER_SAMPLER=0`。
