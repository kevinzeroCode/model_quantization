# Issues

## [2026-07-07] phase0-vllm-cuda13-wheel-incompatible
- 現象:[已驗證] `venv-vllm/bin/python scripts/probe_env.py --venv vllm` 失敗於 `torch_cuda`; torch 2.11.0+cu130 顯示 CUDA driver capability 12080，無法初始化 CUDA。
- 診斷:[已驗證] 未釘版本的 `uv pip install vllm` 解析到 vLLM 0.24.0 與 CUDA 13.0 PyTorch wheel；本機 NVIDIA driver 570.211.01 對目前 PyTorch wheel 太舊。
- 處置:[進行中] 改裝 CUDA 12.x 相容的 vLLM/PyTorch 版本後重跑 probe。
- 影響:env/freeze_vllm.txt 會在修正後重寫；第一次 probe_vllm.json 保留作為失敗紀錄。

## [2026-07-07] phase0-hf-cuda13-wheel-incompatible
- 現象:[已驗證] `venv-hf/bin/python scripts/probe_env.py --venv hf` 失敗於 `torch_cuda` 與 `quantized_cache_smoke`; torch 2.12.1+cu130 無法在目前 driver 初始化 CUDA。
- 診斷:[已驗證] 未釘 PyTorch 版本時解析到 CUDA 13.0 wheel；本機 driver 570.211.01 對該 wheel 太舊。
- 處置:[進行中] 改裝 CUDA 12.8 相容的 torch 2.8.0 後重跑 probe。
- 影響:env/freeze_hf.txt 會在修正後重寫；第一次 probe_hf.json 保留作為失敗紀錄。

## [2026-07-08] phase0-prep-models-broken-symlink
- 現象:[已驗證] `scripts/prep_models.py` 首次執行後，`models/Qwen2.5-7B-Instruct-yarn64k/config.json` 是斷掉的 symlink，導致 FileNotFoundError。
- 診斷:[已驗證] `snapshot_download()` 回傳的 snapshot 檔案是相對 symlink 到 HF cache blobs；原 `copy_function=os.link` hardlink 了 symlink 本身，而不是真實 blob 目標。
- 處置:[已驗證] 修正 `hardlink_real()` 先 `realpath()` 再 hardlink/copy；壞掉的模型目錄會刪除後重建。
- 影響:首次 `prep_models.py` 失敗未產生有效 `models/` 變體，需重跑。

## [2026-07-08] phase0-hybrid-e2e-length-assumption
- 現象:[已驗證] `pytest scripts/test_hybrid_cache.py` 中 6/7 通過；低 bit e2e 生成正常跑完但比 fp16 早 EOS，`len(low) == len(ref)` 失敗。
- 診斷:[已驗證] 低 bit KV 量化改變生成分布，提早 EOS 是品質影響的一部分，不代表 cache pipeline 崩潰。
- 處置:[已驗證] 將 e2e smoke test 改為要求正常產生至少一個新 token 且不超過 `max_new_tokens`；品質差異留給 Phase 6 指標量測。
- 影響:測試語意從「低 bit 輸出等長」改為「低 bit pipeline 可正常生成」。

## [2026-07-08] phase0-vllm-transformers5-tokenizer-incompatible
- 現象:[已驗證] `vllm serve models/Qwen2.5-0.5B-Instruct-yarn64k` 失敗，`Qwen2Tokenizer has no attribute all_special_tokens_extended`。
- 診斷:[已驗證] vLLM 0.10.2 與 transformers 5.13 tokenizer API 不相容。
- 處置:[已驗證] 將 `venv-vllm` 的 transformers 降到 4.57.6、huggingface_hub 降到 0.36.2。
- 影響:env/freeze_vllm.txt 已重寫；後續 vLLM 實驗固定在此版本組合。

## [2026-07-08] phase0-vllm-flashinfer-needs-ninja
- 現象:[已驗證] vLLM 0.5B smoke 載入模型後在 flashinfer sampling JIT 階段失敗，`FileNotFoundError: ninja`。
- 診斷:[已驗證] flashinfer JIT 需要 `ninja` 可執行檔，直接執行 `venv-vllm/bin/vllm` 時 PATH 未包含 venv binary。
- 處置:[進行中] 在 `venv-vllm` 安裝 ninja，後續 vLLM 命令以 `PATH=$PWD/venv-vllm/bin:$PATH` 執行。
- 影響:smoke 需重跑；後續所有 vLLM serve/bench 指令都應帶 venv PATH 或先 activate。

## [2026-07-08] phase0-vllm-flashinfer-sampler-compile-fail
- 現象:[已驗證] vLLM 0.10.2 smoke 在 flashinfer sampling JIT 編譯時失敗，CUDA/CUB 報 `BlockAdjacentDifference<...> has no member FlagHeads`。
- 診斷:[已驗證] flashinfer 0.6.12 的 sampling JIT 與本機 CUDA toolchain 組合不相容；模型載入與 vLLM engine 其他部分正常。
- 處置:[已驗證] 設定 `VLLM_USE_FLASHINFER_SAMPLER=0`，讓 vLLM 使用 PyTorch-native top-k/top-p sampler；0.5B OpenAI endpoint smoke 通過。
- 影響:後續所有 vLLM serve/bench 命令需在環境中設定 `VLLM_USE_FLASHINFER_SAMPLER=0`，速度可能略低於 flashinfer sampler，但避免啟動失敗。

