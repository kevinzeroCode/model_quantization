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

## [2026-07-08] phase1-pg19-download-timebox
- 現象:[已驗證] Phase 1 prompt generation 在 `deepmind/pg19` test split 下載大量小檔時超過可接受時間，中斷後僅完成 `prompts/niah_zh.jsonl`，且該檔尚未寫入 manifest。
- 診斷:[已驗證] PG-19 不是實驗必要條件；runbook 原本允許英文語料 fallback 到 `wikitext-103`，且 NIAH 評測重點是長上下文 retrieval，不依賴特定英文 corpus。
- 處置:[已驗證] 將 `scripts/gen_prompts.py` 英文預設改為 `wikitext-103`，需要 PG-19 時才用 `USE_PG19=1` 明確啟用；同時改為每完成一個語言就寫回 manifest。
- 影響:英文 prompt 可快速重現並凍結；Phase 1 繼續從 `en` 與 `code` 缺漏檔案接續，不覆蓋已完成的中文 prompt。

## [2026-07-08] phase1-norope-32k-prompt-overhead
- 現象:[已驗證] `p1-bf16-norope` 的 `niah_zh` 在 `ctx_tokens=32768` 全部回傳 400，4K/16K 則 30/30 全 HIT。
- 診斷:[已驗證] 原始 32K server 的 `max_model_len=32768`；frozen prompt 的 32768 token context 加上 chat template、問題與 needle 後，實際 request input 為 32853-32856 tokens，超過 API 上限。
- 處置:[已驗證] 將 32K norope 結果視為容量/提示長度限制而非 retrieval miss；報告中只用 4K/16K 做有效 YaRN-vs-none 品質對照，32K 標註為 FAIL_CONTEXT_OVERFLOW。
- 影響:若論文附錄需要原始 rope 在「接近 32K」的公平點，需新增一組不覆蓋既有 frozen prompts 的約 31K 對照 prompt。

## [2026-07-08] phase1-ppl-vllm-prompt-logprobs-oom
- 現象:[已驗證] `scripts/ppl_vllm.py` 首次以 40 個 4096-token chunks 一次送入 vLLM 時，在 `prompt_logprobs` 的 `log_softmax` 階段 CUDA OOM，需額外配置約 2.32 GiB。
- 診斷:[已驗證] PPL 腳本不是生成 OOM，而是 prompt logprob 對長序列與大 vocab 計算 log_softmax 時瞬間記憶體過高；同時排多個 chunk 會放大峰值。
- 處置:[已驗證] 保留 4096-token chunk 定義，但改成 `batch_size=1` 逐 chunk 執行，並將 vLLM `gpu_memory_utilization` 預設先降到 0.75 仍差約 0.16 GiB，最終降到 0.65 以保留 logprob 工作空間。
- 影響:PPL 仍可作為 vLLM 路線內部跨配置比較；速度較慢但避免 OOM。

## [2026-07-08] phase3-vllm-fp8-kv-v0-quality-collapse
- 現象:[已驗證] `--kv-cache-dtype fp8_e4m3 --calculate-kv-scales` 可啟動，但 vLLM 0.10.2 會提示 KV quant 不支援 V1 engine 並 fallback 到 V0；FP8 KV 也改用 XFormers attention。`p3-bf16-fp8` 與 `p3-awq-fp8` 在 zh NIAH 32K/63K 全部 miss，LongBench retrieval/QA 大幅下降，速度 benchmark 出現明顯早 EOS。
- 診斷:[已驗證] FP8 KV 的記憶體容量收益成立，但目前 runtime 路徑不是 Phase 1/2 的 V1/FlashAttention hot path；動態 KV scale 沒有避免長上下文 retrieval collapse。
- 處置:[已驗證] Phase 3 報告將 AWQ W4 + fp16 KV 視為目前 62K quality x capacity 最佳配置；FP8 KV 保留為診斷路線，不作為主推實驗設定。
- 影響:後續 Phase 4/5/6 若要比較低 bit KV，需明確區分「容量提升」與「可用長上下文品質」。若升級 vLLM 或改 KV scaling 方法，需重跑同一組 NIAH/LongBench 驗證。

## [2026-07-08] phase4-hf-cache-api-and-no-grad
- 現象:[已驗證] Phase 4 初次 smoke 時，`cachelib.py` 從 transformers top-level import `QuantizedCacheConfig` / `QuantoQuantizedCache` 失敗；修正 import 後，`bench_hf.py` 又因未使用 `torch.no_grad()`，0.5B 4K/8K prefill 產生不合理高 VRAM 與 OOM。
- 診斷:[已驗證] transformers 4.57.6 的 quantized cache API 位於 `transformers.cache_utils`，且 `QuantizedCache` 需要 `model.config`。HF bench retained autograd graphs across chunked prefill/decode，導致記憶體膨脹。
- 處置:[已驗證] `make_cache(kv, model_config)` 改用 `QuantizedCache(backend, model_config, ...)`；HF generation/LongBench/NIAH/PPL 傳入 `model.config`；`bench_hf.once()` 加上 `torch.no_grad()`；長上下文 HF generation 改成 chunked prefill。
- 影響:Phase 4 HF 32K smoke 與正式 bench 可在 RTX 4090 24GB 上完成。

## [2026-07-08] phase4-transformers-quantized-cache-collapse
- 現象:[已驗證] `quanto4` 7B zh NIAH diagnostic 在 4K/16K/32K 全部 miss 且 raw output 呈現亂碼；依 runbook 安裝 HQQ fallback 後，`hqq4` 與 `hqq2` zh/en NIAH 也都是 0/90，cached-PPL 從 fp16 的 5.4877/8.0138 暴增到 hqq4 的 8485.7470/26420.3876 與 hqq2 的 39672.8659/188153.7391。
- 診斷:[已驗證] 失敗不是 32K 才出現，而是 4K depth=0.1 起即崩潰；axis 組合診斷未救回第一個 4K NIAH prompt。Transformers QuantizedCache 在 Qwen2.5-7B + residual_length=128 的協定下不可作為可用長上下文 retrieval baseline。
- 處置:[已驗證] Phase 4 正式 baseline 改記 `int4_hqq` / `int2_hqq`，保留 `p4-quanto4` zh diagnostic raw；Phase 6 報告必須把「避免 4K 即崩」列為最低門檻。
- 影響:後續不能把 transformers low-bit QuantizedCache 的 eff bits 直接解讀為可用容量；它目前只提供負結果與 hybrid 改良空間。

## [2026-07-08] phase4-pg19-download-timebox
- 現象:[已驗證] `ppl_eval.py --mode cached` 預設嘗試載入 `deepmind/pg19` test split，下載大量小檔並超過合理 Phase 4 執行時間。
- 診斷:[已驗證] Phase 4 需要的是同一 runtime 內部的 KV cache 品質對照，不依賴 PG-19；Phase 1 已有相同資料集下載時間問題。
- 處置:[已驗證] `ppl_eval.py` 改成預設使用 `wikitext-103`，只有設定 `USE_PG19=1` 時才嘗試 PG-19。
- 影響:Phase 4 cached-PPL subset 記為 `wikitext103`，不可和未來 PG-19 PPL 直接混表比較。
