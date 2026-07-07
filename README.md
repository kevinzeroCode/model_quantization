# 長上下文 LLM 量化實驗

本目錄是 `EXPERIMENT_RUNBOOK.txt` 的實作工作區，用來執行長上下文 LLM
量化實驗。

核心問題：

> 在長上下文推論中，壓縮 model weights 比較划算，還是壓縮 KV cache 比較划算？

預計研究貢獻是 Hybrid Polar-KIVI KV cache quantization：

- Key cache：極座標 fake quantization。
- Value cache：per-token fake quantization。
- 比較對象：相近 effective bit budget 下的 uniform int2/int4 KV cache quantization。

## 目前環境判定

探查日期：2026-07-07。

| 條件 | 目前狀態 | 判定 |
|---|---:|---|
| OS / 架構 | Ubuntu Linux, x86_64 | PASS |
| GPU | NVIDIA GeForce RTX 4090 | PASS |
| GPU memory | 24564 MiB | PASS |
| Driver | 570.211.01 | PASS |
| GPU 空閒狀態 | 探查時沒有 active compute process | PASS |
| 磁碟空間 | `/` 尚有 614 GB | PASS |
| RAM | 125 GiB total, 116 GiB available | PASS |
| Python | 3.12.3 | PASS |
| CUDA compiler | CUDA 12.0 `nvcc` 存在 | PASS |
| Hugging Face 連線 | `curl -I https://huggingface.co` 回 HTTP 200 | PASS |
| git / curl | 已安裝 | PASS |
| `uv` | 尚未安裝 | TODO |
| Python packages | system Python 尚無 `torch` / `transformers` / `vllm` | TODO |
| Git repository | 尚未 `git init` | TODO |

結論：這台機器的硬體條件適合做實驗，符合 runbook 對 4090 實驗機的主要假設。
但它還不是已完成 provision 的實驗環境。正式跑 7B benchmark 前，必須先完成
Phase 0：安裝 `uv`、建立雙 venv、安裝依賴、跑 probe、下載模型、建立結果追蹤檔。

## 實驗設計

實驗分成兩條 runtime 路線。

| 路線 | Runtime | 用途 | 主要輸出 |
|---|---|---|---|
| 部署路線 | vLLM | 量 BF16、AWQ、GPTQ、FP8 KV，以及可選 TurboQuant / LMDeploy fallback 的真實 serving 表現 | TTFT、TPOT、吞吐、KV 容量、OOM 邊界 |
| 研究路線 | Hugging Face transformers | 量 int4/int2 quantized cache 與 Hybrid Polar-KIVI 的品質影響 | NIAH accuracy、LongBench score、cached PPL |

兩條路線共用凍結 prompt 與結果 schema；但不同 runtime 的速度數字不得放在同一張速度比較表。

### 實驗因子

| 因子 | 規劃值 |
|---|---|
| 模型 | Qwen2.5-7B-Instruct；0.5B 用於 smoke test |
| Weight quantization | `bf16`, `awq_w4`, `gptq_w4` |
| KV quantization | `fp16`, `fp8_e4m3` 或 `fp8_e5m2`, `int4_quanto`, `int2_quanto`, `hybrid_pk`, optional `turbo` / `int8_lmd` / `int4_lmd` |
| Context length | 4K, 16K, 32K, 62K/64K；Spark 上可選 128K-1M |
| 品質任務 | NIAH zh/en/code、LongBench 子集、cached PPL |

### 公平比較規則

- Prompt 只生成一次，並在 `prompts/MANIFEST.json` 記錄 SHA256。
- 解碼固定 greedy：`temperature=0`、`seed=42`。
- 所有數字都必須來自 `results/perf.csv`、`results/quality.csv` 或 `results/raw/`。
- 失敗要記 `FAIL`、`OOM` 或 `SKIP`；OOM 邊界本身也是實驗數據。
- HF simulated KV quantization 不宣稱實測 VRAM 節省，只用 `eff_kv_bits` 做解析比較。
- vLLM 與 HF 的速度數字不互比。
- 套件版本與第三方 repo commit 都要記錄。

## Phase 規劃

| Phase | 目標 | 主要證據 |
|---|---|---|
| P0 | 建置環境與腳本 | `env/probe_*.json`、`pip freeze`、smoke test、pytest |
| P1 | BF16 baseline 與凍結 prompts | baseline `perf.csv`、NIAH/LongBench/PPL rows |
| P2 | AWQ/GPTQ weight-only baseline | weight memory 與品質代價 |
| P3 | vLLM 2x2 factorial：`{bf16, awq} x {fp16 KV, fp8 KV}` | 部署路線主表 |
| P4 | HF int4/int2 quantized KV baseline | 低 bit KV 品質曲線 |
| P5 | 可選 TurboQuant 或 LMDeploy low-bit KV | 部署級低 bit KV 證據 |
| P6 | Hybrid Polar-KIVI | 研究主貢獻與 ablation |
| P7 | 場景分析 | 中文、程式碼、RAG/multi-document 敏感度 |
| P8 | 可選 Spark 容量邊界 | 128K-1M 可行性 |
| P9 | 彙整與最終報告 | `reports/final_report.md` 與 summary tables |

流程圖見 [docs/experiment_flow.mmd](docs/experiment_flow.mmd)。

## 第三方 Repo 策略

會使用外部 repo，但要分清楚角色。

| Repo | 在本專案的角色 | 是否進主表 |
|---|---|---|
| `THUDM/LongBench` | benchmark config 與 metric 來源 | 是，作為共用 evaluation harness |
| `jy-yuan/KIVI` | paper-faithful KIVI feasibility 與附錄比較 | 只有模型/runtime 可比時才進主表 |
| `SqueezeAILab/KVQuant` | 強 baseline audit 或附錄復現 | 通常放附錄 |
| `spcl/QuaRot` | rotation-based quantization 參考 | 附錄或 related baseline |
| `facebookresearch/SpinQuant` | learned-rotation quantization 參考 | 附錄或 related baseline |

主表應盡量固定同一模型、同一批 prompts、同一套 metrics、同一 runtime 類別。
如果第三方 repo 只支援 Llama/Mistral，而主實驗用 Qwen2.5，結果應放在附錄或文獻對照，
不要混進主 factorial table。

## 實作順序

1. 初始化 workspace：

   ```bash
   git init
   mkdir -p env scripts prompts results/raw reports logs models third_party
   ```

2. 若 `uv` 仍不存在，先安裝：

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. 將 `EXPERIMENT_RUNBOOK.txt` 附錄中的 scripts 抽出到 `scripts/`。

4. 編譯檢查 scripts：

   ```bash
   python3 -m py_compile scripts/*.py
   ```

5. 依 runbook 建立 `venv-vllm` 與 `venv-hf`。

6. 完成 Phase 0 probes 與 smoke tests 後，才開始下載/跑 7B grid。

7. 依依賴順序跑實驗：

   ```text
   P0 -> P1 -> P2 -> P3
         P4 -> P6 -> P7 -> P9
         P5 optional
         P8 optional on Spark
   ```

## 結果檔案

| 路徑 | 意義 |
|---|---|
| `STATE.json` | Phase 進度與路線決策 |
| `ISSUES.md` | 失敗、fallback、偏離、stale 結果紀錄 |
| `env/` | probe outputs、版本檔、package freezes |
| `prompts/` | 凍結 prompt sets 與 SHA256 manifest |
| `results/perf.csv` | 速度、容量、記憶體、OOM 資料 |
| `results/quality.csv` | NIAH、LongBench、PPL 品質資料 |
| `results/raw/` | 原始 benchmark JSON/JSONL |
| `reports/` | Phase reports 與 final report |
| `third_party/` | 外部 repo clone 與 commit 紀錄 |

## 目前下一步

先做 `P0-lite`：初始化 git、建立目錄骨架、從 runbook 抽出 scripts、做 compile check，
並新增第三方 repo compatibility audit。這些通過後，再進入完整依賴安裝與模型下載。
