# Summary Tables(自動生成,勿手改)

## T1 速度/容量(vLLM runtime)

|                        |   (4096, 1) |   (4096, 4) |   (4096, 8) |   (16384, 1) |   (16384, 4) |   (16384, 8) |   (32768, 1) |   (32768, 4) |   (32768, 8) |   (63488, 1) |   (63488, 4) |
|:-----------------------|------------:|------------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| ('awq_w4', 'fp16')     |    126.898  |     379.906 |     544.817 |      81.7457 |     181.314  |     185.653  |     43.5642  |      77.7967 |      45.9565 |    21.0706   |     20.4968  |
| ('awq_w4', 'fp8_e4m3') |    nan      |     nan     |     nan     |      53.2705 |      53.8519 |     nan      |      8.07989 |      11.8034 |     nan      |     0.215062 |      2.50588 |
| ('bf16', 'fp16')       |     57.3082 |     197.343 |     326.938 |      43.4541 |     106.859  |      92.5137 |     30.5524  |      41.5776 |      38.9637 |    17.421    |     15.7592  |
| ('bf16', 'fp8_e4m3')   |    nan      |     nan     |     nan     |      30.2335 |      55.2046 |     nan      |     16.385   |      14.7992 |     nan      |     3.03181  |      5.42127 |
| ('gptq_w4', 'fp16')    |    127.166  |     381.447 |     530.142 |      82.0036 |     181.717  |     185.504  |     43.6566  |      78.1501 |      45.5989 |    21.1149   |     19.8737  |

## T1b HF runtime 參考(不得與 T1 互比)

| run_id   | kv_quant   |   ctx_len |   ttft_ms_mean |   tpot_ms_mean |   vram_peak_gb |   eff_kv_bits | status   |
|:---------|:-----------|----------:|---------------:|---------------:|---------------:|--------------:|:---------|
| p4-fp16  | fp16       |     16384 |         2582.4 |          19.73 |          15.33 |          16   | OK       |
| p4-fp16  | fp16       |     32768 |         7070.4 |          22.65 |          16.45 |          16   | OK       |
| p4-hqq4  | int4_hqq   |     16384 |         7338.5 |          27.22 |          14.75 |           4.5 | OK       |
| p4-hqq4  | int4_hqq   |     32768 |        27081.9 |          41.75 |          15.27 |           4.5 | OK       |
| p4-hqq2  | int2_hqq   |     16384 |         8092.2 |          28.2  |          14.64 |           2.5 | OK       |
| p4-hqq2  | int2_hqq   |     32768 |        31043.4 |          43.73 |          15.05 |           2.5 | OK       |

## T1c LMDeploy runtime 參考(不得與 T1/HF 互比)

| run_id      | kv_quant   |   ctx_len |   batch |   ttft_ms_mean |   tpot_ms_mean |   gen_tps |   kv_pool_tokens |   eff_kv_bits | status   |
|:------------|:-----------|----------:|--------:|---------------:|---------------:|----------:|-----------------:|--------------:|:---------|
| p5-lmd-int8 | int8_lmd   |     32768 |       1 |         4282.1 |          17.37 |      6.97 |           217408 |             8 | OK       |
| p5-lmd-int8 | int8_lmd   |     32768 |       4 |         9063.6 |         167.3  |     10.68 |           217408 |             8 | OK       |
| p5-lmd-int8 | int8_lmd   |     63488 |       1 |        10768.5 |          18.17 |      5.3  |           217408 |             8 | OK       |
| p5-lmd-int8 | int8_lmd   |     63488 |       3 |        18020.6 |         241.54 |      5.7  |           217408 |             8 | OK       |
| p5-lmd-int4 | int4_lmd   |     32768 |       1 |         4263.5 |          16.58 |     12.06 |           422080 |             4 | OK       |
| p5-lmd-int4 | int4_lmd   |     32768 |       4 |         8563.7 |         147.78 |     14.21 |           422080 |             4 | OK       |
| p5-lmd-int4 | int4_lmd   |     63488 |       1 |        10750   |          17.39 |      5.28 |           422080 |             4 | OK       |
| p5-lmd-int4 | int4_lmd   |     63488 |       4 |        20920.8 |         361.11 |      5.83 |           422080 |             4 | OK       |

## T2 NIAH 準確率(config × ctx,depth 取平均)

|                                                                     |   4096 |   16384 |   32768 |   63488 |
|:--------------------------------------------------------------------|-------:|--------:|--------:|--------:|
| ('p1-bf16', 'vllm', 'bf16', 'fp16', 'yarn4', 'niah_code')           |  1     |   1     |   1     |       1 |
| ('p1-bf16', 'vllm', 'bf16', 'fp16', 'yarn4', 'niah_en')             |  1     |   1     |   1     |       1 |
| ('p1-bf16', 'vllm', 'bf16', 'fp16', 'yarn4', 'niah_zh')             |  1     |   1     |   1     |       1 |
| ('p1-bf16-norope', 'vllm', 'bf16', 'fp16', 'none', 'niah_zh')       |  1     |   1     |   0     |     nan |
| ('p2-awq', 'vllm', 'awq_w4', 'fp16', 'yarn4', 'niah_en')            |  1     |   1     |   1     |       1 |
| ('p2-awq', 'vllm', 'awq_w4', 'fp16', 'yarn4', 'niah_zh')            |  1     |   1     |   1     |       1 |
| ('p2-gptq', 'vllm', 'gptq_w4', 'fp16', 'yarn4', 'niah_en')          |  1     |   1     |   1     |       1 |
| ('p2-gptq', 'vllm', 'gptq_w4', 'fp16', 'yarn4', 'niah_zh')          |  1     |   1     |   1     |       1 |
| ('p3-awq-fp8', 'vllm', 'awq_w4', 'fp8_e4m3', 'yarn4', 'niah_zh')    |  0.6   |   0     |   0     |       0 |
| ('p3-bf16-fp8', 'vllm', 'bf16', 'fp8_e4m3', 'yarn4', 'niah_zh')     |  0     |   0.133 |   0     |       0 |
| ('p4-fp16', 'hf', 'bf16', 'fp16', 'yarn4', 'niah_en')               |  1     |   1     |   1     |     nan |
| ('p4-fp16', 'hf', 'bf16', 'fp16', 'yarn4', 'niah_zh')               |  1     |   1     |   1     |     nan |
| ('p4-hqq2', 'hf', 'bf16', 'int2_hqq', 'yarn4', 'niah_en')           |  0     |   0     |   0     |     nan |
| ('p4-hqq2', 'hf', 'bf16', 'int2_hqq', 'yarn4', 'niah_zh')           |  0     |   0     |   0     |     nan |
| ('p4-hqq4', 'hf', 'bf16', 'int4_hqq', 'yarn4', 'niah_en')           |  0     |   0     |   0     |     nan |
| ('p4-hqq4', 'hf', 'bf16', 'int4_hqq', 'yarn4', 'niah_zh')           |  0     |   0     |   0     |     nan |
| ('p4-quanto4', 'hf', 'bf16', 'int4_quanto', 'yarn4', 'niah_zh')     |  0     |   0     |   0     |     nan |
| ('p5-lmd-int4', 'lmdeploy', 'bf16', 'int4_lmd', 'yarn4', 'niah_zh') |  0     |   0     |   0     |       0 |
| ('p5-lmd-int8', 'lmdeploy', 'bf16', 'int8_lmd', 'yarn4', 'niah_zh') |  0.467 |   0.067 |   0.067 |       0 |
| ('p6-main', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_en')          |  0     |   0     |   0     |     nan |
| ('p6-main', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_zh')          |  0     |   0     |   0     |     nan |
| ('p6-norot', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_en')         |  0     |   0     |   0     |     nan |
| ('p6-norot', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_zh')         |  0     |   0     |   0     |     nan |
| ('p6-skip01', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_en')        |  1     |   0.933 |   1     |     nan |
| ('p6-skip01', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_zh')        |  1     |   1     |   1     |     nan |
| ('p6-theta3', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_en')        |  0     |   0     |   0     |     nan |
| ('p6-theta3', 'hf', 'bf16', 'hybrid_pk', 'yarn4', 'niah_zh')        |  0     |   0     |   0     |     nan |

## T3 PPL

| run_id    | task       | subset      |   ctx_len | runtime   | weight_quant   | kv_quant   |       value |      n |
|:----------|:-----------|:------------|----------:|:----------|:---------------|:-----------|------------:|-------:|
| p1-bf16   | ppl_vllm   | wikitext2   |      4096 | vllm      | bf16           | fp16       |      6.9572 | 163800 |
| p2-awq    | ppl_vllm   | wikitext2   |      4096 | vllm      | awq_w4         | fp16       |      7.3326 | 163800 |
| p2-gptq   | ppl_vllm   | wikitext2   |      4096 | vllm      | gptq_w4        | fp16       |      7.2823 | 163800 |
| p4-fp16   | ppl_cached | wikitext103 |     16384 | hf        | bf16           | fp16       |      5.4877 |      1 |
| p4-fp16   | ppl_cached | wikitext103 |     32768 | hf        | bf16           | fp16       |      8.0138 |      1 |
| p4-hqq4   | ppl_cached | wikitext103 |     16384 | hf        | bf16           | int4_hqq   |   8485.75   |      1 |
| p4-hqq4   | ppl_cached | wikitext103 |     32768 | hf        | bf16           | int4_hqq   |  26420.4    |      1 |
| p4-hqq2   | ppl_cached | wikitext103 |     16384 | hf        | bf16           | int2_hqq   |  39672.9    |      1 |
| p4-hqq2   | ppl_cached | wikitext103 |     32768 | hf        | bf16           | int2_hqq   | 188154      |      1 |
| p6-main   | ppl_cached | wikitext103 |     16384 | hf        | bf16           | hybrid_pk  |   3613.9    |      1 |
| p6-main   | ppl_cached | wikitext103 |     32768 | hf        | bf16           | hybrid_pk  |   8218.26   |      1 |
| p6-theta3 | ppl_cached | wikitext103 |     16384 | hf        | bf16           | hybrid_pk  |   3485.33   |      1 |
| p6-theta3 | ppl_cached | wikitext103 |     32768 | hf        | bf16           | hybrid_pk  |   6465.6    |      1 |
| p6-norot  | ppl_cached | wikitext103 |     16384 | hf        | bf16           | hybrid_pk  |   3138.16   |      1 |
| p6-norot  | ppl_cached | wikitext103 |     32768 | hf        | bf16           | hybrid_pk  |   2679.74   |      1 |
| p6-skip01 | ppl_cached | wikitext103 |     16384 | hf        | bf16           | hybrid_pk  |      5.9968 |      1 |
| p6-skip01 | ppl_cached | wikitext103 |     32768 | hf        | bf16           | hybrid_pk  |      8.7105 |      1 |

## T4 LongBench(subset × config)

| subset               |   ('vllm', 'awq_w4', 'fp16') |   ('vllm', 'awq_w4', 'fp8_e4m3') |   ('vllm', 'bf16', 'fp16') |   ('vllm', 'bf16', 'fp8_e4m3') |   ('vllm', 'gptq_w4', 'fp16') |
|:---------------------|-----------------------------:|---------------------------------:|---------------------------:|-------------------------------:|------------------------------:|
| 2wikimqa             |                        0.428 |                            0.087 |                      0.486 |                          0.157 |                         0.512 |
| dureader             |                        0.275 |                            0.128 |                      0.273 |                          0.141 |                         0.26  |
| hotpotqa             |                        0.521 |                            0.078 |                      0.552 |                          0.15  |                         0.524 |
| lcc                  |                        0.123 |                            0.142 |                      0.093 |                          0.183 |                         0.075 |
| multifieldqa_zh      |                        0.655 |                            0.355 |                      0.629 |                          0.371 |                         0.632 |
| passage_retrieval_en |                        0.96  |                            0.034 |                      1     |                          0.138 |                         1     |
| passage_retrieval_zh |                        0.82  |                            0.193 |                      0.86  |                          0.24  |                         0.8   |
| repobench-p          |                        0.049 |                            0.153 |                      0.053 |                          0.189 |                         0.043 |
