import math

import pytest
import torch

from hybrid_cache import (HybridCfg, HybridPolarKiviCache, pertoken_fakequant,
                          polar_fakequant, random_orthogonal)

torch.manual_seed(0)  # 測試資料固定,避免統計性斷言偶發飄移


def rel_err(a, b):
    return (torch.linalg.vector_norm((a - b).float())
            / torch.linalg.vector_norm(b.float())).item()


def test_rotation_orthogonal():
    q = random_orthogonal(128, seed=0)
    assert torch.allclose(q @ q.T, torch.eye(128), atol=1e-5)


def test_polar_high_bits_near_exact():
    x = torch.randn(2, 4, 64, 128)
    y = polar_fakequant(x, bits_theta=16, bits_r=16, rot=random_orthogonal(128))
    assert rel_err(y, x) < 1e-2


def test_polar_4bit_reasonable():
    x = torch.randn(2, 4, 256, 128)
    y = polar_fakequant(x, bits_theta=4, bits_r=4, rot=random_orthogonal(128))
    e = rel_err(y, x)
    assert 0.02 < e < 0.25, f"unexpected rel err {e}"


def test_pertoken_monotonic_in_bits():
    v = torch.randn(2, 4, 256, 128)
    e2 = rel_err(pertoken_fakequant(v, 2, 64), v)
    e4 = rel_err(pertoken_fakequant(v, 4, 64), v)
    assert e4 < e2 < 0.6


def test_cache_residual_untouched():
    cfg = HybridCfg(residual=128)
    cache = HybridPolarKiviCache(cfg)
    k0 = torch.randn(1, 4, 300, 128)
    v0 = torch.randn(1, 4, 300, 128)
    k, v = cache.update(k0.clone(), v0.clone(), 0)
    assert torch.equal(k[..., -128:, :], k0[..., -128:, :])   # 殘差窗完全不動
    assert not torch.equal(k[..., :172, :], k0[..., :172, :])  # 窗外必然被改
    assert not torch.equal(v[..., :172, :], v0[..., :172, :])
    assert cache.q_ptr[0] == 172


def test_cache_multi_update_pointer():
    cfg = HybridCfg(residual=50)
    cache = HybridPolarKiviCache(cfg)
    a = torch.randn(1, 4, 100, 128)
    b = torch.randn(1, 4, 100, 128)
    cache.update(a.clone(), a.clone(), 0)
    k, _ = cache.update(b.clone(), b.clone(), 0)
    assert k.shape[-2] == 200 and cache.q_ptr[0] == 150
    assert torch.equal(k[..., -50:, :], b[..., -50:, :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_e2e_generation_matches_fp16_when_nothing_quantized():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    ids = tok.apply_chat_template([{"role": "user", "content": "請背誦圓周率前十位。"}],
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    def gen(cache):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=48, do_sample=False,
                                 past_key_values=cache)
        return out[0].tolist()
    from transformers import DynamicCache
    ref = gen(DynamicCache())
    # residual 大於總長 -> 什麼都不量化 -> 必須與 fp16 完全一致(驗證管線本身無副作用)
    same = gen(HybridPolarKiviCache(HybridCfg(residual=4096)))
    assert same == ref
    # 低 bit 真量化 -> 需正常跑完並產出新 token；量化後允許提早 EOS。
    low = gen(HybridPolarKiviCache(HybridCfg(bits_theta=4, bits_r=4, bits_v=2, residual=32)))
    assert ids.shape[1] < len(low) <= ids.shape[1] + 48
