"""Hybrid Polar-KIVI KV cache(研究鷹架,simulated quantization)。

Key   : 極座標量化 —— 旋轉後把 channel 兩兩配對 -> (r, theta);theta 均勻量化 bits_theta bit,
        r 逐 token 以非負均勻量化 bits_r bit。這是 PolarQuant 想法的簡化獨立實作
        (independent implementation),不是 Google 官方程式碼。
Value : KIVI 風格 per-token 非對稱均勻量化(bits_v bit,沿 channel 每 v_group 一組 scale/zero)。
共同  : 最近 residual 個 token 保持全精度(KIVI 的 residual window)。
Simulated = 量化後立刻反量化、仍以 bf16 存放:品質影響是真的,記憶體節省是解析值
(用 eff_bits_* 計,不得宣稱實測 VRAM 節省)。已在 transformers 4.5x 驗證 DynamicCache.update
回傳的張量即內部儲存,原地覆寫會保留;若升級後 e2e 測試失敗,先查這個假設。"""
import math
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class HybridCfg:
    bits_theta: int = 4
    bits_r: int = 4
    bits_v: int = 2
    v_group: int = 64
    residual: int = 128
    rot_seed: int = 0
    use_rotation: bool = True
    skip_layers: tuple = field(default_factory=tuple)  # 保持全精度的層(mixed-precision ablation)

    def eff_bits_key(self):
        # 每 2 個 channel 存 1 個角度 + 1 個半徑;每 token 另存 1 個 fp16 的 r_max scale
        return (self.bits_theta + self.bits_r) / 2 + 16 / 128
    def eff_bits_value(self):
        return self.bits_v + 32 / self.v_group  # 每組 fp16 scale + fp16 zero
    def eff_bits(self):
        return round((self.eff_bits_key() + self.eff_bits_value()) / 2, 2)


def random_orthogonal(d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))  # 固定符號,消除 QR 不唯一性


def polar_fakequant(x: torch.Tensor, bits_theta: int, bits_r: int,
                    rot: torch.Tensor | None) -> torch.Tensor:
    """x: (..., d), d 為偶數。回傳同 shape、同 dtype 的量化-反量化結果。"""
    orig_dtype = x.dtype
    x = x.float()
    if rot is not None:
        x = x @ rot
    pairs = x.view(*x.shape[:-1], -1, 2)
    r = torch.linalg.vector_norm(pairs, dim=-1)                    # (..., d/2), >= 0
    theta = torch.atan2(pairs[..., 1], pairs[..., 0])              # [-pi, pi)
    L = 2 ** bits_theta
    q_theta = torch.round((theta + math.pi) * L / (2 * math.pi)) % L
    theta_hat = q_theta * (2 * math.pi / L) - math.pi
    Lr = 2 ** bits_r - 1
    r_max = r.amax(dim=-1, keepdim=True).clamp(min=1e-6)           # 每 token 一個 scale
    q_r = torch.round(r / r_max * Lr).clamp(0, Lr)
    r_hat = q_r / Lr * r_max
    y = torch.stack((r_hat * torch.cos(theta_hat), r_hat * torch.sin(theta_hat)), dim=-1)
    y = y.reshape(*x.shape)
    if rot is not None:
        y = y @ rot.T
    return y.to(orig_dtype)


def pertoken_fakequant(v: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """v: (..., d)。沿最後一維每 group 個 channel 一組,非對稱均勻量化(KIVI value 風格)。"""
    orig_dtype = v.dtype
    x = v.float()
    g = x.view(*x.shape[:-1], -1, group)
    vmin = g.amin(dim=-1, keepdim=True)
    vmax = g.amax(dim=-1, keepdim=True)
    L = 2 ** bits - 1
    scale = ((vmax - vmin) / L).clamp(min=1e-8)
    q = torch.round((g - vmin) / scale).clamp(0, L)
    return (q * scale + vmin).reshape(*v.shape).to(orig_dtype)


class HybridPolarKiviCache(DynamicCache):
    def __init__(self, cfg: HybridCfg):
        super().__init__()
        self.cfg = cfg
        self.q_ptr: dict[int, int] = {}   # 每層已量化到的 token index
        self._rot: torch.Tensor | None = None

    def _rotation(self, d: int, device) -> torch.Tensor:
        if self._rot is None or self._rot.shape[0] != d:
            self._rot = random_orthogonal(d, self.cfg.rot_seed).float()
        if self._rot.device != device:
            self._rot = self._rot.to(device)
        return self._rot

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = super().update(key_states, value_states, layer_idx, cache_kwargs)
        boundary = k.shape[-2] - self.cfg.residual
        done = self.q_ptr.get(layer_idx, 0)
        if boundary > done:
            if layer_idx not in tuple(self.cfg.skip_layers):
                sl = slice(done, boundary)
                rot = self._rotation(k.shape[-1], k.device) if self.cfg.use_rotation else None
                k[..., sl, :] = polar_fakequant(k[..., sl, :],
                                                self.cfg.bits_theta, self.cfg.bits_r, rot)
                v[..., sl, :] = pertoken_fakequant(v[..., sl, :],
                                                   self.cfg.bits_v, self.cfg.v_group)
            self.q_ptr[layer_idx] = boundary
        return k, v
