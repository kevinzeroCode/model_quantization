"""HF runtime 的 cache 工廠與 forward 兼容層(transformers 版本差異都關在這裡)。"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def make_cache(kv: str, model_config=None):
    """kv: fp16 | quanto4 | quanto2 | hybrid:<cfg.json 路徑>"""
    if kv == "fp16":
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            from transformers import DynamicCache
        return DynamicCache()
    if kv.startswith(("quanto", "hqq")):
        backend = "quanto" if kv.startswith("quanto") else "hqq"
        nbits = int(kv[-1])
        if model_config is None:
            raise ValueError(f"{backend} cache requires model_config")
        try:
            from transformers.cache_utils import QuantizedCache
            return QuantizedCache(backend, model_config, nbits=nbits,
                                  q_group_size=64, residual_length=128)
        except (ImportError, TypeError):
            if backend == "hqq":
                raise
            try:
                from transformers.cache_utils import QuantoQuantizedCache
                return QuantoQuantizedCache(model_config, nbits=nbits,
                                            q_group_size=64, residual_length=128)
            except (ImportError, TypeError):
                from transformers import QuantizedCacheConfig, QuantoQuantizedCache
                cfg = QuantizedCacheConfig(backend="quanto", nbits=nbits,
                                           q_group_size=64, residual_length=128)
                try:
                    return QuantoQuantizedCache(cfg)
                except TypeError:
                    return QuantoQuantizedCache(cache_config=cfg)
    if kv.startswith("hybrid:"):
        from hybrid_cache import HybridCfg, HybridPolarKiviCache
        with open(kv.split(":", 1)[1], encoding="utf-8") as f:
            return HybridPolarKiviCache(HybridCfg(**json.load(f)))
    raise ValueError(f"unknown kv config: {kv}")

EFF_BITS = {"fp16": 16, "quanto4": 4.5, "quanto2": 2.5, "hqq4": 4.5, "hqq2": 2.5}  # hybrid 用 cfg.eff_bits()

def fwd(model, ids, cache):
    """帶 cache 的單次 forward;只保留最後一個位置的 logits 以省 VRAM(旗標名隨版本變)。"""
    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
        try:
            return model(input_ids=ids, past_key_values=cache, use_cache=True, **kw)
        except TypeError:
            continue
    raise RuntimeError("forward failed with all logits-kwarg variants")
