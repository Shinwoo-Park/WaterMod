from __future__ import annotations
import math
from typing import Any, Dict
import torch
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from ..base import BaseWatermark, BaseConfig

class SAMConfig(BaseConfig):

    def initialize_parameters(self) -> None:
        self.delta = self.config_dict["delta"]
        self.z_threshold = self.config_dict["z_threshold"]
        self.prefix_length = self.config_dict["prefix_length"]
        self.f_scheme = self.config_dict["f_scheme"]
        self.hash_key = self.config_dict["hash_key"]
        self.entropy_type = self.config_dict["entropy_type"]
        self.tau = self.config_dict["tau"]
        self.H_scale = self.config_dict["H_scale"]

    @property
    def algorithm_name(self) -> str:
        return "SAM"

# ============================================
# 2. SAMUtils — PRF, Entropy, z-score helper
# ============================================
class SAMUtils:
    def __init__(self, config: SAMConfig):
        self.config = config
        self._f_map = {
            "additive": self._f_additive,
            "time":     self._f_time,
            "skip":     self._f_skip,
            "min":      self._f_min,
        }

    # PRF variants
    def _f_additive(self, ids): return int(ids[-self.config.prefix_length:].sum())
    def _f_time(self,     ids): return int(torch.prod(ids[-self.config.prefix_length:]))
    def _f_skip(self,     ids): return int(ids[-self.config.prefix_length].item())
    def _f_min(self,      ids): return int(ids[-self.config.prefix_length:].min())

    def f(self, ids: torch.LongTensor) -> int:
        return self._f_map[self.config.f_scheme](ids)

    def _hash_to_uniform(self, seed: int) -> float:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed ^ self.config.hash_key)
        return float(torch.rand((), generator=g))

    def choose_group(self, logits: torch.Tensor, prefix_ids: torch.LongTensor) -> int:
        # entropy based even/odd
        probs = torch.softmax(logits.float(), dim=-1)
        if self.config.entropy_type == "shannon":
            H = float(-(probs * torch.log(probs + 1e-12)).sum())
            H_max = math.log2(self.config.vocab_size)
        else:
            τ = self.config.tau
            H = float((probs / (1.0 + τ * probs)).sum())
            H_max = 1.0 / (1.0 + (τ / self.config.vocab_size))
        p_even = min(1.0, max(0.0, (H / H_max)**self.config.H_scale))
        seed = self.f(prefix_ids)
        u = self._hash_to_uniform(seed)
        return 0 if u < p_even else 1

    def z_score(self, g_count: int, T: int) -> float:
        p_null = 1.0 / 2
        mu = p_null * T
        sigma = math.sqrt(p_null * (1-p_null) * T) if T > 0 else 1e-8
        return (g_count - mu) / sigma

# ============================================
# 3. SAMLogitsProcessor — generate() bias
# ============================================
class SAMLogitsProcessor(LogitsProcessor):
    def __init__(self, config: SAMConfig, utils: SAMUtils):
        self.config = config
        self.utils = utils

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.size(-1) < self.config.prefix_length:
            return scores

        for b in range(scores.size(0)):
            logits = scores[b]
            prefix = input_ids[b]
            gid = self.utils.choose_group(logits, prefix)
            rank = torch.argsort(logits, descending=True)
            idx = torch.arange(len(rank), device=rank.device)
            mask = ((idx + 1) % 2 == gid)
            green = rank[mask]
            scores[b, green] += self.config.delta

        return scores

# ============================================
# 4. SAMWatermark — public interface
# ============================================
class SAM(BaseWatermark):
    def __init__(self, algorithm_config: str | SAMConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SAMConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SAMConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be path or SAMConfig instance")
        self.utils = SAMUtils(self.config)
        self.logits_processor = SAMLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str) -> str:
        lp = LogitsProcessorList([self.logits_processor])
        inputs = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        out = self.config.generation_model.generate(
            logits_processor=lp,
            **inputs,
            **self.config.gen_kwargs
        )
        return self.config.generation_tokenizer.decode(out[0], skip_special_tokens=True)

    def detect_watermark(self, text: str) -> Dict[str, Any]:
        toks = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)
        if len(toks) <= self.config.prefix_length:
            return {"is_watermarked": False, "score": 0.0}

        g_count = 0
        T = 0
        for i in range(self.config.prefix_length, len(toks)):
            prefix = toks[:i]
            logits = self.config.generation_model(prefix.unsqueeze(0))["logits"][0, -1]
            gid = self.utils.choose_group(logits, prefix)
            rank = torch.argsort(logits, descending=True)
            idx = torch.arange(len(rank), device=rank.device)
            mask = ((idx + 1) % 2 == gid)
            in_green = int(toks[i] in rank[mask])
            g_count += in_green
            T += 1

        z = self.utils.z_score(g_count, T)
        return {"is_watermarked": z > self.config.z_threshold,
                "score": z}
    
    def get_data_for_visualization(self, text: str, *_, **__):
        raise NotImplementedError