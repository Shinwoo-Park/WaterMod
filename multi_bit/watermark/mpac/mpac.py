from __future__ import annotations
import math
from typing import Any, List, Dict, Optional
import torch
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from ..base import BaseWatermark, BaseConfig

class MPACConfig(BaseConfig):
    def initialize_parameters(self) -> None:
        
        self.b = self.config_dict["b"]
        self.k = int(self.config_dict["k"])
        self.gamma = 1.0 / self.k
        self.b_tilde = int(math.ceil(self.b / math.log2(self.k)))
        self.delta = float(self.config_dict["delta"])
        self.z_threshold = float(self.config_dict["z_threshold"])
        self.prefix_length = int(self.config_dict["prefix_length"])
        self.f_scheme = self.config_dict["f_scheme"]
        self.hash_key = int(self.config_dict["hash_key"])
        def int_to_base_k(x: int, k: int, length: int) -> List[int]:
            msg = [0] * length
            for i in range(length - 1, -1, -1):
                msg[i] = x % k
                x //= k
            return msg
        max_val = self.k ** self.b_tilde
        hash_key_mod = self.hash_key % max_val
        self.message = int_to_base_k(hash_key_mod, self.k, self.b_tilde)

    @property
    def algorithm_name(self) -> str:
        return "MPAC"

class MPACUtils:
    def __init__(self, config: MPACConfig):
        self.config = config
        self._f_map = {
            "additive": self._f_additive,
            "time":     self._f_time,
            "skip":     self._f_skip,
            "min":      self._f_min,
        }

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

    def choose_position(self, prefix_ids: torch.LongTensor) -> (int, int, float):
        seed = self.f(prefix_ids)
        u    = self._hash_to_uniform(seed)
        p    = min(int(u * self.config.b_tilde), self.config.b_tilde - 1)
        return p, seed, u

    def partition_vocab(self, seed: int) -> List[torch.LongTensor]:
        g    = torch.Generator(self.config.device)
        g.manual_seed(seed)
        perm = torch.randperm(self.config.vocab_size, generator=g, device=self.config.device)
        size = self.config.vocab_size // self.config.k
        return [perm[i*size:(i+1)*size] for i in range(self.config.k)]

class MPACLogitsProcessor(LogitsProcessor):
    def __init__(self, config: MPACConfig, utils: MPACUtils):
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(-1) < self.config.prefix_length:
            return scores

        for b in range(scores.size(0)):
            prefix = input_ids[b]
            p, seed, _ = self.utils.choose_position(prefix)
            m = self.config.message[p]
            lists = self.utils.partition_vocab(seed)
            color_ids = lists[m]
            scores[b, color_ids] += self.config.delta
        return scores

class MPAC(BaseWatermark):
    def __init__(self, algorithm_config: str | MPACConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = MPACConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, MPACConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be path or MPACConfig instance")
        self.utils = MPACUtils(self.config)
        self.processor = MPACLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str) -> Dict[str, Any]:
        lp = LogitsProcessorList([self.processor])
        inputs = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        out = self.config.generation_model.generate(logits_processor=lp, **inputs, **self.config.gen_kwargs)
        text = self.config.generation_tokenizer.decode(out[0], skip_special_tokens=True)
        return text

    def detect_watermark(self, text: str) -> Dict[str, Any]:
        toks = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        b_tilde, k = self.config.b_tilde, self.config.k
        if len(toks) <= self.config.prefix_length:
            return {"is_watermarked": False, "score": 0.0}
        counts = [[0]*k for _ in range(b_tilde)]
        T = 0
        for i in range(self.config.prefix_length, len(toks)):
            prefix = toks[:i]
            p, seed, _ = self.utils.choose_position(prefix)
            lists = self.utils.partition_vocab(seed)
            for m, lst in enumerate(lists):
                if int(toks[i]) in lst:
                    counts[p][m] += 1
                    T += 1
                    break
        g = sum(counts[p][ self.config.message[p] ] for p in range(b_tilde))
        p_null = 1.0/k
        # z-score
        sigma = math.sqrt(T * p_null * (1-p_null) + 1e-12)
        z = (g - p_null * T) / sigma
        return {"is_watermarked": (z > self.config.z_threshold), "score": z}

    def get_data_for_visualization(self, text: str, *_, **__):
        raise NotImplementedError