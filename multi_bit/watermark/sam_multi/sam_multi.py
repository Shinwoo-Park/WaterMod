from __future__ import annotations

import math
from typing import Any, List, Dict, Optional

import torch
from collections import Counter

from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList

from ..base import BaseWatermark, BaseConfig


class SAM_MULTIConfig(BaseConfig):
    def initialize_parameters(self) -> None:
        self.k = self.config_dict["k"]
        self.b = self.config_dict["b"]
        self.b_tilde = int(math.ceil(self.b / math.log2(self.k)))

        self.delta = self.config_dict["delta"]
        self.z_threshold = self.config_dict["z_threshold"]
        self.prefix_length = self.config_dict["prefix_length"]

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
        return "SAM_MULTI"


class SAM_MULTIUtils:
    def __init__(self, config: SAM_MULTIConfig):
        self.config = config
        self._f_map = {
            "additive": self._f_additive,
            "time":     self._f_time,
            "skip":     self._f_skip,
            "min":      self._f_min,
        }

    # PRF variants on the last L (=prefix_length) tokens
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
        u = self._hash_to_uniform(seed)
        p = min(int(u * self.config.b_tilde), self.config.b_tilde - 1)
        return p, seed, u

    def z_score(self, g_count: int, T: int) -> float:
        p_null = 1.0 / self.config.k
        mu = p_null * T
        sigma = math.sqrt(p_null * (1 - p_null) * T) if T > 0 else 1e-8
        return (g_count - mu) / sigma


class SAMLogitsProcessor(LogitsProcessor):
    def __init__(self, config: SAM_MULTIConfig, utils: SAM_MULTIUtils):
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Do nothing until we have enough prefix tokens
        if input_ids.size(-1) < self.config.prefix_length:
            return scores

        # Batch-wise processing
        for b in range(scores.size(0)):
            logits = scores[b]
            prefix = input_ids[b]

            # Choose digit position p (1-base convention is maintained elsewhere)
            p, _, _ = self.utils.choose_position(prefix)
            m = self.config.message[p]  # expected color digit in base-k

            # Rank tokens by descending logit and build 1-base indices
            rank = torch.argsort(logits, descending=True)
            idx1 = torch.arange(len(rank), device=rank.device) + 1  # 1-base

            # 1-base modular mask: (idx1 % k == m) is the "green list"
            mask = (idx1 % self.config.k == m)
            green = rank[mask]

            # Bias the selected color by delta
            scores[b, green] += self.config.delta

        return scores


class SAM_MULTI(BaseWatermark):
    def __init__(
        self,
        algorithm_config: str | SAM_MULTIConfig,
        transformers_config: TransformersConfig | None = None,
        *args, **kwargs
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = SAM_MULTIConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SAM_MULTIConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be path or SAM_MULTIConfig instance")

        self.utils = SAM_MULTIUtils(self.config)
        self.processor = SAMLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str) -> str:
        lp = LogitsProcessorList([self.processor])
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
            return {
                "is_watermarked": False,
                "score": 0.0,
                "recovered_message": [],
                "recovered_message_base_k_str": "",
                "T": 0,
                "counts": [],
            }

        b_tilde, k = self.config.b_tilde, self.config.k
        # counts[p][d] = how many times observed color d occurred at digit position p
        counts: List[List[int]] = [[0] * k for _ in range(b_tilde)]
        T = 0

        for i in range(self.config.prefix_length, len(toks)):
            prefix = toks[:i]
            # White-box re-logit with the same model
            logits = self.config.generation_model(prefix.unsqueeze(0))["logits"][0, -1]

            # Recompute the digit position p using the same PRF + key schedule
            p, _, _ = self.utils.choose_position(prefix)  # 0 <= p < b_tilde

            # Build rank (descending) and 1-base rank positions
            rank = torch.argsort(logits, descending=True)

            # Find 1-base rank r1 of the *observed* token x_t = toks[i]
            pos0 = (rank == int(toks[i])).nonzero(as_tuple=True)[0]
            if pos0.numel() == 0:
                # If not found (edge: vocab mismatch), skip this step
                continue
            r1 = int(pos0.item()) + 1  # 1-base rank

            # Observed color with 1-base modular rule
            d_obs = r1 % k  # 0..k-1

            # Tally the observed color at digit position p
            counts[p][d_obs] += 1
            T += 1

        # Detection z-score uses hits on the *expected* color m[p]
        G = 0
        for p in range(b_tilde):
            m_expected = self.config.message[p]  # 0..k-1
            G += counts[p][m_expected]

        z = self.utils.z_score(G, T)

        # Message recovery by majority per digit position
        m_hat = [int(torch.tensor(counts[p]).argmax().item()) for p in range(b_tilde)]
        recovered_str = "".join(self._digit_to_char(d) for d in m_hat)

        return {
            "is_watermarked": bool(z > self.config.z_threshold),
            "score": float(z),
            "recovered_message": m_hat,                  # list[int], length = b_tilde
            "recovered_message_base_k_str": recovered_str,
            "T": int(T),
            "counts": counts,                            # optional: for debugging/plots
        }

    # Pretty-print base-k digits up to k=36 (0-9a-z)
    def _digit_to_char(self, d: int) -> str:
        if 0 <= d <= 9:
            return str(d)
        return chr(ord('a') + (d - 10))
