# ======================================================
# itse_edit.py
# Implementation of ITSEdit watermarking (Inverse‑Transform Sampling)
# ======================================================

from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch import Tensor

from ..base import BaseWatermark, BaseConfig
from .mersenne import MersenneRNG
from utils.transformers_config import TransformersConfig

class _TorchMersenneAdaptor:
    def __init__(self, seed: int):
        self._rng = MersenneRNG(seed)

    def uniform(self, *size: int) -> Tensor:
        vals = [self._rng.rand() for _ in range(int(np.prod(size)))]
        return torch.tensor(vals, dtype=torch.float32).view(*size).clamp_min(1e-12)

    def randperm(self, n: int) -> Tensor:
        return torch.tensor(self._rng.randperm(n), dtype=torch.int64)


# ───────────────────────────────────────────────────────
# 1.  Config
# ───────────────────────────────────────────────────────
class ITSEditConfig(BaseConfig):

    def initialize_parameters(self) -> None:
        self.pseudo_length   = self.config_dict["pseudo_length"]
        self.sequence_length = self.config_dict["sequence_length"]
        self.n_runs          = self.config_dict["n_runs"]
        self.p_threshold     = self.config_dict["p_threshold"]
        self.key             = self.config_dict["key"]
        self.top_k           = self.config_dict.get("top_k", 0)

        emb_vs = self.generation_model.get_input_embeddings().weight.size(0)
        tok_vs = self.generation_tokenizer.vocab_size
        self.vocab_size = min(emb_vs, tok_vs)

    @property
    def algorithm_name(self) -> str:
        return "ITSEdit"

# ───────────────────────────────────────────────────────
# 2.  Utilities
# ───────────────────────────────────────────────────────
class ITSUtils:
    def __init__(self, cfg: ITSEditConfig):
        self.cfg = cfg
        self.rng = _TorchMersenneAdaptor(cfg.key)

        self.xi: Tensor = self.rng.uniform(cfg.pseudo_length, 1)

        self.pi: Tensor = self.rng.randperm(cfg.vocab_size)
        self.inv_pi: Tensor = torch.empty_like(self.pi)
        self.inv_pi[self.pi] = torch.arange(self.pi.numel())

    def transform_sampling(self, probs: Tensor, u: Tensor) -> Tensor:
        top_k = self.cfg.top_k if self.cfg.top_k > 0 else probs.size(1)
        top_k = min(top_k, probs.size(1))

        idx = self.pi[:top_k]                        # (K,)
        cdf = torch.cumsum(probs[:, idx], dim=1)     # (B,K)
        sel = torch.searchsorted(cdf, u, right=True).clamp(max=top_k-1)
        return idx[sel.squeeze(1)].unsqueeze(1)      # (B,1)

    def phi(self, tokens: Tensor, shift: int) -> Tensor:
        ranks  = self.inv_pi[tokens]                        # (L,)
        u_hat  = (ranks.float() + 0.5) / self.cfg.vocab_size
        xi_seq = self.xi[(shift + torch.arange(tokens.size(0))) %
                         self.cfg.pseudo_length, 0]
        return torch.abs(u_hat - xi_seq).sum()

# ───────────────────────────────────────────────────────
# 3.  ITSEdit main
# ───────────────────────────────────────────────────────
class ITSEdit(BaseWatermark):
    def __init__(
        self,
        algorithm_config: Dict[str, Any] | str | ITSEditConfig,
        transformers_config: TransformersConfig | None = None
    ):
        if isinstance(algorithm_config, str):
            self.config = ITSEditConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, ITSEditConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be JSON path/dict/ITSEditConfig")

        self.utils = ITSUtils(self.config)

    def _rand_shift(self) -> int:
        return torch.randint(self.config.pseudo_length, (1,)).item()

    def _clip(self, ids: Tensor) -> Tensor:
        return torch.remainder(ids, self.config.vocab_size)

    def generate_watermarked_text(self, prompt: str, *_, **__) -> str:
        cfg = self.config
        tok, model = cfg.generation_tokenizer, cfg.generation_model

        encoded = tok.encode(prompt, return_tensors="pt",
                             add_special_tokens=True)
        encoded = self._clip(encoded).to(cfg.device)

        past, shift = None, self._rand_shift()
        for i in range(cfg.sequence_length):
            with torch.no_grad():
                if past is None:
                    out = model(encoded, attention_mask=torch.ones_like(encoded))
                else:
                    out = model(encoded[:, -1:], past_key_values=past)

            probs = torch.softmax(out.logits[:, -1, :cfg.vocab_size], dim=-1).cpu()
            u_row = self.utils.xi[(shift + i) % cfg.pseudo_length].unsqueeze(0)
            token = self._clip(self.utils.transform_sampling(probs, u_row)).to(cfg.device)

            encoded = torch.cat([encoded, token], dim=-1)
            past    = out.past_key_values

        return tok.decode(encoded[0].cpu(), skip_special_tokens=True)

    def detect_watermark(
        self,
        text: str,
        *,
        return_dict: bool = True
    ) -> Dict[str, Any] | Tuple[bool, float]:

        cfg, tok = self.config, self.config.generation_tokenizer
        tokens = self._clip(torch.tensor(tok.encode(text, add_special_tokens=False),
                                         dtype=torch.int64))

        if tokens.numel() == 0:
            res = {"is_watermarked": False, "score": 1.0}
            return res if return_dict else tuple(res.values())

        phis = torch.stack([self.utils.phi(tokens, s) for s in range(cfg.pseudo_length)])
        best_val, best_shift = torch.min(phis, dim=0)

        null_vals = []
        for _ in range(cfg.n_runs):
            xi_rand = torch.rand_like(self.utils.xi).clamp_min(1e-12)
            diff = torch.abs(((self.utils.inv_pi[tokens].float() + 0.5) / cfg.vocab_size) -
                             xi_rand[(best_shift + torch.arange(tokens.size(0))) %
                                     cfg.pseudo_length, 0])
            null_vals.append(diff.sum())
        p_val = (torch.stack(null_vals) <= best_val).float().mean().item()

        res = {"is_watermarked": p_val < cfg.p_threshold, "score": p_val}
        return res if return_dict else tuple(res.values())

    def get_data_for_visualization(self, *_, **__):
        raise NotImplementedError("Visualization is not implemented for ITSEdit.")