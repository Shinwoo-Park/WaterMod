# ======================================================
# exp_edit.py
# Implementation of EXPEdit watermarking (Exponential‑Minimum)
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
        flat = [self._rng.rand() for _ in range(int(np.prod(size)))]
        return torch.tensor(flat, dtype=torch.float32).view(*size)

    def randint(self, high: int, *size: int) -> Tensor:
        flat = [self._rng.randint() % high for _ in range(int(np.prod(size)))]
        return torch.tensor(flat, dtype=torch.int64).view(*size)

    def randperm(self, n: int) -> Tensor:
        return torch.tensor(self._rng.randperm(n), dtype=torch.int64)


# ───────────────────────────────────────────────────────
# 1.   Config
# ───────────────────────────────────────────────────────
class EXPEditConfig(BaseConfig):

    def initialize_parameters(self) -> None:
        self.pseudo_length: int = self.config_dict["pseudo_length"]        
        self.sequence_length: int = self.config_dict["sequence_length"]    
        self.n_runs: int = self.config_dict["n_runs"]                      
        self.p_threshold: float = self.config_dict["p_threshold"]          
        self.key: int = self.config_dict["key"]                            
        self.top_k: int = self.config_dict.get("top_k", 0)                 

    @property
    def algorithm_name(self) -> str:
        return "EXPEdit"


# ───────────────────────────────────────────────────────
# 2.   Algorithm Utilities
# ───────────────────────────────────────────────────────
class EXPUtils:
    def __init__(self, cfg: EXPEditConfig):
        self.cfg = cfg
        self.rng = _TorchMersenneAdaptor(cfg.key)

        # ξ : (n, |V|)   –  (0,1] 
        self.xi: Tensor = self.rng.uniform(cfg.pseudo_length, cfg.vocab_size).clamp_min(1e-12)

    def expmin_sampling(self, probs: Tensor, xi_row: Tensor, top_k: int = 0) -> Tensor:
        xi_row = xi_row.clamp_min(1e-12)              
        if top_k > 0:
            top_probs, top_idx = torch.topk(probs, top_k, dim=-1)        # (B,K)
            xi_sel = torch.gather(xi_row, 1, top_idx)                    # (B,K)
            gumbel_ratio = -torch.log(xi_sel) / top_probs
            sel = torch.argmin(gumbel_ratio, dim=-1, keepdim=True)       # (B,1)
            return torch.gather(top_idx, 1, sel)                         # (B,1)

        # full‑vocab
        gumbel_ratio = -torch.log(xi_row) / probs                        # (B,|V|)
        return torch.argmin(gumbel_ratio, dim=-1, keepdim=True)          # (B,1)

    def psi(self, tokens: Tensor, shift: int) -> Tensor:
        idx = (shift + torch.arange(tokens.numel())) % self.cfg.pseudo_length
        xi_slice = self.xi[idx, tokens]                                       
        xi_slice = xi_slice.clamp_max(1 - 1e-12)                              
        return torch.log1p(-xi_slice).sum()                                   


# ───────────────────────────────────────────────────────
# 3.   EXPEdit Top‑Level
# ───────────────────────────────────────────────────────
class EXPEdit(BaseWatermark):
    def __init__(
        self,
        algorithm_config: Dict[str, Any] | str | EXPEditConfig,
        transformers_config: TransformersConfig | None = None
    ):
        if isinstance(algorithm_config, str):
            self.config = EXPEditConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, EXPEditConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be path/JSON or EXPEditConfig.")

        self.utils = EXPUtils(self.config)

    def _rand_shift(self) -> int:
        return torch.randint(self.config.pseudo_length, (1,)).item()

    def generate_watermarked_text(self, prompt: str, *_, **__) -> str:
        cfg = self.config
        tok, model = cfg.generation_tokenizer, cfg.generation_model

        encoded = tok.encode(prompt, return_tensors="pt", add_special_tokens=True).to(cfg.device)
        attn_mask = torch.ones_like(encoded)
        past = None
        shift = self._rand_shift()

        for i in range(cfg.sequence_length):
            with torch.no_grad():
                output = (model(encoded[:, -1:], past_key_values=past, attention_mask=attn_mask)
                          if past else model(encoded))
            probs = torch.softmax(output.logits[:, -1, :cfg.vocab_size], dim=-1).cpu()

            xi_row = self.utils.xi[(shift + i) % cfg.pseudo_length].unsqueeze(0)      # (1,|V|)
            token = self.utils.expmin_sampling(probs, xi_row, cfg.top_k).to(cfg.device)  # (1,1)

            encoded = torch.cat([encoded, token], dim=-1)
            past = output.past_key_values
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=-1)

        return tok.decode(encoded[0].cpu(), skip_special_tokens=True)

    def detect_watermark(
        self,
        text: str,
        *,
        return_dict: bool = True
    ) -> Dict[str, Any] | Tuple[bool, float]:

        cfg = self.config
        tok = cfg.generation_tokenizer
        tokens = torch.tensor(tok.encode(text, add_special_tokens=False), dtype=torch.int64)

        if tokens.numel() == 0:
            result = {"is_watermarked": False, "score": 1.0}
            return result if return_dict else tuple(result.values())

        psis = torch.stack([self.utils.psi(tokens, s) for s in range(cfg.pseudo_length)])   # (n,)
        best_val, best_shift = torch.min(psis, dim=0)

        null_vals = []
        for _ in range(cfg.n_runs):
            xi_rand = torch.rand_like(self.utils.xi).clamp_max(1 - 1e-12)

            idx = (best_shift + torch.arange(tokens.numel())) % cfg.pseudo_length
            slice_r = xi_rand[idx, tokens]
            null_vals.append(torch.log1p(-slice_r).sum())
        null_vals = torch.stack(null_vals)

        p_val = (null_vals <= best_val).float().mean().item()

        result = {"is_watermarked": p_val < cfg.p_threshold, "score": p_val}
        return result if return_dict else tuple(result.values())

    def get_data_for_visualization(self, *_, **__):
        raise NotImplementedError("Visualization is not implemented for EXPEdit.")
