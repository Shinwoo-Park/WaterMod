from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from functools import partial
from typing import Any, List, Dict
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from collections import defaultdict
from ..base import BaseWatermark, BaseConfig

class LSHConfig(BaseConfig):
    def initialize_parameters(self) -> None:
        d: Dict[str, Any] = self.config_dict
        self.gamma: float = d['gamma']
        self.delta: float = d['delta']
        self.hash_size: int = d['hash_size']
        self.lsh_seed: int = d['lsh_seed']
        self.phi: float = d['phi']
        self.entropy_threshold: float = d['entropy_threshold']
        self.z_threshold: float = d['z_threshold']
        self.prefix_length: int = d['prefix_length']

    @property
    def algorithm_name(self) -> str:
        return 'LSH'

class LSHUtils:
    def __init__(self, config: LSHConfig) -> None:
        self.config = config
        emb_layer = self.config.generation_model.get_input_embeddings()
        self.embedding_weights = emb_layer.weight.detach()  # [vocab_size, emb_dim]
        self.embedding_dim = emb_layer.embedding_dim
        torch.manual_seed(self.config.lsh_seed)
        self.hyperplanes = torch.randn(
            self.config.hash_size,
            self.embedding_dim,
            device=self.config.device
        )  # [d, emb_dim]
        self.semantic_sets: Dict[int, List[int]] = defaultdict(list)
        for tok in range(self.config.vocab_size):
            emb = self.embedding_weights[tok]
            bits = (self.hyperplanes @ emb >= 0).to(torch.int32)
            key = int(''.join(bits.cpu().numpy().astype(str)), 2)
            self.semantic_sets[key].append(tok)

    def get_key(self, input_ids: torch.LongTensor) -> int:
        last_id = input_ids[-1].item()
        emb = self.embedding_weights[last_id]
        bits = (self.hyperplanes @ emb >= 0).to(torch.int32)
        return int(''.join(bits.cpu().numpy().astype(str)), 2)

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> List[int]:
        key = self.get_key(input_ids)
        g = torch.Generator().manual_seed(key)
        green_ids: List[int] = []
        for bucket in self.semantic_sets.values():
            perm = torch.randperm(len(bucket), generator=g)
            split = int(len(bucket) * self.config.gamma)
            for idx in perm[:split].cpu().numpy():
                green_ids.append(bucket[idx])
        return green_ids

    def compute_entropy(self, scores: torch.FloatTensor) -> float:
        probs = F.softmax(scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        return entropy

    def _compute_z_score(self, observed_count: int, total: int) -> float:
        exp_count = self.config.gamma * total
        num = observed_count - exp_count
        den = math.sqrt(total * self.config.gamma * (1 - self.config.gamma))
        return num / den
    
    def score_sequence(self, input_ids: torch.LongTensor) -> tuple[float, List[int]]:
        total = 0
        count = 0
        flags: List[int] = []
        seq = input_ids.cpu().tolist()
        for i in range(self.config.prefix_length, len(seq)):
            prefix = torch.tensor(seq[:i], device=self.config.device)
            with torch.no_grad():
                out = self.config.generation_model(
                    input_ids=prefix.unsqueeze(0), output_logits=True
                )
                scores = out.logits[0, -1]
            ent = self.compute_entropy(scores)
            if ent < self.config.entropy_threshold:
                continue
            green_ids = set(self.get_greenlist_ids(prefix))
            tok = seq[i]
            flag = 1 if tok in green_ids else 0
            flags.append(flag)
            count += flag
            total += 1
        if total == 0:
            return 0.0, flags
        z = self._compute_z_score(count, total)
        return z, flags

class LSHLogitsProcessor(LogitsProcessor):
    def __init__(self, config: LSHConfig, utils: LSHUtils) -> None:
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        for b in range(scores.size(0)):
            seq = input_ids[b]
            ent = self.utils.compute_entropy(scores[b])
            if ent < self.config.entropy_threshold:
                continue
            mask = torch.zeros_like(scores[b], dtype=torch.bool)
            green_ids = self.utils.get_greenlist_ids(seq)
            mask[green_ids] = True
            bias = self.config.delta / (ent + self.config.phi)
            scores[b][mask] += bias
        return scores

class LSH(BaseWatermark):
    def __init__(self, algorithm_config: str | LSHConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = LSHConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, LSHConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be path or LSHConfig instance")
        self.utils = LSHUtils(self.config)
        self.logits_processor = LSHLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str) -> str:
        gen_fn = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )
        enc = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        out = gen_fn(**enc)
        return self.config.generation_tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def detect_watermark(self, text: str) -> Dict[str, Any]:
        toks = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        z, _ = self.utils.score_sequence(toks)
        is_wm = z > self.config.z_threshold
        return {"is_watermarked": is_wm, "score": z}
    
    def get_data_for_visualization(self, text: str, *_, **__):
        raise NotImplementedError