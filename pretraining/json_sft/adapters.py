"""Low-rank attention/FFN adaptation with ordinary merged serving weights.

The optimizer checkpoint additionally holds the unmerged adapter tensors, so a
resume reconstructs the frozen parent rather than applying adapters twice.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        self.base = base
        base.freeze()
        self.scale = alpha / rank
        self.lora_a = (mx.random.normal((rank, base.weight.shape[1])) / math.sqrt(base.weight.shape[1])).astype(base.weight.dtype)
        self.lora_b = mx.zeros((base.weight.shape[0], rank), dtype=base.weight.dtype)

    def __call__(self, x):
        return self.base(x) + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale

    def merged_weight(self):
        return (self.base.weight.astype(mx.float32) + self.scale * (self.lora_b.astype(mx.float32) @ self.lora_a.astype(mx.float32))).astype(self.base.weight.dtype)


def install(model, rank=16, alpha=32):
    model.freeze()
    for block in model.blocks:
        for module, key in ((block.attn, 'qkv_proj'), (block.attn, 'proj'), (block.ffn, 'gate_up'), (block.ffn, 'down')):
            module[key] = LoRALinear(module[key], rank, alpha)
    return model


def merged_parameters(model):
    flat = dict(tree_flatten(model.parameters()))
    result = {}
    for key, value in flat.items():
        if key.endswith(('.lora_a', '.lora_b')):
            continue
        result[key.replace('.base.', '.')] = value
    for i, block in enumerate(model.blocks):
        for module, key, path in ((block.attn, 'qkv_proj', 'attn.qkv_proj'), (block.attn, 'proj', 'attn.proj'), (block.ffn, 'gate_up', 'ffn.gate_up'), (block.ffn, 'down', 'ffn.down')):
            result[f'blocks.{i}.{path}.weight'] = module[key].merged_weight()
    return tree_unflatten(list(result.items()))


class ExportView:
    def __init__(self, model):
        self.model = model

    def parameters(self):
        return merged_parameters(self.model)
