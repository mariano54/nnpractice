import math
from typing import Tuple, Optional

import torch

from src.gpt2_weights import GPT2Weights


def linear_rw(
    input_dim: int,
    output_dim: int,
    zero_biases: bool = False,
    residual_scaling_num_layers: Optional[int] = None,
    device: str = "cpu",
    std_dev: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if std_dev is None:
        std_dev = math.sqrt(1) / math.sqrt(output_dim)
    if residual_scaling_num_layers is not None:
        std_dev /= math.sqrt(residual_scaling_num_layers)

    weights = torch.normal(mean=0, std=std_dev, size=(input_dim, output_dim)).float().to(device)
    if zero_biases:
        bias = torch.zeros(output_dim).float().to(device)
    else:
        bias = torch.normal(mean=0, std=1, size=(output_dim,)).float().to(device)
    return weights, bias


def layer_batch_norm_rw(layer_size: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    bn_gain = torch.randn((1, layer_size)).to(device) * 0.1 + 1.0
    bn_bias = torch.randn((1, layer_size)).to(device) * 0.1
    return bn_gain, bn_bias
