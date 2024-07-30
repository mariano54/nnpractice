import math
from typing import Tuple, Optional

import torch

from src.gpt2_weights import GPT2Weights


def linear_rw(input_dim: int, output_dim: int, zero_biases: bool= False, residual_scaling_num_layers: Optional[int]=None, device:str="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    std_dev = math.sqrt(1) / math.sqrt(output_dim)
    if residual_scaling_num_layers is not None:
        std_dev /= 2 * math.sqrt(residual_scaling_num_layers)

    weights = (
        torch.normal(mean=0, std=std_dev, size=(input_dim, output_dim))
        .float()
        .to(device)
    )
    if zero_biases:
        bias = torch.zeros(output_dim).float().to(device)
    else:
        bias = (
            torch.normal(mean=0, std=1, size=(output_dim,)).float().to(device)
        )
    return weights, bias


def layer_batch_norm_rw(layer_size: int, device: str="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    bn_gain = torch.randn((1, layer_size)).to(device) * 0.1 + 1.0
    bn_bias = torch.randn((1, layer_size)).to(device) * 0.1
    return bn_gain, bn_bias


def generate_random_weights(T: int, C: int, vocab_size: int, device) -> GPT2Weights:
    # TODO: finish

    ln_scaling = 0.1
    ln_scaling_plus = 1.0
    std = 1 / math.sqrt(C)
    wte = torch.normal(0, std, (vocab_size, C)).to(device)
    wpe = torch.normal(0, std, (T, C)).to(device)
    for i in range(12):
        ln_1_weight = torch.randn((1, C)).to(device) * ln_scaling + ln_scaling_plus
        ln_1_bias = torch.randn((1, C)).to(device) * ln_scaling
        # attn_weight =

        ln_2_weight = torch.randn((1, C)).to(device) * ln_scaling + ln_scaling_plus
        ln_2_bias = torch.randn((1, C)).to(device) * ln_scaling

    lm_head_weight = wte
    ln_f_gain = torch.randn((1, C)).to(device) * ln_scaling + ln_scaling_plus
    ln_f_bias = torch.randn((1, C)).to(device) * ln_scaling
