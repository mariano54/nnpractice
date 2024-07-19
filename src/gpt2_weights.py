import dataclasses
from typing import List

import torch


@dataclasses.dataclass
class AttentionWeights:
    q_weight: torch.Tensor
    q_bias: torch.Tensor
    k_weight: torch.Tensor
    k_bias: torch.Tensor
    v_weight: torch.Tensor
    v_bias: torch.Tensor
    proj_weight: torch.Tensor
    proj_bias: torch.Tensor

    def to(self, device: str) -> "AttentionWeights":
        return AttentionWeights(
            self.q_weight.to(device), self.q_bias.to(device),
            self.k_weight.to(device), self.k_bias.to(device),
            self.v_weight.to(device), self.v_bias.to(device),
            self.proj_weight.to(device), self.proj_bias.to(device),
        )


@dataclasses.dataclass
class MLPWeights:
    fc_weight: torch.Tensor
    fc_bias: torch.Tensor
    proj_weight: torch.Tensor
    proj_bias: torch.Tensor

    def to(self, device: str) -> "MLPWeights":
        return MLPWeights(self.fc_weight.to(device), self.fc_bias.to(device), self.proj_weight.to(device),
                          self.proj_bias.to(device))


@dataclasses.dataclass
class TransformerWeights:
    ln_1_weight: torch.Tensor
    ln_1_bias: torch.Tensor
    attention: AttentionWeights
    mlp: MLPWeights
    ln_2_weight: torch.Tensor
    ln_2_bias: torch.Tensor

    def to(self, device: str) -> "TransformerWeights":
        return TransformerWeights(self.ln_1_weight.to(device), self.ln_1_bias.to(device), self.attention.to(device),
                                  self.mlp.to(device), self.ln_2_weight.to(device), self.ln_2_bias.to(device))


@dataclasses.dataclass
class GPT2Weights:
    wte: torch.Tensor
    wpe: torch.Tensor
    transformer: List[TransformerWeights]
    transformer_ln_f_weight: torch.Tensor
    transformer_ln_f_bias: torch.Tensor
    lm_head_weight: torch.Tensor

    def to(self, device: str) -> "GPT2Weights":
        return GPT2Weights(self.wte.to(device), self.wpe.to(device), [t.to(device) for t in self.transformer],
                           self.transformer_ln_f_weight.to(device), self.transformer_ln_f_bias.to(device),
                           self.lm_head_weight.to(device))
