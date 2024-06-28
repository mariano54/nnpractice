import dataclasses
from typing import List, Tuple
import torch
import math


@dataclasses.dataclass
class LayerTorch:
    weights: torch.Tensor  # 2d Array of weights: output x input
    biases: torch.Tensor  # added to the W x a


@dataclasses.dataclass
class NNWeightsTorch:
    layers: List[LayerTorch]


def relu_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    x = x - torch.max(x, dim=1, keepdim=True).values
    exps = torch.exp(x)
    denominators = torch.sum(exps, dim=1, keepdim=True)
    return exps / denominators


def compute_nn_pytorch(xs: torch.Tensor, network: NNWeightsTorch) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    activations: List[torch.Tensor] = [xs]
    last_layer_outputs: torch.Tensor = xs  # Two dimensional
    zs: List[torch.Tensor] = []
    for layer_num, layer in enumerate(network.layers):
        z: torch.Tensor = (
                last_layer_outputs @ layer.weights.T + layer.biases
        )  # type:ignore
        zs.append(z)
        next_layer_outputs = (
            relu_torch(z) if layer_num != len(network.layers) - 1 else softmax_torch(z)
        )
        activations.append(next_layer_outputs)
        last_layer_outputs = next_layer_outputs
    return zs, activations


def random_weights_nn(data_size: int, layer_sizes: List[int]) -> NNWeightsTorch:
    activation_size = data_size

    all_layers: List[LayerTorch] = []
    for layer_num, layer_size in enumerate(layer_sizes):
        std_dev = math.sqrt(2) / math.sqrt(activation_size)
        weights = torch.normal(mean=0, std=std_dev, size=(layer_size, activation_size)).float()
        # weights = np.random.normal(0, size=(layer_size, activation_size)) * std_dev
        if layer_num == len(layer_sizes) - 1:
            # biases = np.zeros(layer_size)
            biases = torch.zeros(layer_size).float()
        else:
            # biases = np.random.normal(0, 1, layer_size) * std_dev
            biases = torch.normal(mean=0, std=1, size=(layer_size,)).float()

        all_layers.append(LayerTorch(weights, biases))
        activation_size = layer_size
    return NNWeightsTorch(all_layers)
