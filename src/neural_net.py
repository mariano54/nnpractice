import dataclasses
from typing import List, Tuple, Optional
import torch
import math


@dataclasses.dataclass
class LayerTorch:
    weights: torch.Tensor  # 2d Array of weights: output x input
    biases: torch.Tensor  # added to the W x a
    batch_norm: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]  # Two tensors of length output, gain and bias
    running_mean: Optional[float]
    running_var: Optional[float]


@dataclasses.dataclass
class NNWeightsTorch:
    layers: List[LayerTorch]

class NeuralNetwork:

    def __init__(self, network: NNWeightsTorch):
        self.network = network
        self.eval = False

    def relu_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))

    def gelu(sekf, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.0044715 * x ** 3)))


    def softmax_torch(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.max(x, dim=1, keepdim=True).values
        exps = torch.exp(x)
        denominators = torch.sum(exps, dim=1, keepdim=True)
        return exps / denominators


    def compute_batch_norm(self, z: torch.Tensor, running_mean: Optional[float], running_var: Optional[float],
                           bn_gain: torch.Tensor, bn_bias: torch.Tensor) -> torch.Tensor:
        if not self.eval:
            bn_mean = (1.0 / z.shape[0]) * (z.sum(0, keepdim=True))
            bn_diff = z - bn_mean
            bn_diff_sq = bn_diff ** 2
            bn_var = (1.0 / z.shape[0]) * bn_diff_sq.sum(
                0, keepdim=True
            )  # 1/m  * sum(xi-xhat)^2
            bn_var_2 = (1.0 / (z.shape[0] - 1)) * bn_diff_sq.sum(0, keepdim=True)

        else:
            assert running_mean is not None
            assert running_var is not None
            bn_mean = running_mean
            bn_var = running_var
            bn_var_2 = running_var
            bn_diff = z - bn_mean

        bn_var_inv = (bn_var + 1e-5) ** -0.5  # This is 1/sqrt(var + epsilon)
        x_hat = bn_diff * bn_var_inv
        preact = (
                bn_gain
                * (z - z.mean(0, keepdim=True))
                / torch.sqrt(z.var(0, keepdim=True, unbiased=True) + 1e-5)
                + bn_bias
        )
        return (preact, bn_var_inv, x_hat, bn_mean, bn_var_2)
        # preact_list.append((bn_var_inv, x_hat))
        # batch_means.append(bn_mean)
        # batch_vars.append(bn_var_2)

    def compute_nn_pytorch(
        self, xs: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[Optional[torch.Tensor]],
        List[Optional[float]],
        List[Optional[float]],
    ]:
        # print(f"l1 Biases: {network.layers[-1].biases}")
        # print(f"l0 Weights: {network.layers[0].weights}")
        activations: List[torch.Tensor] = [xs]
        last_layer_outputs: torch.Tensor = xs  # Two dimensional
        zs: List[torch.Tensor] = []
        preact_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        batch_means: List[Optional[float]] = []
        batch_vars: List[Optional[float]] = []
        for layer_num, layer in enumerate(self.network.layers):
            z: torch.Tensor = (
                    last_layer_outputs @ layer.weights.T + layer.biases
            )  # type:ignore

            if layer.batch_norm is not None:
                pass
            else:
                preact = z
                preact_list.append(None)
                batch_means.append(None)
                batch_vars.append(None)
            zs.append(preact)
            next_layer_outputs = (
                self.relu_torch(preact)
                if layer_num != len(self.network.layers) - 1
                else self.softmax_torch(preact)
            )
            # print(f"Next layer: {next_layer_outputs}")
            activations.append(next_layer_outputs)
            last_layer_outputs = next_layer_outputs
        return zs, activations, preact_list, batch_means, batch_vars


def random_weights_nn(
        data_size: int, layer_sizes: List[Tuple[int, bool]], device: str, seed: Optional[int] = None
) -> NNWeightsTorch:
    if seed is not None:
        torch.manual_seed(seed)
    activation_size = data_size

    all_layers: List[LayerTorch] = []
    for layer_num, (layer_size, has_batchnorm) in enumerate(layer_sizes):
        std_dev = math.sqrt(2) / math.sqrt(activation_size)
        weights = torch.normal(
            mean=0, std=std_dev, size=(layer_size, activation_size)
        ).float()
        # weights = np.random.normal(0, size=(layer_size, activation_size)) * std_dev
        if layer_num == len(layer_sizes) - 1:
            # biases = np.zeros(layer_size)
            biases = torch.zeros(layer_size).float()
        else:
            # biases = np.random.normal(0, 1, layer_size) * std_dev
            biases = torch.normal(mean=0, std=1, size=(layer_size,)).float()

        batch_norm = None
        running_mean = None
        running_var = None
        if has_batchnorm:
            bn_gain = torch.randn((1, layer_size)) * 0.1 + 1.0
            bn_bias = torch.randn((1, layer_size)) * 0.1
            batch_norm = (bn_gain.to(device), bn_bias.to(device))
            running_mean = torch.zeros(layer_size).to(device)
            running_var = torch.ones(layer_size).to(device)

        all_layers.append(
            LayerTorch(
                weights.to(device),
                biases.to(device),
                batch_norm,
                running_mean,
                running_var,
            )
        )
        activation_size = layer_size
    return NNWeightsTorch(all_layers)
