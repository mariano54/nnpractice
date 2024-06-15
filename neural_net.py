import dataclasses
from typing import List
from numpy.typing import NDArray
import numpy as np
import math

from load_data import load_data

float32 = np.float32


@dataclasses.dataclass
class Layer:
    weights: NDArray[float32]  # 2d Array of weights: output x input
    biases: NDArray[float32]  # added to the W x a


@dataclasses.dataclass
class NNWeightsNew:
    layers: List[Layer]


NNWeightsNew.__module__ = __name__  # provide name for pickling the class


def relu(x: NDArray[float32]) -> NDArray[float32]:
    return np.maximum(x, 0)


def softmax(x: NDArray[float32]) -> NDArray[float32]:
    exps = [np.exp(x_i) for x_i in x]
    for e in exps:
        if math.isnan(e):
            print("Xs:", x)
    denominator = np.sum(exps) + 0.000001
    output = np.zeros(x.size, dtype=float32)
    for i, e in enumerate(exps):
        output[i] = e / denominator
        # output.append(e / denominator)
    return output


def compute_nn(x: NDArray[float32], network: NNWeightsNew) -> List[NDArray[float32]]:
    activations: List[NDArray[float32]] = [x]
    last_layer_outputs: NDArray[float32] = x
    for layer_num, layer in enumerate(network.layers):
        z: NDArray[float32] = (
            layer.weights.dot(last_layer_outputs) + layer.biases
        )  # type:ignore
        next_layer_outputs = (
            relu(z) if layer_num != len(network.layers) - 1 else softmax(z)
        )

        activations.append(next_layer_outputs)
        last_layer_outputs = next_layer_outputs
    return activations


def calc_mse_loss(predicted: NDArray[float32], target: NDArray[float32]) -> float32:
    assert len(predicted) == len(target)
    total = float32(0)
    for i in range(len(predicted)):
        total += np.square(predicted[i] - target[i])
    return total


def calc_cross_entropy_loss(
    predicted: NDArray[float32], target: NDArray[float32]
) -> float32:
    assert len(predicted) == len(target)
    return -np.sum(
        [
            target[i] * (np.log(predicted[i]) if predicted[i] != 0 else 0)
            for i in range(len(predicted))
        ]
    )


def random_weights_nn(data_size: int, layer_sizes: List[int]) -> NNWeightsNew:
    activation_size = data_size

    all_layers: List[Layer] = []
    for layer_num, layer_size in enumerate(layer_sizes):
        weights = np.random.normal(0, 0.01, (layer_size, activation_size))
        biases = np.random.normal(0, 0.01, layer_size)
        all_layers.append(Layer(weights, biases))
        activation_size = layer_size
    return NNWeightsNew(all_layers)
