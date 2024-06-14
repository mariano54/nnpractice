import dataclasses
import time
from typing import List
from numpy.typing import NDArray

import numpy as np

float32 = np.float32


@dataclasses.dataclass
class LayerNode:
    weights: NDArray[float32]  # How much each prev layer node weighs in
    bias: float32


@dataclasses.dataclass
class LayerWeights:
    nodes: List[LayerNode]


@dataclasses.dataclass
class NNWeights:
    layers: List[LayerWeights]


def relu(x: float32) -> float32:
    if x < 0:
        return 0
    else:
        return x


def softmax(x: NDArray[float32]) -> List[float]:
    exps = [np.exp(x_i) for x_i in x]
    denominator = sum(exps)
    output = []
    for e in exps:
        output.append(e / denominator)
    return output


def compute_nn(x: NDArray[float32], weights: NNWeights) -> List[float]:
    last_layer_outputs: NDArray[float32] = x
    for layer_num, layer in enumerate(weights.layers):
        next_layer_outputs: List[float32] = []
        for node in layer.nodes:
            next_layer_outputs.append(relu(last_layer_outputs.dot(node.weights) + node.bias))
        last_layer_outputs = np.array(next_layer_outputs)
    return softmax(last_layer_outputs)


def calculate_cost(predicted: NDArray[float32], target: NDArray[float32]) -> float32:
    assert len(predicted) == len(target)
    total = float32(0)
    for i in range(len(predicted)):
        total += np.square(predicted[i] - target[i])
    return total


def random_weights_nn(data_size: int, layer_sizes: List[int]) -> NNWeights:
    activation_size = data_size

    all_layers: List[LayerWeights] = []
    for layer_num, layer_size in enumerate(layer_sizes):
        all_layernodes: List[LayerNode] = []
        for i in range(layer_size):
            weights = np.random.normal(0, 0.01, size=activation_size).astype(float32)
            all_layernodes.append(LayerNode(weights, float32(np.random.normal(0, 0.01))))
        all_layers.append(LayerWeights(all_layernodes))
        activation_size = layer_size
    return NNWeights(all_layers)


def gradient_descent(dataset: NDArray[NDArray[float32]]):
    for i in range(0, len(dataset), 1000):
        # iterate 1000 at a time
        pass

def main():
    with open("train-images-idx3-ubyte", "rb") as f:
        data = f.read()

    num_images = int.from_bytes(data[4:8], byteorder="big")
    rows = int.from_bytes(data[8:12], byteorder="big")
    cols = int.from_bytes(data[12:16], byteorder="big")

    total_bytes = 16 + num_images * rows * cols
    assert total_bytes == len(data)

    images_py = []
    for i in range(num_images):
        if i % 1000 == 0:
            print(i)
        start_index = 16 + i * (rows * cols)
        image_bytes = data[start_index:start_index + rows*cols]
        image_arr = [byte for byte in image_bytes]
        images_py.append(image_arr)
    print("processed images")
    images: NDArray[NDArray[float32]] = np.array(object=images_py)
    print("converted to np")
    initial_w = random_weights_nn(len(images[0]), [20, 25, 10])

    print("Starting SGD")
    all_results: List[NDArray[np.float64]] = []
    start_time = time.time()
    num_entries_per_sgd: int = 100
    for i in range(num_entries_per_sgd):
        print(f"Processed image {i}")
        res = compute_nn(images[0], initial_w)
        all_results.append(res)
    end_time = time.time()
    print(f"total time: {end_time - start_time}, per compute: {(end_time - start_time) / num_entries_per_sgd}")
    print("Last layer node0 weights", initial_w.layers[-1].nodes[0].weights)
    print("Res:", res)
    # print("Cost:", c)



if __name__ == '__main__':
    main()
