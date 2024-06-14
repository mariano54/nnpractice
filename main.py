import dataclasses
import time
from typing import List, Tuple

import numpy
from numpy.typing import NDArray

import numpy as np

float32 = np.float32


@dataclasses.dataclass
class Layer:
    weights: NDArray[float32]  # 2d Array of weights: output x input
    biases: NDArray[float32]  # added to the W x a


@dataclasses.dataclass
class NNWeightsNew:
    layers: List[Layer]


def relu(x: NDArray[float32]) -> NDArray[float32]:
    return np.maximum(x, 0)


def softmax(x: NDArray[float32]) -> NDArray[float32]:
    exps = [np.exp(x_i) for x_i in x]
    denominator = np.sum(exps)
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
    return np.sum(
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


def load_data() -> Tuple[List[NDArray[float32]], List[int]]:
    with open("train-images-idx3-ubyte", "rb") as f:
        data = f.read()

    with open("train-labels-idx1-ubyte", "rb") as f:
        labels_data = f.read()

    num_labels = int.from_bytes(data[4:8], byteorder="big")
    labels: List[int] = []
    for i in range(num_labels):
        labels.append(labels_data[8 + i])

    num_images = int.from_bytes(data[4:8], byteorder="big")
    rows = int.from_bytes(data[8:12], byteorder="big")
    cols = int.from_bytes(data[12:16], byteorder="big")

    total_bytes = 16 + num_images * rows * cols
    assert total_bytes == len(data)
    assert num_labels == num_images

    images_py = []
    for i in range(num_images):
        if i % 1000 == 0:
            print(i)
        start_index = 16 + i * (rows * cols)
        image_bytes = data[start_index : start_index + rows * cols]
        image_arr = np.array([byte for byte in image_bytes])
        images_py.append(image_arr)
    print("processed images")
    return images_py, labels


def apply_gradient(network: NNWeightsNew, deltas: NDArray[float32]):
    all_bias_deltas = []
    processed = 0
    for layer in reversed(network.layers):
        num_biases = len(layer.biases)
        bias_deltas = np.array(deltas[-(processed + num_biases + 1) : -(processed + 1)])
        all_bias_deltas.append(bias_deltas)
        processed += num_biases
    all_bias_deltas = list(reversed(all_bias_deltas))

    processed_w = 0
    for layer_i, layer in enumerate(network.layers):
        to_process = layer.weights.size
        flat_weight_deltas = np.array(
            deltas[processed_w : (processed_w + to_process)], dtype=float32
        )
        weight_deltas = flat_weight_deltas.reshape(layer.weights.shape)
        new_biases = layer.biases + all_bias_deltas[layer_i]
        new_weights = layer.weights + weight_deltas
        network.layers[layer_i] = Layer(weights=new_weights, biases=new_biases)
        processed_w += to_process
    assert processed + processed_w == 16485


def process_sample(
    sample: NDArray[float32], label: int, network: NNWeightsNew
) -> Tuple[float32, NDArray[float32]]:
    activations = compute_nn(sample, network)
    assert len(activations) == 4  # layers
    outputs = activations[-1]
    assert len(outputs) == 10
    y = np.zeros(len(outputs), dtype=float32)
    y[label] = 1.0

    loss = calc_cross_entropy_loss(outputs, y)

    all_W_deltas = []
    all_bias_deltas = []
    deltas = np.array([outputs[d] - y[d] for d in range(len(outputs))])

    flattenned_ordered_deltas: NDArray[float32] = np.array([], dtype=float32)

    for layer_index in [2, 1, 0]:
        activation_index = layer_index + 1  # Activation also includes Xs in there
        W_deltas: NDArray[float32] = np.zeros(
            shape=(
                len(activations[activation_index]),
                len(activations[activation_index - 1]),
            ),
            dtype=float32,
        )
        bias_deltas = np.zeros(
            shape=(len(activations[activation_index])),
        )
        new_deltas = np.zeros(
            shape=(len(activations[activation_index])),
        )
        for k in range(len(activations[activation_index])):
            if activation_index != 3:
                delta = deltas.dot(network.layers[layer_index + 1].weights[:, k]) * (
                    1 if activations[activation_index][k] > 0 else 0
                )
            else:
                delta = deltas[k]
            new_deltas[k] = delta
            for j in range(len(activations[activation_index - 1])):
                W_deltas[k, j] = delta * activations[activation_index - 1][j]
            bias_deltas[k] = delta
        all_W_deltas.append(W_deltas)
        all_bias_deltas.append(bias_deltas)
        deltas = new_deltas

    for W_delta_matrix in reversed(all_W_deltas):
        flattenned_ordered_deltas = numpy.concatenate(
            [flattenned_ordered_deltas, W_delta_matrix.flatten()]
        )
    for bias_d_vector in reversed(all_bias_deltas):
        flattenned_ordered_deltas = numpy.concatenate(
            [flattenned_ordered_deltas, bias_d_vector]
        )

    return (loss, flattenned_ordered_deltas)


def train_model(
    images: List[NDArray[float32]], labels: List[int], initial_w: NNWeightsNew
):
    print("Starting SGD")
    all_results: List[NDArray[np.float64]] = []
    start_time = time.time()
    batch_size: int = 100
    parameter_size = 784 * 20 + 20 * 25 + 25 * 10 + (20 + 25 + 10)
    ls = [initial_w.layers[l].weights.shape[0] for l in range(len(initial_w.layers))]
    parameter_size_2 = (
        (len(images[0]) + 1) * ls[0] + (ls[0] + 1) * ls[1] + (ls[1] + 1) * ls[2]
    )
    assert parameter_size == parameter_size_2

    cumulative_flattened_deltas: NDArray[float32] = np.zeros(
        parameter_size, dtype=float32
    )
    batch_loss = 0.0
    step_size = 0.001
    for batch in range(1, 50):
        for batch_index in range(batch_size):
            print(f"Processed image {batch_index}")
            loss, flattenned_ordered_deltas = process_sample(
                images[batch_index], labels[batch_index], initial_w
            )
            batch_loss += loss
            print("Loss:", loss)
            cumulative_flattened_deltas += flattenned_ordered_deltas

    print("batch loss", batch_loss)
    apply_gradient(initial_w, step_size * -1 * cumulative_flattened_deltas)

    new_batch_loss = 0.0
    for batch_index in range(batch_size):
        print(f"Processed image {batch_index}")
        loss, flattenned_ordered_deltas = process_sample(
            images[batch_index], labels[batch_index], initial_w
        )
        new_batch_loss += loss
        print("Loss:", loss)
        cumulative_flattened_deltas += flattenned_ordered_deltas
    print("batch loss", batch_loss, new_batch_loss)
    end_time = time.time()
    print(
        f"total time: {end_time - start_time}, per compute: {(end_time - start_time) / batch_size}"
    )


def main():
    images, labels = load_data()
    print("Initializing weights")
    initial_w = random_weights_nn(len(images[0]), [20, 25, 10])
    train_model(images, labels, initial_w)


if __name__ == "__main__":
    main()
