from typing import List, Tuple
import numpy
from numpy.typing import NDArray
import numpy as np
import pickle

from load_data import load_data
from neural_net import (
    NNWeightsNew,
    Layer,
    compute_nn,
    calc_cross_entropy_loss,
    random_weights_nn,
)
from test_model import test_model

float32 = np.float32


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


def process_sample(
    sample: NDArray[float32], label: int, network: NNWeightsNew
) -> Tuple[float32, NDArray[float32]]:
    # print("\n\n\n")
    # start_time = time.time()
    activations = compute_nn(sample, network)
    assert len(activations) == len(network.layers) + 1  # layers
    outputs = activations[-1]
    assert len(outputs) == 10
    # print("1", 10000*(time.time() - start_time))
    # start_time = time.time()
    y = np.zeros(len(outputs), dtype=float32)
    y[label] = 1.0
    # print(f"correct: {label}")
    loss = calc_cross_entropy_loss(outputs, y)

    all_W_deltas = []
    all_bias_deltas = []
    deltas = np.array([outputs[d] - y[d] for d in range(len(outputs))])
    flattenned_ordered_deltas: NDArray[float32] = np.array([], dtype=float32)

    for layer_index in reversed(range(len(network.layers))):
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
            if activation_index != len(network.layers):
                delta = deltas.dot(network.layers[layer_index + 1].weights[:, k]) * (
                    1 if activations[activation_index][k] > 0 else 0
                )
            else:
                delta = deltas[k]

            new_deltas[k] = delta
            W_deltas[k, :] = delta * activations[activation_index - 1]
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
    batch_size: int = 1000
    # parameter_size = 784 * 20 + 20 * 25 + 25 * 10 + (20 + 25 + 10)
    ls = [initial_w.layers[l].weights.shape[0] for l in range(len(initial_w.layers))]
    parameter_size_2 = (len(images[0]) + 1) * ls[0] + sum(
        [(ls[i] + 1) * ls[i + 1] for i in range(0, len(ls) - 1)]
    )
    # assert parameter_size == parameter_size_2

    cumulative_flattened_deltas: NDArray[float32] = np.zeros(
        parameter_size_2, dtype=float32
    )
    step_size = 0.00001
    num_passes = 5
    for pass_i in range(num_passes):
        for batch in range(int(len(images) / batch_size)):
            batch_loss = 0.0
            print(
                f"Pass: {pass_i+1}/{num_passes} Processing batch {batch} with step size: {step_size}"
            )
            for batch_index in range(batch_size):
                image_index = batch * batch_size + batch_index

                loss, flattenned_ordered_deltas = process_sample(
                    images[image_index], labels[image_index], initial_w
                )

                batch_loss += loss
                cumulative_flattened_deltas += flattenned_ordered_deltas

            apply_gradient(
                initial_w, (step_size / batch_size) * -1 * cumulative_flattened_deltas
            )
            step_size = batch_loss / 100000000


def main():
    images, labels = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    print("Initializing weights")
    initial_w = random_weights_nn(len(images[0]), [32, 10])

    train_model(images, labels, initial_w)

    with open("weights.pkl", "wb") as f:
        pickle.dump(initial_w, f)
    print("Wrote model to: weights.pkl")

    test_model("weights.pkl")


if __name__ == "__main__":
    main()
