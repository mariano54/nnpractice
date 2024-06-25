import time
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
    random_weights_nn, compute_nn_batch,
)
from test_model import test_model

float32 = np.float32


def apply_gradient(
    network: NNWeightsNew, deltas: NDArray[float32], step_size: float
) -> None:
    all_bias_deltas = []
    processed = 0

    for layer in reversed(network.layers):
        num_biases = len(layer.biases)
        bias_deltas = np.array(
            deltas[-(processed + num_biases) : len(deltas) - (processed)]
        )
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
        new_biases = layer.biases + step_size * all_bias_deltas[layer_i]
        new_weights = layer.weights + step_size * weight_deltas
        network.layers[layer_i] = Layer(weights=new_weights, biases=new_biases)
        processed_w += to_process


def process_sample(
    label: int, network: NNWeightsNew, activations: List[NDArray[float32]]
) -> NDArray[float32]:
    assert len(activations) == len(network.layers) + 1  # layers
    outputs = activations[-1]
    y = np.zeros(len(outputs), dtype=float32)
    y[label] = 1.0

    all_W_deltas = []
    all_bias_deltas = []
    deltas = np.array([outputs[d] - y[d] for d in range(len(outputs))])

    for layer_index in reversed(range(len(network.layers))):
        activation_index = layer_index + 1  # Activation also includes Xs in there
        if activation_index != len(network.layers):
            deltas = deltas @ network.layers[layer_index + 1].weights
            for k in range(len(activations[activation_index])):
                if activations[activation_index][k] <= 0:
                    deltas[k] = 0

        W_deltas = np.outer(deltas, activations[activation_index - 1])
        all_W_deltas.append(W_deltas.flatten())
        all_bias_deltas.append(deltas)
    flattenned_ordered_deltas: NDArray[float32] = numpy.concatenate(
        [w for w in reversed(all_W_deltas)] + list(reversed(all_bias_deltas))
    )
    return flattenned_ordered_deltas


def train_model(
    dataset: List[NDArray[float32]],
    labels: List[int],
    initial_w: NNWeightsNew,
):
    batch_size: int = 32
    # parameter_size = 784 * 20 + 20 * 25 + 25 * 10 + (20 + 25 + 10)
    ls = [initial_w.layers[l].weights.shape[0] for l in range(len(initial_w.layers))]
    parameter_size_2 = (len(dataset[0]) + 1) * ls[0] + sum(
        [(ls[i] + 1) * ls[i + 1] for i in range(0, len(ls) - 1)]
    )
    labels_np = np.array(labels)
    step_size = 0.1
    num_passes = 20000
    for pass_i in range(num_passes):
        if pass_i == 10000:
            step_size = 0.01

        cumulative_flattened_deltas: NDArray[float32] = np.zeros(
            parameter_size_2, dtype=float32
        )
        image_indexes = np.random.choice(len(dataset), batch_size, replace=False)
        data_input = []
        for img_index in image_indexes:
            data_input.append(dataset[img_index])
        activations = compute_nn_batch(np.array(data_input), initial_w)
        batch_labels = labels_np[image_indexes]
        batch_loss = -np.sum(np.log(activations[-1][np.arange(activations[-1].shape[0]), batch_labels]))

        for batch_index in range(batch_size):
            image_index = image_indexes[batch_index]
            sample_activation = [activations[i][batch_index] for i in range(len(activations))]
            flattenned_ordered_deltas = process_sample(
                labels[image_index], initial_w, sample_activation
            )

            cumulative_flattened_deltas += flattenned_ordered_deltas
        if pass_i % 1000 == 0:
            print(
                f"Pass: {pass_i + 1}/{num_passes} Processing batch {pass_i} with step size: {step_size}, "
                f"loss: {batch_loss/batch_size}"
            )
        apply_gradient(
            initial_w,
            (1.0 / batch_size) * -1 * cumulative_flattened_deltas,
            step_size=step_size,
        )
        # if batch_loss < 0.5:
        #     step_size = 0.00002
        # if batch_loss < 0.3:
        #     step_size = 0.00001
        # if batch_loss < 0.2:
        #     step_size = 0.000003
        # if batch_loss < 0.1:
        #     step_size = 0.000001
        # step_size = batch_loss / 3000
        if pass_i % 1000 == 0:
            with open("weights.pkl", "wb") as f:
                pickle.dump(initial_w, f)
            print("Wrote model to: weights.pkl")

            test_model("weights.pkl")


def main():
    images, labels = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte")

    print("Initializing weights")

    initial_w = random_weights_nn(len(images[0]), [50, 10])

    train_model(images, labels, initial_w)

    with open("weights.pkl", "wb") as f:
        pickle.dump(initial_w, f)
    print("Wrote model to: weights.pkl")

    test_model("weights.pkl")


if __name__ == "__main__":
    main()
