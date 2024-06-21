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
    random_weights_nn,
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
    batch_size: int = 10000
    # parameter_size = 784 * 20 + 20 * 25 + 25 * 10 + (20 + 25 + 10)
    ls = [initial_w.layers[l].weights.shape[0] for l in range(len(initial_w.layers))]
    parameter_size_2 = (len(dataset[0]) + 1) * ls[0] + sum(
        [(ls[i] + 1) * ls[i + 1] for i in range(0, len(ls) - 1)]
    )

    step_size = 0.0005
    num_passes = 500
    for pass_i in range(num_passes):

        cumulative_flattened_deltas: NDArray[float32] = np.zeros(
            parameter_size_2, dtype=float32
        )
        for batch in range(int(len(dataset) / batch_size)):
            activations = []
            to_log = []
            start_time = time.time()
            for batch_index in range(batch_size):
                image_index = batch * batch_size + batch_index
                activations.append(compute_nn(dataset[image_index], initial_w))
                predicted_prob = activations[-1][-1][labels[image_index]]
                if predicted_prob == 0:
                    to_log.append(0.0001)
                else:
                    to_log.append(activations[-1][-1][labels[image_index]])

            batch_loss = -sum(np.log(np.array(to_log)))

            for batch_index in range(batch_size):
                image_index = batch * batch_size + batch_index

                flattenned_ordered_deltas = process_sample(
                    labels[image_index], initial_w, activations[batch_index]
                )

                cumulative_flattened_deltas += flattenned_ordered_deltas

            print(
                f"Pass: {pass_i + 1}/{num_passes} Processing batch {batch} with step size: {step_size}, "
                f"loss: {batch_loss/batch_size}"
            )
            apply_gradient(
                initial_w,
                (1.0 / batch_size) * -1 * cumulative_flattened_deltas,
                step_size=step_size,
            )
            if batch_loss < 0.5:
                step_size = 0.00004
            if batch_loss < 0.3:
                step_size = 0.00002
            if batch_loss < 0.2:
                step_size = 0.00001
            # step_size = batch_loss / 3000

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
