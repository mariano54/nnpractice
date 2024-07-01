import time
from typing import List, Tuple
import torch
import pickle

from load_data import load_data
from neural_net import (
    random_weights_nn,
    compute_nn_pytorch,
    LayerTorch,
    NNWeightsTorch,
    softmax_torch,
)
from test_model import test_model

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
# device = "cpu"
print(f"Using {device} device")


def update_parameters(
    nerual_net: NNWeightsTorch,
    gradients: List[Tuple[torch.Tensor, torch.Tensor]],
    step_size: torch.float32,
) -> None:
    for layer_i, layer in enumerate(nerual_net.layers):
        layer.weights -= gradients[layer_i][0].T * step_size
        layer.biases -= gradients[layer_i][1] * step_size


def backprop(
    zs: List[torch.Tensor],
    activations: List[torch.Tensor],
    neural_net: NNWeightsTorch,
    labels: torch.Tensor,
) -> None:
    n = activations[-1].shape[0]
    dlogits = softmax_torch(zs[-1])
    dlogits[range(n), labels] -= 1
    dlogits /= n

    # TODO: generalize to more hidden layers
    activations_hidden = activations[-2]

    dactivations_hidden = dlogits @ neural_net.layers[-1].weights

    dw2 = activations_hidden.T @ dlogits

    db2 = dlogits.sum(0)

    dz = torch.zeros_like(zs[-2])

    over_zero = torch.nonzero(zs[-2] > 0, as_tuple=False)

    dz[over_zero[:, 0], over_zero[:, 1]] = 1

    dz = dz * dactivations_hidden

    xs = activations[0]

    dw1 = xs.T @ dz
    db1 = dz.sum(0)

    # assert db1.allclose(neural_net.layers[0].biases.grad.T)
    # assert dw1.allclose(neural_net.layers[0].weights.grad.T)

    gradients = [(dw1, db1), (dw2, db2)]
    update_parameters(neural_net, gradients, 0.1)


def train_model(
    dataset: List[torch.Tensor],
    labels: List[int],
    initial_w: NNWeightsTorch,
):
    batch_size: int = 32
    # parameter_size = 784 * 20 + 20 * 25 + 25 * 10 + (20 + 25 + 10)
    ls = [initial_w.layers[l].weights.shape[0] for l in range(len(initial_w.layers))]
    labels_pytorch = torch.as_tensor(labels).to(device)
    step_size = 0.1
    num_passes = 40000

    layers_torch = []
    for layer in initial_w.layers:
        w = layer.weights.to(device)
        b = layer.biases.to(device)
        w.requires_grad = False
        b.requires_grad = False
        layers_torch.append(LayerTorch(weights=w, biases=b))

    nn_torch = NNWeightsTorch(layers_torch)

    # start_t = time.time()
    for pass_i in range(num_passes):
        if pass_i == 10000:
            step_size = 0.01

        image_indexes = torch.randperm(len(dataset))[:batch_size]
        batch_labels_pytorch = labels_pytorch[image_indexes]

        data_input_torch = []
        for img_index in image_indexes:
            data_input_torch.append((dataset[img_index]).float().to(device))

        zs, activations_pytorch = compute_nn_pytorch(
            torch.stack(data_input_torch).to(device), nn_torch, device
        )

        batch_loss_2 = -torch.mean(
            torch.log(activations_pytorch[-1][range(batch_size), batch_labels_pytorch])
        )

        if pass_i % 250 == 0:
            print(
                f"Pass: {pass_i + 1}/{num_passes} Processing batch {pass_i} with step size: {step_size}, "
                f"loss: {batch_loss_2 / batch_size}"
            )

        backprop(zs, activations_pytorch, nn_torch, labels_pytorch[image_indexes])

        if pass_i % 1000 == 0:
            with open("weights.pkl", "wb") as f:
                pickle.dump(nn_torch, f)
            print("Wrote model to: weights.pkl")

            test_model("weights.pkl", device=device)


def main():
    images, labels = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte")

    print("Initializing weights")

    initial_w = random_weights_nn(len(images[0]), [50, 10], device)

    train_model(images, labels, initial_w)

    with open("weights.pkl", "wb") as f:
        pickle.dump(initial_w, f)
    print("Wrote model to: weights.pkl")

    test_model("weights.pkl", device)


if __name__ == "__main__":
    main()
