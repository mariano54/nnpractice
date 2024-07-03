import time
from typing import List, Tuple, Optional
import torch
import pickle

from load_data import load_data
from neural_net import (
    random_weights_nn,
    compute_nn_pytorch,
    NNWeightsTorch,
    softmax_torch,
)
from test_model import test_model

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


def update_parameters(
    nerual_net: NNWeightsTorch,
    gradients: List[
        Tuple[
            torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
        ]
    ],
    step_size: torch.float32,
) -> None:
    for layer_i, layer in enumerate(nerual_net.layers):
        layer.weights -= gradients[layer_i][0].T * step_size
        layer.biases -= gradients[layer_i][1] * step_size
        if gradients[layer_i][2] is not None:
            assert gradients[layer_i][3] is not None
            new_gain = layer.batch_norm[0] - gradients[layer_i][2] * step_size
            new_bias = layer.batch_norm[1] - gradients[layer_i][3] * step_size
            layer.batch_norm = (new_gain, new_bias)


def backprop(
    preacts: List[torch.Tensor],
    all_activations: List[torch.Tensor],
    preact_list: List[Optional[torch.Tensor]],
    neural_net: NNWeightsTorch,
    labels: torch.Tensor,
    step_size: torch.float32,
) -> None:

    n = all_activations[-1].shape[0]
    d_logits = softmax_torch(preacts[-1])
    d_logits[range(n), labels] -= 1
    d_preact = d_logits / n

    # activations = act(z)  OR activations = act(batch_norm(z))
    all_gradients: List[
        Tuple[
            torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
        ]
    ] = []
    for layer_i in reversed(range(len(all_activations) - 1)):
        assert (preact_list[layer_i] is None) == (
            neural_net.layers[layer_i].batch_norm is None
        )
        if neural_net.layers[layer_i].batch_norm is not None:
            bnvar_inv: torch.Tensor = preact_list[layer_i][0]
            bnraw: torch.Tensor = preact_list[layer_i][1]

            # The following is taken from
            # github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipyn
            dz = (
                neural_net.layers[layer_i].batch_norm[0]
                * bnvar_inv
                / n
                * (
                    n * d_preact
                    - d_preact.sum(0)
                    # - n / (n - 1) * To align with pytorch, keep this commented
                    - bnraw * (d_preact * bnraw).sum(0)
                )
            )
            d_gain = (bnraw * d_preact).sum(0, keepdim=True)
            d_bias = d_preact.sum(0, keepdim=True)

        else:
            dz = d_preact
            d_gain = None
            d_bias = None

        activations = all_activations[layer_i]

        dw2 = activations.T @ dz
        db2 = dz.sum(0)

        all_gradients = [(dw2, db2, d_gain, d_bias)] + all_gradients
        if layer_i != 0:
            d_activations = dz @ neural_net.layers[layer_i].weights
            dpreact_template = torch.zeros_like(preacts[layer_i - 1])
            over_zero = torch.nonzero(preacts[layer_i - 1] > 0, as_tuple=False)
            dpreact_template[over_zero[:, 0], over_zero[:, 1]] = 1
            d_preact = dpreact_template * d_activations
    update_parameters(neural_net, all_gradients, step_size)


def train_model(
    dataset: List[torch.Tensor],
    labels: List[int],
    nn_torch: NNWeightsTorch,
    momentum=0.1,
):
    batch_size: int = 32
    labels_pytorch = torch.as_tensor(labels).to(device)
    step_size = 0.1
    num_passes = 40000
    nn_torch.layers[0].weights.requires_grad = False
    nn_torch.layers[0].biases.requires_grad = False
    nn_torch.layers[0].batch_norm[0].requires_grad = False
    nn_torch.layers[0].batch_norm[0].requires_grad = False
    nn_torch.layers[1].weights.requires_grad = False
    nn_torch.layers[1].biases.requires_grad = False
    test_model(nn_torch, device_i=device)

    for pass_i in range(num_passes):
        if pass_i == 10000:
            step_size = 0.01

        image_indexes = torch.randperm(len(dataset))[:batch_size]
        batch_labels_pytorch = labels_pytorch[image_indexes]

        data_input_torch = []
        for img_index in image_indexes:
            data_input_torch.append((dataset[img_index]).float().to(device))
        input_data = torch.stack(data_input_torch).to(device)
        preacts, activations_pytorch, preact_list, batch_means, batch_vars = (
            compute_nn_pytorch(input_data, nn_torch, True)
        )
        for layer_i, (batch_mean, batch_var) in enumerate(zip(batch_means, batch_vars)):
            if batch_mean is not None:
                layer = nn_torch.layers[layer_i]
                layer.running_mean = (1 - momentum) * layer.running_mean + (
                    momentum * batch_mean
                )
                layer.running_var = (1 - momentum) * layer.running_var + (
                    momentum * batch_var
                )
        backprop(
            preacts,
            activations_pytorch,
            preact_list,
            nn_torch,
            labels_pytorch[image_indexes],
            step_size,
        )
        if pass_i % 2000 == 0:
            batch_loss_2 = -torch.mean(
                torch.log(
                    activations_pytorch[-1][range(batch_size), batch_labels_pytorch]
                )
            )
            print(
                f"Pass: {pass_i + 1}/{num_passes} Processing batch {pass_i} with step size: {step_size}, "
                f"loss: {batch_loss_2 / batch_size}"
            )

            test_model(nn_torch, device_i=device)


def main():
    images, labels = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte")

    print("Initializing weights")

    # initial_w = random_weights_nn(
    #     len(images[0]),
    #     [(50, True), (10, False)],
    #     device,
    # )

    with open("weights.pkl", "rb") as f:
        initial_w = pickle.loads(f.read())

    train_model(images, labels, initial_w)

    with open("weights.pkl", "wb") as f:
        pickle.dump(initial_w, f)
    print("Wrote model to: weights.pkl")

    test_model(initial_w, device)


if __name__ == "__main__":
    main()
