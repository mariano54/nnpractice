from typing import List, Tuple, Optional
import torch
import pickle

from src.load_data import load_data
from src.neural_net import (
    GPT2Model,
    device,
    random_weights_nn,
)
from test.test_MNIST_performance import test_model


def train_model(
    dataset: List[torch.Tensor],
    labels: List[int],
    momentum=0.1,
    num_passes=400000,
    seed: int = 1338,
) -> GPT2Model:

    initial_w = random_weights_nn(
        dataset[0].shape,
        [(1000, True), (10, False)],
        device,
    )
    #
    # with open("./weights/weights.pkl", "wb") as f:
    #     pickle.dump(initial_w, f)
    # print("Wrote model to: weights.pkl")

    # with open("./weights/weights.pkl", "rb") as f:
    #     initial_w = pickle.loads(f.read())

    # initial_w.layers[0].weights.requires_grad = True
    # initial_w.layers[0].biases.requires_grad = True
    # initial_w.layers[0].batch_norm[0].requires_grad = True
    # initial_w.layers[0].batch_norm[1].requires_grad = True
    # initial_w.layers[1].weights.requires_grad = True
    # initial_w.layers[1].biases.requires_grad = True
    # print(initial_w.layers[0].batch)
    modular_network = GPT2Model(None, momentum)
    labels_pytorch = torch.as_tensor(labels).to(device)
    batch_size: int = 32
    step_size = 0.1

    # print("original bn gain", modular_network.layers[1].bn_gain[:10])
    for pass_i in range(num_passes):
        if pass_i == 40000:
            step_size = 0.01
        if pass_i == 80000:
            step_size = 0.005
        if pass_i == 120000:
            step_size = 0.001

        image_indexes = torch.randperm(len(dataset))[:batch_size]
        # image_indexes = torch.arange(32*pass_i, 32*(pass_i + 1))
        batch_labels_pytorch = labels_pytorch[image_indexes]

        data_input_torch = []
        for img_index in image_indexes:
            data_input_torch.append((dataset[img_index]).float().to(device))
        # for d in data_input_torch:
        #     d.requires_grad = True
        input_data = torch.stack(data_input_torch).to(device)

        probs = modular_network.forward(input_data)

        modular_network.backward(batch_labels_pytorch)
        modular_network.apply_gradient(step_size)

        if pass_i % 2000 == 0:
            batch_loss_2 = -torch.mean(
                torch.log(probs[range(batch_size), batch_labels_pytorch])
            )
            print(
                f"Pass: {pass_i + 1}/{num_passes} Processing batch {pass_i} with step size: {step_size}, "
                f"loss: {batch_loss_2}"
            )

            test_model(modular_network, device_i=device)
        return modular_network
