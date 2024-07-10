from typing import List, Tuple, Optional
import torch
import pickle

from src.load_data import load_data
from src.neural_net import (
    NNWeightsTorch,
    ModularNetwork,
    device,
    random_weights_nn,
)
from test.test_MNIST_performance import test_model


def train_model(
    dataset: List[torch.Tensor],
    labels: List[int],
    momentum=0.1,
    num_passes=400000,
) -> ModularNetwork:
    torch.manual_seed(1338)
    initial_w = random_weights_nn(
        len(dataset[0]),
        [(1000, True), (50, True), (10, False)],
        device,
    )

    modular_network = ModularNetwork(initial_w, momentum)
    labels_pytorch = torch.as_tensor(labels).to(device)
    batch_size: int = 32
    step_size = 0.01

    torch.manual_seed(2000)
    for pass_i in range(num_passes):
        if pass_i == 30000:
            step_size = 0.005
        if pass_i == 70000:
            step_size = 0.001
        if pass_i == 100000:
            step_size = 0.0005

        image_indexes = torch.randperm(len(dataset))[:batch_size]
        batch_labels_pytorch = labels_pytorch[image_indexes]

        data_input_torch = []
        for img_index in image_indexes:
            data_input_torch.append((dataset[img_index]).float().to(device))
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
                f"loss: {batch_loss_2 / batch_size}"
            )

            test_model(modular_network, device_i=device)
    return modular_network


def main():
    images, labels = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte")

    print("Initializing weights")

    # initial_w = random_weights_nn(
    #     len(images[0]),
    #     [(200, True), (100, True), (10, False)],
    #     seed=12345,
    # )

    # with open("weights.pkl", "rb") as f:
    #     initial_w = pickle.loads(f.read())
    with torch.no_grad():
        trained_model = train_model(images, labels)

    # with open("../weights/weights.pkl", "wb") as f:
    #     pickle.dump(initial_w, f)
    # print("Wrote model to: weights.pkl")
    #
    # test_model(initial_w, device)


if __name__ == "__main__":
    main()
