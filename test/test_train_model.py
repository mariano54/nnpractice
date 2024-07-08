import torch

from src.load_data import load_data
from src.neural_net import random_weights_nn
from src.train_model import device, train_model
from test.test_MNIST_performance import test_model
import pickle
def main():
    images, labels = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    initial_w = random_weights_nn(
        len(images[0]),
        [(100, True), (50, True), (10, False)],
        device,
        seed=1239
    )

    with open("../weights/weights_test_100_50_10k.pkl", "rb") as f:
        comparison_weights = pickle.loads(f.read())

    train_model(images, labels, initial_w, num_passes=10000)

    # with open("../weights/weights_test_100_50_10k.pkl", "wb") as f:
    #     pickle.dump(initial_w, f)
    # print("Wrote model to: weights.pkl")

    test_model(initial_w, device)

    assert torch.allclose(initial_w.layers[0].weights, comparison_weights.layers[0].weights)

if __name__ == "__main__":
    main()
