from load_data import load_data
import pickle

from neural_net import compute_nn


def test_model(weights_filename: str):
    images, labels = load_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

    print("Initializing weights")
    with open(weights_filename, "rb") as f:
        weights = pickle.loads(f.read())

    failure = 0
    for image, label in zip(images, labels):
        activations = compute_nn(image, weights)
        output = max((v, i) for i, v in enumerate(activations[-1]))[1]

        if output != label:
            failure += 1

    print(f"Error rate: {failure/len(images)}")


if __name__ == "__main__":
    test_model("weights.pkl")
