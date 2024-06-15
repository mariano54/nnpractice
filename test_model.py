from load_data import load_data
import pickle
from train_model import NNWeightsNew, Layer, compute_nn


def main():
    images, labels = load_data("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")

    print("Initializing weights")
    with open("weights.pkl", "rb") as f:
        weights = pickle.loads(f.read())

    failure = 0
    for image, label in zip(images, labels):
        activations = compute_nn(image, weights)
        output = max((v, i) for i, v in enumerate(activations[-1]))[1]

        if output != label:
            failure += 1

    print(f"Error rate: {failure/len(images)}")



if __name__ == "__main__":
    main()
