from load_data import load_data
import torch
import pickle

from neural_net import compute_nn_pytorch


def test_model(weights_filename: str, device: str):
    images, labels = load_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

    print("Initializing weights")
    with open(weights_filename, "rb") as f:
        weights = pickle.loads(f.read())

    failure = 0
    batch_size = 32
    for i in range(0, len(images), batch_size):
        img_batch = torch.stack(images[i : i + batch_size]).to(device)
        labels_batch = torch.tensor(labels[i : i + batch_size]).to(device)
        real_batch_size = img_batch.shape[0]

        _, activations, _ = compute_nn_pytorch(img_batch, weights, device=device)

        predictions = torch.argmax(activations[-1], dim=1).to(device)
        pred_correct = torch.sum(predictions == labels_batch)
        failure += real_batch_size - pred_correct

    print(f"Error rate: {failure/len(images)}")


if __name__ == "__main__":
    test_model("weights.pkl")
