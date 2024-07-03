from load_data import load_data
import torch
import pickle

from neural_net import compute_nn_pytorch, NNWeightsTorch


def test_model_from_file(weights_filename: str, device_i: str):
    print("Initializing weights")
    with open(weights_filename, "rb") as f:
        weights = pickle.loads(f.read())
    test_model(weights, device_i)


def test_model(weights: NNWeightsTorch, device_i: str):
    images, labels = load_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    failure = 0
    batch_size = 32
    for i in range(0, len(images), batch_size):
        img_batch = torch.stack(images[i : i + batch_size]).to(device_i)
        labels_batch = torch.tensor(labels[i : i + batch_size]).to(device_i)
        real_batch_size = img_batch.shape[0]

        preacts, activations, _, _, _ = compute_nn_pytorch(
            img_batch, weights, training=False
        )

        predictions = torch.argmax(activations[-1], dim=1).to(device_i)
        pred_correct = torch.sum(predictions == labels_batch)
        failure += real_batch_size - pred_correct
    print(f"Error rate: {failure/len(images)}")


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")
    test_model_from_file("weights.pkl", device)
