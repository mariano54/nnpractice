import pickle
from os import abort

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from neural_net import NNWeightsNew

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
# device = "cpu"
print(f"Using {device} device")
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True,
)

test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())
#
train_data_cuda = train_data.data.to(device)
train_labels_cuda = train_data.targets.to(device)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def forward_softmax(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return self.softmax(logits)


def test_model(model):

    outputs = model.forward(test_data.data.to(device)[:].float())
    fail = 0
    for img_i, output in enumerate(outputs):
        output_i = max((v, i) for i, v in enumerate(output))[1]
        if output_i != test_data[img_i][1]:
            fail += 1
    print(f"Fail rate: {fail/len(outputs)}")


def main():

    print("Initializing weights")
    with open("weights.pkl", "rb") as f:
        weights: NNWeightsNew = pickle.loads(f.read())

    batch_size = 1
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    model = NeuralNetwork().to(device)

    for i, params in enumerate(list(model.parameters())):
        if i == 0:
            params.data = torch.from_numpy(weights.layers[0].weights).float().to(device)
        elif i == 1:
            params.data = torch.from_numpy(weights.layers[0].biases).float().to(device)
        elif i == 2:
            params.data = torch.from_numpy(weights.layers[1].weights).float().to(device)
        elif i == 3:
            params.data = torch.from_numpy(weights.layers[1].biases).float().to(device)
        else:
            print("fail")
            abort()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005)
    num_epochs = 500

    for epoch in range(num_epochs):
        for batch_index in range(int(len(train_data_cuda) / batch_size)):
            inputs = train_data_cuda[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ].float()
            labels = train_labels_cuda[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            print("Output", model.softmax(outputs[0]))
            # Compute loss
            loss = criterion(outputs, labels)

            print("Loss", loss)
            # Backward pass and optimization
            loss.backward()

            print(list(model.parameters())[0].grad.flatten().tolist()[:200])
            print("\n")
            print(list(model.parameters())[1].grad.flatten().tolist()[:200])
            print("\n")
            print(list(model.parameters())[2].grad.flatten().tolist()[:200])
            print("\n")
            print(list(model.parameters())[3].grad.flatten().tolist()[:200])
            quit()
            optimizer.step()

        if epoch % 25 == 0:
            print(f"Epoch {epoch}")
            test_model(model)


if __name__ == "__main__":
    main()
