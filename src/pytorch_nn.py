import pickle
from os import abort

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from neural_net import NNWeightsTorch

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


class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("Intermediate output:")
        print(x.shape)
        print(x[0])
        # print()
        return x



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            # PrintLayer(),
            nn.LayerNorm(1000),
            # PrintLayer(),
            nn.ReLU(),
            # PrintLayer(),
            nn.Linear(1000, 10),
            # PrintLayer(),
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
    model.eval()
    print(test_data.data[0].tolist())
    normalized_test_data = test_data.data / 256
    # print(f"Len of test data: {len(test_data.data)}")
    outputs = model.forward(normalized_test_data.to(device)[:].float())
    fail = 0
    for img_i, output in enumerate(outputs):
        # print(f"Outputs: {outputs}")
        output_i = max((v, i) for i, v in enumerate(output))[1]
        if output_i != test_data[img_i][1]:
            fail += 1
    print(f"Fail rate: {fail/len(outputs)}")
    # quit()
    model.train()


def main():

    print("Initializing weights")
    with open("./weights/weights.pkl", "rb") as f:
        weights: NNWeightsTorch = pickle.loads(f.read())

    batch_size = 32
    model = NeuralNetwork().to(device)
    print("Prams length", len(list(model.parameters())))
    for i, params in enumerate(list(model.parameters())):
        if i == 0:
            params.data = weights.layers[0].weights.float().to(device)
        elif i == 1:
            params.data = weights.layers[0].biases.float().to(device)
        elif i == 2:
            params.data = weights.layers[0].batch_norm[0].float().to(device).sum(0)
        elif i == 3:
            params.data = weights.layers[0].batch_norm[1].float().to(device).sum(0)
        elif i == 4:
            params.data = weights.layers[1].weights.float().to(device)
        elif i == 5:
            params.data = weights.layers[1].biases.float().to(device)
        else:
            print("fail")
            abort()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 500
    train_data_cuda_2 = train_data_cuda / 256
    print(f"Gain: {list(model.parameters())[2][:10]}")
    print(f"Bias: {list(model.parameters())[3][:10]}")
    # print(f"l0 weights: {list(model.parameters())[0]}")

    # test_model(model)
    print("Starting training")
    for epoch in range(num_epochs):
        for batch_index in range(int(len(train_data_cuda_2) / batch_size)):
            inputs = train_data_cuda_2[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ].float()
            labels = train_labels_cuda[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            optimizer.zero_grad()

            # print("Inputs: ", inputs)
            # Forward pass
            outputs = model(inputs)
            # print(outputs)

            print("Output", model.softmax(outputs[0]))
            # Compute loss
            loss = criterion(outputs, labels)
            print("Loss", loss)

            # Backward pass and optimization
            loss.backward()
            for name, param in model.linear_relu_stack.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad[:10]}")

            # print("w1 grad", list(model.parameters())[0].grad.flatten().tolist()[:5])
            # # print("\n")
            # print("b1 grad", list(model.parameters())[1].grad.flatten().tolist()[:5])
            # # print("\n")
            # print("gain grad:", list(model.parameters())[2].grad.flatten().tolist()[:5])
            # print("bias grad", list(model.parameters())[3].grad.flatten().tolist()[:5])
            # # print("\n")
            # # print(
            #     "w1 grad", list(model.parameters())[0].grad.flatten().tolist()[400:420]
            # )
            # print("b1 grad", list(model.parameters())[1].grad.flatten().tolist()[:5])
            # print(
            #     "gain grad",
            #     list(model.parameters())[2].grad.shape,
            #     list(model.parameters())[2].grad.tolist()[:5],
            # )
            # print(
            #     "bias grad",
            #     list(model.parameters())[3].grad.shape,
            #     list(model.parameters())[3].grad.tolist()[:5],
            # )
            # print(f"RM: {model.linear_relu_stack[1].running_mean}")
            # print(f"RV: {model.linear_relu_stack[1].running_var}")
            # if batch_index == 10:
            #     quit()
            # quit()
            optimizer.step()
            print("new W", list(model.parameters())[0].flatten().tolist()[400:450])
            print("new bias", list(model.parameters())[1][0:10])
            print("new gain", list(model.parameters())[2][0:10])
            print("new lnbias", list(model.parameters())[3][0:10])

            if batch_index == 1:
                quit()

        test_model(model)
        # quit()
        # print("New W", list(model.parameters())[0].flatten().tolist()[400:450])

        # if epoch % 1 == 0:
        # print(f"Epoch {epoch} loss: {loss}")


if __name__ == "__main__":
    main()
