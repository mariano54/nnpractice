from pathlib import Path
from typing import List

from src.neural_net import ModularNetwork, device
from src.tokenization import GPT2Tokenizer
import torch
import pickle


def get_batch(dataset: List[int], block_size: int, batch_size: int):
    xs = []
    ys = []
    for i in range(batch_size):
        index_start = torch.randint(0, len(dataset) - (block_size + 1), (1,))[0]
        data_slice = torch.tensor(dataset[index_start : index_start + block_size + 1])
        xs.append(data_slice[:-1])
        ys.append(data_slice[1:])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def main():
    torch.manual_seed(54)
    gpt2_tokenizer = GPT2Tokenizer()
    if Path("data/shakespeare.pkl").is_file():
        encoded_dataset = pickle.load(open("data/shakespeare.pkl", "rb"))
    else:
        with open("data/shakespeare.txt", "r") as f:
            text = f.read()

        encoded_dataset = gpt2_tokenizer.encode(text)
        pickle.dump(encoded_dataset, open("data/shakespeare.pkl", "wb"))
        decoded = gpt2_tokenizer.decode(encoded_dataset)
        assert decoded == text

    batch_size = 32
    block_size = 128  # change to 1024 later
    emb_dimension = 768
    vocab_size = 50256

    xs, ys = get_batch(encoded_dataset, block_size - 10, batch_size)
    print(xs.shape)

    llm = ModularNetwork(None, 0.1, batch_size, block_size, emb_dimension, vocab_size)
    all_probs = llm.forward(xs, ys)

    print("generating predictions...")
    xs, ys = get_batch(encoded_dataset, block_size, 2)
    new_xs = llm.generate(xs, 10)
    for i in range(new_xs.shape[0]):
        print("First prediction: ")
        print(gpt2_tokenizer.decode(new_xs[i][block_size - 10 :].tolist()))


if __name__ == "__main__":
    main()
