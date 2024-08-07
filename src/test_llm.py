import pickle
from pathlib import Path
from typing import List

import torch

from src.data_loading import get_batch
from src.gpt2_weights import GPT2Weights
from src.neural_net import GPT2Model, TransformerBlock
from src.tokenization import GPT2Tokenizer
from src.torch_settings import get_device, ConditionalAutocast


def test_forward_backward(gpt2_tokenizer: GPT2Tokenizer, encoded_dataset: List[int]):
    if Path("data/gpt2_weights.pkl").is_file():
        weights: GPT2Weights = pickle.load(open("data/gpt2_weights.pkl", "rb"))
        weights = weights.to(get_device())
    else:
        raise RuntimeError("Run the load_gpt2_weights file first to create the weights")

    batch_size = 16
    block_size = 64
    emb_dimension = 768
    vocab_size = 50257
    n_heads = 12
    dropout = 0.0
    topk = 200
    weight_decay = 0.1
    step_size = 0.0001
    llm = GPT2Model(
        weights,
        batch_size,
        block_size,
        emb_dimension,
        vocab_size,
        n_heads,
        dropout,
        weight_decay,
        (0.9, 0.999),
    )
    first_encoding = torch.tensor(
        [
            gpt2_tokenizer.encode("I am very curious about"),
            gpt2_tokenizer.encode("Why do you always complain"),
        ],
        dtype=torch.long,
    ).to(get_device())

    new_tokens = llm.generate(first_encoding, 5, topk, temperature=0.001)
    results = []
    for i in range(new_tokens.shape[0]):
        results.append(gpt2_tokenizer.decode(new_tokens[i][:].tolist()))
    assert results == [
        "I am very curious about the nature of the relationship",
        "Why do you always complain about the lack of quality",
    ]

    print("First test successful.")

    torch.manual_seed(100)
    xs, ys = get_batch(
        torch.tensor(encoded_dataset, dtype=torch.int32).to(get_device()),
        block_size,
        batch_size,
    )
    _, loss1 = llm.forward(xs, ys)
    assert torch.isclose(loss1, torch.tensor(4.84), atol=2e-1)
    llm.backward(ys)
    llm.apply_gradient(step_size)
    llm.zero_gradients()
    _, loss2 = llm.forward(xs, ys)
    assert torch.isclose(loss2, torch.tensor(3.9), atol=2e-1)
    loss = 0
    for i in range(50):
        _, loss = llm.forward(xs, ys)
        llm.backward(ys)
        llm.apply_gradient(step_size)
        llm.zero_gradients()
    assert torch.isclose(loss, torch.tensor(0.007), atol=3e-3)
    print(f"Successfully optimized batch to 0. Initial losses {loss1.item():0.2f} {loss2.item():0.2f}")


def main():
    with torch.no_grad():
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.set_printoptions(precision=4)

        gpt2_tokenizer = GPT2Tokenizer()
        if Path("data/shakespeare.pkl").is_file():
            encoded_dataset = pickle.load(open("data/shakespeare.pkl", "rb"))
        else:
            with open("data/shakespeare.txt", "r") as f:
                text = f.read()

            encoded_dataset = gpt2_tokenizer.encode(text)
            pickle.dump(encoded_dataset, open("data/shakespeare.pkl", "wb"))
            decoded = gpt2_tokenizer.decode(
                encoded_dataset,
            )
            assert decoded == text

        compile_pytorch = True
        autocast = False

        with ConditionalAutocast(autocast):
            if compile_pytorch:
                GPT2Model.forward = torch.compile(GPT2Model.forward)
                TransformerBlock.backward = torch.compile(TransformerBlock.backward)
            test_forward_backward(gpt2_tokenizer, encoded_dataset)


if __name__ == "__main__":
    main()
