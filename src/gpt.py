import dataclasses
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

from src.gpt2_weights import GPT2Weights

import torch

device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
print(f"PTDtype: {ptdtype}")
ctx = (
    nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
from src.tokenization import GPT2Tokenizer


def get_batch(dataset: torch.tensor, block_size: int, batch_size: int):
    xs = []
    ys = []
    for i in range(batch_size):
        index_start = torch.randint(0, len(dataset) - (block_size + 1), (1,))[0]
        data_slice = dataset[index_start : index_start + block_size + 1].clone()

        xs.append(data_slice[:-1])
        ys.append(data_slice[1:])
    res = torch.stack(xs).to(device), torch.stack(ys).to(device)
    return res


from src.neural_net import GPT2Model, device


def get_batch_consecutive(dataset: torch.tensor, block_size: int, batch_size: int, index_start: int):
    xs = []
    ys = []
    for i in range(batch_size):
        start = index_start + i * block_size
        end = min(start + block_size + 1, len(dataset) - 1)
        data_slice = dataset[start:end].clone()
        xs.append(data_slice[:-1])
        ys.append(data_slice[1:])
        if end > len(dataset) - 1:
            break
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def test_forward_backward(gpt2_tokenizer: GPT2Tokenizer, encoded_dataset: List[int]):
    if Path("data/gpt2_weights.pkl").is_file():
        weights: GPT2Weights = pickle.load(open("data/gpt2_weights.pkl", "rb"))
        weights = weights.to(device)
    else:
        raise RuntimeError("Run the load_gpt2_weights file first to create the weights")

    batch_size = 16
    block_size = 64
    emb_dimension = 768
    vocab_size = 50257
    n_heads = 12
    dropout = 0.0
    topk = 200
    # with ctx:
    llm = GPT2Model(weights, batch_size, block_size, emb_dimension, vocab_size, n_heads, dropout)
    first_encoding = torch.tensor(
        [
            gpt2_tokenizer.encode("I am very curious about"),
            gpt2_tokenizer.encode("Why do you always complain"),
        ],
        dtype=torch.long,
    ).to(device)

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
        torch.tensor(encoded_dataset, dtype=torch.int32).to(device),
        block_size,
        batch_size,
    )
    _, loss = llm.forward(xs, ys)
    print(f"Loss:", loss.item())
    assert torch.isclose(loss, torch.tensor(4.92744), atol=3e-1)
    llm.backward(ys)
    llm.apply_gradient(0.001)
    _, loss = llm.forward(xs, ys)
    print(f"Loss:", loss.item())
    assert torch.isclose(loss, torch.tensor(4.726), atol=3e-1)


def calculate_loss(llm: GPT2Model, dataset: torch.tensor) -> float:
    torch.manual_seed(100)
    losses = []
    for _ in range(10):
        random_index = torch.randint(0, len(dataset) - (llm.max_B * llm.max_T) - 1, (1,)).item()
        xs, ys = get_batch_consecutive(dataset, llm.max_T, llm.max_B, random_index)
        probs, loss = llm.forward(xs, ys)
        losses.append(loss * probs.shape[0] / llm.max_B)  # Normalize for the final batch (maybe smaller)
    return torch.tensor(losses).mean().item()


def train(llm: GPT2Model, dataset: List[int]):
    step_size = 3e-4
    train_up_to = int(len(dataset) * 0.8)
    train_set, test_set = dataset[:train_up_to], dataset[train_up_to:]
    train_set = torch.tensor(train_set, dtype=torch.int32).to(device)
    test_set = torch.tensor(test_set, dtype=torch.int32).to(device)
    start_t0 = time.time()
    torch.manual_seed(101)
    for i in range(50):
        start_t = time.time()
        xs, ys = get_batch(train_set, llm.max_T, llm.max_B)
        _, loss = llm.forward(xs, ys)
        mem_before = int(torch.cuda.memory_allocated() / (1024 * 1024))
        llm.backward(ys)
        mem_after = int(torch.cuda.memory_allocated() / (1024 * 1024))
        llm.apply_gradient(step_size)
        tokens_ps = int((llm.max_B * llm.max_T) / (time.time() - start_t))
        print(
            f"Loss at {i}= {round(loss.item(), 4)}, dt={int(1000 * (time.time() - start_t))}  TPS: {tokens_ps
            }   mem: {mem_before}, {mem_after}"
        )

        if i != 0 and i % 100 == 0:
            train_loss = calculate_loss(llm, train_set)
            test_loss = calculate_loss(llm, test_set)
            print(f"\nTraining loss: {train_loss}")
            print(f"Testing loss: {test_loss}\n")
    torch.cuda.synchronize()
    print(f"Training time: {time.time() - start_t0}")


def main():
    # Below configuration taken from https://github.com/karpathy/nanoGPT
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=4)

    # with ctx:
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
    test_forward_backward(gpt2_tokenizer, encoded_dataset)

    batch_size = 8
    block_size = 1024
    emb_dimension = 768
    vocab_size = 50257
    n_heads = 12
    dropout = 0.0
    adam_betas = (0.9, 0.999)
    # adam_betas = None

    # with ctx:
    llm = GPT2Model(None, batch_size, block_size, emb_dimension, vocab_size, n_heads, dropout, adam_betas)
    train(llm, dataset=encoded_dataset)

    # first_encoding = torch.tensor(
    #     [
    #         gpt2_tokenizer.encode("I am very curious about"),
    #         gpt2_tokenizer.encode("Why do you always complain"),
    #     ],
    #     dtype=torch.long,
    # )
    # new_tokens = llm.generate(first_encoding, 5, topk, temperature=0.001)
    # results = []
    # for i in range(new_tokens.shape[0]):
    #     results.append(gpt2_tokenizer.decode(new_tokens[i][:].tolist()))

    # xs, ys = get_batch(encoded_dataset, current_context, 2)
    # new_xs = llm.generate(xs, 10)
    # for i in range(new_xs.shape[0]):
    #     print("First prediction: ")
    #     print(gpt2_tokenizer.decode(new_xs[i][current_context- 10:].tolist()))
    #


if __name__ == "__main__":
    with torch.no_grad():
        # with ctx:
        main()
