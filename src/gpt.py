import dataclasses
import math
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch
from src.gpt2_weights import GPT2Weights
from torch.profiler import ProfilerActivity
from src.torch_settings import ConditionalAutocast, device
from src.tokenization import GPT2Tokenizer
from src.neural_net import GPT2Model, device, TransformerBlock


def to_ms(secs: float) -> int:
    return int(secs * 1000)


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


def get_batch_consecutive(dataset: torch.tensor, block_size: int, batch_size: int, index_start: int):
    xs = []
    ys = []
    end = index_start
    for i in range(batch_size):
        start = index_start + i * block_size
        end = min(start + block_size + 1, len(dataset))
        data_slice = dataset[start:end].clone()
        if len(xs) > 0 and xs[0].shape[0] != data_slice.shape[0] - 1:
            break
        xs.append(data_slice[:-1])
        ys.append(data_slice[1:])
        if end > len(dataset) - 1:
            break
    return torch.stack(xs).to(device), torch.stack(ys).to(device), end - 1


def get_cosine_step_size(step_index: int, max_steps: int, max_step_size: float, warmup: int) -> float:
    # From nanoGPT (Andrej Karpathy)
    min_step_size_ratio = 0.1
    min_step_size = min_step_size_ratio * max_step_size
    if step_index <= warmup:
        return max_step_size * (step_index + 1) / warmup
    if step_index > max_steps:
        return min_step_size

    proportion_to_end = (step_index - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * proportion_to_end))
    return min_step_size + coeff * (min_step_size_ratio * max_step_size)


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
    weight_decay = 0.1
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
    ).to(device)

    new_tokens = llm.generate(first_encoding, 5, topk, temperature=0.001)
    results = []
    for i in range(new_tokens.shape[0]):
        results.append(gpt2_tokenizer.decode(new_tokens[i][:].tolist()))
    print(f"Results {results}")
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
    _, loss1 = llm.forward(xs, ys)
    assert torch.isclose(loss1, torch.tensor(4.84), atol=2e-1)
    llm.backward(ys)
    llm.apply_gradient(0.0001)
    llm.zero_gradients()
    _, loss2 = llm.forward(xs, ys)
    assert torch.isclose(loss2, torch.tensor(3.9), atol=2e-1)
    loss = 0
    for i in range(50):
        _, loss = llm.forward(xs, ys)
        llm.backward(ys)
        llm.apply_gradient(0.0001)
        llm.zero_gradients()
    assert torch.isclose(loss, torch.tensor(0.007), atol=3e-3)
    print(f"Successfully optimized batch to 0. Initial losses {loss1.item():0.2f} {loss2.item():0.2f}")


def calculate_loss(llm: GPT2Model, dataset: torch.tensor) -> float:
    # torch.manual_seed(100)
    losses = []
    index = 0
    while True:
        if index >= dataset.shape[0]:
            break
        xs, ys, index = get_batch_consecutive(dataset, llm.max_T, llm.max_B, index)
        probs, loss = llm.forward(xs, ys)
        losses.append(loss * probs.shape[0] / llm.max_B)  # Normalize for the final batch (maybe smaller)

    return torch.tensor(losses).mean().item()


def train(llm: GPT2Model, dataset: List[int]):
    max_step_size = 6e-4
    max_grad_norm = 1
    steps = 100
    warmup_steps = 10
    train_up_to = int(len(dataset) * 0.8)
    train_set, test_set = dataset[:train_up_to], dataset[train_up_to:]
    train_set = torch.tensor(train_set, dtype=torch.int32).to(device)
    test_set = torch.tensor(test_set, dtype=torch.int32).to(device)
    start_t0 = time.time()
    torch.manual_seed(102)
    dataset_index = 0
    total_batch_size = 64
    assert total_batch_size % llm.max_B == 0
    num_mini_batches = int(total_batch_size / llm.max_B)

    for step_i in range(steps):
        start_t = time.time()
        entries_processed = 0
        for mini_batch_i in range(num_mini_batches):
            if dataset_index == len(train_set) - 1:
                dataset_index = 0
            xs, ys, dataset_index = get_batch_consecutive(train_set, llm.max_T, llm.max_B, dataset_index)
            _, loss = llm.forward(xs, ys)
            entries_processed += xs.shape[0] * xs.shape[1]
            llm.backward(ys)

        mem_after = int(torch.cuda.memory_allocated() / (1024 * 1024))
        llm.scale_gradients(1 / num_mini_batches)
        norm = llm.get_grad_norm()
        curr_step_size = get_cosine_step_size(step_i, steps, max_step_size, warmup_steps)
        scaling_factor = 1
        if norm > max_grad_norm:
            scaling_factor = norm / max_grad_norm
        llm.scale_gradients(1 / scaling_factor)

        print("first positional emb grad", llm.dpos_embeddings[0][:5])
        llm.apply_gradient(curr_step_size)
        llm.zero_gradients()

        tokens_ps = int((entries_processed) / (time.time() - start_t))
        print(
            f"Loss at {step_i}= {round(loss.item(), 4)}, dt={to_ms(time.time() - start_t)}  TPS: {tokens_ps} "
            f" {mem_after}, Norm: {norm:.2f} scaling {scaling_factor:.2f}, batches{num_mini_batches} lr {curr_step_size:.5f}"
        )
        if step_i != 0 and step_i % 100 == 0:
            train_loss = calculate_loss(llm, train_set)
            test_loss = calculate_loss(llm, test_set)
            print(f"\nTraining loss: {train_loss}")
            print(f"Testing loss: {test_loss}\n")
    print(f"Training time: {time.time() - start_t0}")


def main():
    # Below configuration taken from https://github.com/karpathy/nanoGPT
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

    batch_size = 8
    block_size = 1024

    emb_dimension = 768
    vocab_size = 50304  # 50257
    n_heads = 12
    dropout = 0.0
    adam_betas = (0.9, 0.95)
    weight_decay = 0.1
    local = sys.argv[1].lower()
    if local == "true":
        local = True
    else:
        local = False
    print(f"Local: {local}")

    compile_pytorch = True

    if local:
        batch_size = 4
        block_size = 1024
        compile_pytorch = False

    with ConditionalAutocast(not local):
        if compile_pytorch:
            GPT2Model.forward = torch.compile(GPT2Model.forward)
            TransformerBlock.backward = torch.compile(TransformerBlock.backward)
        test_forward_backward(gpt2_tokenizer, encoded_dataset)

        llm = GPT2Model(
            None,
            batch_size,
            block_size,
            emb_dimension,
            vocab_size,
            n_heads,
            dropout,
            weight_decay,
            adam_betas,
        )
        train(llm, dataset=encoded_dataset)


if __name__ == "__main__":
    with torch.no_grad():
        main()
