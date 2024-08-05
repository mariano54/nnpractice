import dataclasses
import math
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional
import argparse

from src.gpt2_weights import GPT2Weights
from torch.profiler import profile, record_function, ProfilerActivity

import torch
from src.torch_settings import ConditionalAutocast, device


from src.tokenization import GPT2Tokenizer

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


from src.neural_net import GPT2Model, device


def get_cosine_step_size(step_index: int, max_steps: int, max_step_size: float, warmup: int) -> float:
    # From nanoGPT (Andrej Karpathy)
    if step_index <= warmup:
        return max_step_size * (step_index + 1) / warmup
    if step_index > max_steps:
        return 0.1 * max_step_size

    proportion_to_end = (step_index - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * proportion_to_end))
    return 0.1 * max_step_size + coeff * (0.9 * max_step_size)


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
    weight_decay = 0
    # with ctx:
    llm = GPT2Model(weights, batch_size, block_size, emb_dimension, vocab_size, n_heads, weight_decay, dropout)
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
    max_step_size = 6e-4
    max_grad_norm = 1
    steps = 50
    warmup_steps = 10
    train_up_to = int(len(dataset) * 0.8)
    train_set, test_set = dataset[:train_up_to], dataset[train_up_to:]
    train_set = torch.tensor(train_set, dtype=torch.int32).to(device)
    test_set = torch.tensor(test_set, dtype=torch.int32).to(device)
    start_t0 = time.time()
    torch.manual_seed(101)
    activities = [ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    # with ctx:
    for i in range(steps):
        start_t = time.time()
        xs, ys = get_batch(train_set, llm.max_T, llm.max_B)
        batch_t = to_ms(time.time() - start_t)
        start_t_2 = time.time()
        _, loss = llm.forward(xs, ys)

        forward_t = to_ms(time.time() - start_t_2)
        start_t_2 = time.time()
        mem_before = int(torch.cuda.memory_allocated() / (1024 * 1024))
        # with profile(activities=activities, record_shapes=False) as prof:
        #     with record_function(" Backward pass"):
        llm.backward(ys)

        backward_t = to_ms(time.time() - start_t_2)
        start_t_2 = time.time()
        mem_after = int(torch.cuda.memory_allocated() / (1024 * 1024))
        norm = llm.get_grad_norm()
        norm_t = to_ms(time.time() - start_t_2)
        start_t_2 = time.time()
        curr_step_size = get_cosine_step_size(i, steps, max_step_size, warmup_steps)
        if norm > max_grad_norm:
            curr_step_size /=  (norm / max_grad_norm)
        llm.apply_gradient(curr_step_size)

        apply_t= to_ms(time.time() - start_t_2)
        tokens_ps = int((llm.max_B * llm.max_T) / (time.time() - start_t))
        print(f"Loss at {i}= {round(loss.item(), 4)}, dt={to_ms(time.time() - start_t)}  TPS: {tokens_ps}   mem: {mem_before},"
              f" {mem_after}, gb {batch_t} f {forward_t} b {backward_t} a {apply_t} n {norm_t} Norm: {norm}, lr {curr_step_size}")

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # quit()
        if i != 0 and i % 100 == 0:
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

    batch_size = 12
    block_size = 1024
    emb_dimension = 768
    vocab_size = 50304 # 50257
    n_heads = 12
    dropout = 0.0
    adam_betas = (0.9, 0.95)
    weight_decay = 0.1
    use_ctx = True
    local = True

    if local:
        batch_size = 4
        use_ctx = False

    with ConditionalAutocast(use_ctx):
        llm = GPT2Model(None, batch_size, block_size, emb_dimension, vocab_size, n_heads, dropout, weight_decay, adam_betas)
        train(llm, dataset=encoded_dataset)


if __name__ == "__main__":
    with torch.no_grad():
        main()
