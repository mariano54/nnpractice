import math
import os
import signal
import sys
import time
from typing import Callable

import torch

from src.data_loading import (
    get_batch_consecutive,
    get_filepaths,
    partition_filepaths,
    DataLoader,
)
import torch.distributed as dist
import torch.multiprocessing as mp

from src.torch_settings import ConditionalAutocast, get_device
from src.neural_net import GPT2Model, TransformerBlock
from src import neural_net, torch_settings


def to_ms(secs: float) -> int:
    return int(secs * 1000)


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


def calculate_loss(llm: GPT2Model, dataset: torch.tensor) -> float:
    losses = []
    index = 0
    while True:
        if index >= dataset.shape[0]:
            break
        xs, ys, index = get_batch_consecutive(dataset, llm.max_T, llm.max_B, index)
        probs, loss = llm.forward(xs, ys)
        losses.append(loss * probs.shape[0] / llm.max_B)  # Normalize for the final batch (maybe smaller)

    return torch.tensor(losses).mean().item()


def train(llm: GPT2Model, dataset_name: str, rank: int, world_size: int):
    max_step_size = 6e-4
    max_grad_norm = 1
    steps = 19073  # 10B / 2**19
    warmup_steps = 715  # 375 million tokens
    train_paths = get_filepaths(dataset_name, "train")
    train_paths_for_rank = partition_filepaths(train_paths, world_size)[(rank - 1) % world_size]
    print(f"Rank {rank} training on files: {train_paths_for_rank}")
    data_loader = DataLoader(train_paths_for_rank)

    start_t0 = time.time()
    torch.manual_seed(102)
    total_batch_size = 2**22
    assert total_batch_size % llm.max_B == 0
    num_mini_batches = int(total_batch_size / (llm.max_B * llm.max_T * world_size))
    dist_group = dist.new_group(list(range(world_size)))

    print(
        f"Starting training run on {get_device()}, steps: {steps} w{warmup_steps}, max_ss: {max_step_size} max grad: {max_grad_norm} B: {llm.max_B} T: {llm.max_T} large batch {total_batch_size}"
    )

    for step_i in range(steps):
        start_t = time.time()
        entries_processed = torch.tensor([0], dtype=torch.float32).to(get_device())
        total_loss = torch.tensor([0], dtype=torch.float32).to(get_device())
        for mini_batch_i in range(num_mini_batches):
            xs, ys = data_loader.get_batch(llm.max_T, llm.max_B)
            _, loss = llm.forward(xs, ys)
            total_loss += loss
            entries_processed += xs.shape[0] * xs.shape[1]
            llm.backward(ys)
        tokens_ps = int((entries_processed.item()) / (time.time() - start_t))
        print(f"Rank {rank} finished in {to_ms(time.time() - start_t)} with TPS {tokens_ps}")
        total_loss /= num_mini_batches

        mem_after = int(torch.cuda.memory_allocated() / (1024 * 1024))
        llm.scale_gradients(1 / num_mini_batches)
        norm = llm.get_grad_norm()
        curr_step_size = get_cosine_step_size(step_i, steps, max_step_size, warmup_steps)
        scaling_factor = 1
        if norm > max_grad_norm:
            scaling_factor = norm / max_grad_norm
        llm.scale_gradients(1 / scaling_factor)
        llm.synchronize_gradients(dist_group, world_size)
        dist.reduce(entries_processed, 0, op=dist.ReduceOp.SUM, group=dist_group)
        dist.reduce(total_loss, 0, op=dist.ReduceOp.SUM, group=dist_group)

        llm.apply_gradient(curr_step_size)
        llm.zero_gradients()

        if rank == 0:
            total_loss /= world_size
            tokens_ps = int((entries_processed.item()) / (time.time() - start_t))
            print(
                f"Loss at {step_i}= {round(total_loss.item(), 4)}, dt={to_ms(time.time() - start_t)}  TPS: {tokens_ps} "
                f" {mem_after}, Norm: {norm:.2f} scaling {scaling_factor:.2f}, batches{num_mini_batches} lr {curr_step_size:.5f}"
            )
            if step_i != 0 and step_i % 100 == 0:
                # TODO: validation loss and train loss
                pass
                # train_loss = calculate_loss(llm, train_set)
                # test_loss = calculate_loss(llm, test_set)
                # print(f"\nTraining loss: {train_loss}")
                # print(f"Testing loss: {test_loss}\n")

    print(f"Training time: {time.time() - start_t0}")


def main(rank: int, world_size: int):
    # if rank == 0:
    #     torch_settings.device = f"cpu"
    #     neural_net.device = f"cpu"
    # else:
    #     torch_settings.device = "cuda"
    #     neural_net.device = f"cuda"

    if get_device() != "cpu":
        torch_settings.device = f"cuda:{rank}"
        neural_net.device = f"cuda:{rank}"
    print(f"Running main, rank {rank}, world size {world_size} on device: {get_device()}")

    # Below configuration taken from https://github.com/karpathy/nanoGPT
    with torch.no_grad():
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.set_printoptions(precision=4)
        batch_size = 8
        block_size = 1024

        emb_dimension = 768
        vocab_size = 50304  # 50257
        n_heads = 12
        dropout = 0.0
        adam_betas = (0.9, 0.95)
        weight_decay = 0.1
        dataset_name = "sample-10BT"
        compile_pytorch = True
        local = sys.argv[1].lower()
        if local == "true":
            local = True
        else:
            local = False

        if local:
            batch_size = 4
            block_size = 64
            compile_pytorch = False
            dataset_name = "small_shard"

        with ConditionalAutocast(not local):
            if compile_pytorch:
                GPT2Model.forward = torch.compile(GPT2Model.forward)
                TransformerBlock.backward = torch.compile(TransformerBlock.backward)

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
            train(llm, dataset_name=dataset_name, rank=rank, world_size=world_size)


def init_process(rank: int, size: int, fn: Callable, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)

    fn(rank, size)


if __name__ == "__main__":
    num_processes = torch.cuda.device_count()
    # num_processes = 2
    processes = []
    mp.set_start_method("spawn")

    for my_rank in range(num_processes):
        p = mp.Process(target=init_process, args=(my_rank, num_processes, main))
        p.start()
        processes.append(p)

    def signal_handler(sig, frame):
        print("You pressed Ctrl+C!, closing processes..")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
    for p in processes:
        p.join()
