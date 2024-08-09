import math
import os
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import Callable

import torch

from src.data_loading import (
    get_filepaths,
    partition_filepaths,
    DataLoader,
    get_batch,
)
import torch.distributed as dist
import torch.multiprocessing as mp

from src.gpt2_weights import GPT2Weights
from src.tokenization import GPT2Tokenizer
from src.torch_settings import ConditionalAutocast, get_device
from src.neural_net import GPT2Model, TransformerBlock
from src import neural_net, torch_settings


def to_ms(secs: float) -> int:
    return int(secs * 1000)


def get_cosine_step_size(step_index: int, max_steps: int, max_step_size: float, warmup: int) -> float:
    # Based on nanoGPT (Andrej Karpathy)
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
    for i in range(100):
        xs, ys = get_batch(dataset, llm.max_T, llm.max_B)
        probs, loss = llm.forward(xs, ys)
        losses.append(loss * probs.shape[0] / llm.max_B)  # Normalize for the final batch (maybe smaller)

    return torch.tensor(losses).mean().item()


def train(
    llm: GPT2Model,
    dataset_name: str,
    rank: int,
    world_size: int,
    max_step_size: float,
    max_grad_norm: float,
    steps: int,
    warmup_steps: int,
    total_batch_size: int,
    validation_freq: int,
):
    gpt2_tokenizer = GPT2Tokenizer()
    train_paths = get_filepaths(dataset_name, "train")
    train_paths_for_rank = partition_filepaths(train_paths, world_size)[(rank - 1) % world_size]
    print(f"Rank {rank} training on files: {train_paths_for_rank}")
    data_loader = DataLoader(train_paths_for_rank)
    if rank == 0:
        validation_paths = get_filepaths(dataset_name, "val")
        val_data_loader = DataLoader(validation_paths)
    else:
        val_data_loader = None

    start_t0 = time.time()
    torch.manual_seed(102)
    assert total_batch_size % llm.max_B == 0
    num_mini_batches = int(total_batch_size / (llm.max_B * llm.max_T * world_size))
    dist_group = dist.new_group(list(range(world_size)))
    min_validation_loss = 99999

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
        total_loss /= num_mini_batches

        mem_after = int(torch.cuda.memory_allocated() / (1024 * 1024))
        llm.scale_gradients(1 / num_mini_batches)
        llm.synchronize_gradients(dist_group, world_size)
        dist.reduce(entries_processed, 0, op=dist.ReduceOp.SUM, group=dist_group)
        dist.reduce(total_loss, 0, op=dist.ReduceOp.SUM, group=dist_group)

        norm = llm.get_grad_norm()
        curr_step_size = get_cosine_step_size(step_i, steps, max_step_size, warmup_steps)
        scaling_factor = 1
        if norm > max_grad_norm:
            scaling_factor = norm / max_grad_norm
        llm.scale_gradients(1 / scaling_factor)

        llm.apply_gradient(curr_step_size)
        llm.zero_gradients()

        if rank == 0:
            total_loss /= world_size
            tokens_ps = int((entries_processed.item()) / (time.time() - start_t))
            print(
                f"Loss at {step_i}= {round(total_loss.item(), 4)}, dt={to_ms(time.time() - start_t)}  TPS: {tokens_ps} "
                f" {mem_after}, Norm: {norm:.2f} scaling {scaling_factor:.2f}, batches{num_mini_batches} lr {curr_step_size:.5f}"
            )
            if step_i != 0 and step_i % validation_freq == 0:
                train_loss = calculate_loss(llm, data_loader.curr_file)
                validation_loss = calculate_loss(llm, val_data_loader.curr_file)
                print(f"Training loss: {train_loss:05f}, Validation loss: {validation_loss:05f}")
                if validation_loss < min_validation_loss:
                    weights: GPT2Weights = llm.extract_weights()
                    if step_i % (2 * validation_freq) == 0:
                        path = Path(f"weights/trained_weights_0.pkl")
                    else:
                        path = Path(f"weights/trained_weights_1.pkl")
                    print(f"Writing to disk to {path}")
                    if path.exists():
                        path.unlink()
                    pickle.dump(weights, open(path, "wb"))
                    print(f"Wrote to disk.")
                xs = torch.tensor(
                    [gpt2_tokenizer.encode("Hi, I am a language model and") for _ in range(8)]
                ).to(get_device())
                gen = llm.generate(
                    xs,
                    10,
                    200,
                    0.8,
                )
                for i in range(gen.shape[0]):
                    print("\t", gpt2_tokenizer.decode(gen[i].tolist()))
    print(f"Training time: {time.time() - start_t0}")


def main(rank: int, world_size: int):
    if get_device() != "cpu":
        torch_settings.device = f"cuda:{rank}"
        neural_net.device = f"cuda:{rank}"
    print(f"Running main, rank {rank}, world size {world_size} on device: {get_device()}")

    # Below configuration taken from https://github.com/karpathy/nanoGPT and the GPT2 paper
    with torch.no_grad():
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.set_printoptions(precision=4)
        batch_size = 8  # Can increase this up to 64 or higher if it fits in GPU
        block_size = 1024
        weights_filename = "weights/trained_weights_1.pkl"

        if weights_filename is not None:
            path = Path(weights_filename)
            weights: GPT2Weights = pickle.load(open(path, "rb")).to(get_device())
        else:
            weights = None

        emb_dimension = 768
        vocab_size = 50304  # 50257, increased to 50304 since it's a nicer number
        n_heads = 12
        dropout = 0.0
        adam_betas = (0.9, 0.95)  # Based on GPT-2 paper
        weight_decay = 0.1
        dataset_name = "sample-10BT"  # Fineweb dataset
        compile_pytorch = True  # This takes a while, set to False for debugging
        max_grad_norm = 1  # Prevents bad batches from messing up the weights
        total_batch_size = 2**21  # Increased by 4x from GPT-2 for efficiency
        warmup_steps = int(715 / (total_batch_size / (2**19)))  # 375 million tokens
        steps = int(19073 / (total_batch_size / (2**19)))  # 10B / 2**19
        max_step_size = 6e-4 * math.sqrt(total_batch_size / (2**19))  # Step size must increase as well
        validation_freq = (
            100  # Steps, log the validation/train loss, write the weights, and generate a bit
        )

        local = sys.argv[1].lower()
        if local == "true":
            local = True
        else:
            local = False

        if local:
            batch_size = 8
            block_size = 64
            total_batch_size = 2**13
            compile_pytorch = False
            dataset_name = "small_shard"

        with ConditionalAutocast(not local):
            if compile_pytorch:
                GPT2Model.forward = torch.compile(GPT2Model.forward)
                TransformerBlock.backward = torch.compile(TransformerBlock.backward)

            llm = GPT2Model(
                weights,
                batch_size,
                block_size,
                emb_dimension,
                vocab_size,
                n_heads,
                dropout,
                weight_decay,
                adam_betas,
            )
            train(
                llm,
                dataset_name,
                rank,
                world_size,
                max_step_size,
                max_grad_norm,
                steps,
                warmup_steps,
                total_batch_size,
                validation_freq,
            )


def init_process(rank: int, size: int, fn: Callable, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)

    fn(rank, size)


if __name__ == "__main__":
    num_processes = torch.cuda.device_count()
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
