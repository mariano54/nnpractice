import os
import pickle
import time

import torch
from datasets import load_dataset
import multiprocessing

from src.tokenization import GPT2Tokenizer

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

remote_name = "sample-10BT"

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

gpt2_tokenizer = GPT2Tokenizer()

# From Andrej Karpathy GPT-2 video
eot = 50256
max_shard_size = 100 * (1024)


def tokenize(doc) -> torch.tensor:
    tokens = [eot]
    tokens.extend(gpt2_tokenizer.encode(doc["text"]))
    return torch.tensor(tokens, dtype=torch.uint16)


remote_name = "small_shard"
nprocs = max(1, os.cpu_count() // 2)
shard_index = 0
chunksize = 32
max_shards = 10  # Set to non-zero to stop at a certain number of shards

with multiprocessing.Pool(nprocs) as pool:
    total_num_tokens = 0
    shard_contents = torch.zeros(max_shard_size)
    start_t = time.time()
    for tokens in pool.imap(tokenize, fw, chunksize=chunksize):
        if 0 < max_shards == shard_index:
            break
        if total_num_tokens + len(tokens) > max_shard_size:
            # Finished shard, write to disk
            space_left = max_shard_size - total_num_tokens
            shard_contents[-space_left:] = tokens[:space_left]
            val_or_train = "val" if shard_index == 0 else "train"
            print(f"Writing shard: {shard_index} time taken: {time.time() - start_t}")
            start_t = time.time()
            pickle.dump(
                shard_contents, open(f"data/{remote_name}_{val_or_train}_{shard_index:06d}.p", "wb")
            )
            total_num_tokens = len(tokens) - space_left
            shard_contents.zero_()
            shard_contents[:total_num_tokens] = tokens[space_left:]
            shard_index += 1
        else:
            # append to shard
            shard_contents[: len(tokens)] = tokens
            total_num_tokens += len(tokens)
            total = time.time() - start_t
    if total_num_tokens > 0:
        val_or_train = "val" if shard_index == 0 else "train"
        pickle.dump(shard_contents, open(f"data/{remote_name}_{val_or_train}_{shard_index:06d}.p", "wb"))
