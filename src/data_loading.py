import pickle
from pathlib import Path
from typing import List, Tuple

import torch
from src.torch_settings import get_device


def get_batch(dataset: torch.tensor, block_size: int, batch_size: int):
    xs = []
    ys = []
    for i in range(batch_size):
        index_start = torch.randint(0, len(dataset) - (block_size + 1), (1,))[0]
        data_slice = dataset[index_start : index_start + block_size + 1].clone()

        xs.append(data_slice[:-1])
        ys.append(data_slice[1:])
    res = torch.stack(xs).to(get_device()), torch.stack(ys).to(get_device())
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
    return torch.stack(xs).to(get_device()), torch.stack(ys).to(get_device()), end - 1


def get_filepaths(dataset_name: str, train_or_val: str) -> List[Path]:
    assert train_or_val in ["train", "val"]
    if train_or_val == "val":
        return [Path(f"data/{dataset_name}_val_{str(0).zfill(6)}.p")]
    else:
        paths = []
        i = 1
        while True:
            p = Path(f"data/{dataset_name}_train_{str(i).zfill(6)}.p")
            if not p.exists():
                if len(paths) == 0:
                    raise RuntimeError(f"Error, file {p} not found")
                return paths
            paths.append(p)
            i += 1


def partition_filepaths(paths: List[Path], num_partitions: int) -> List[List[Path]]:
    partitions = [[] for _ in range(num_partitions)]
    for i in range(len(paths)):
        partitions[i % num_partitions].append(paths[i])
    return partitions


class DataLoader:
    def __init__(self, filenames: List[Path]):
        self.filenames = filenames
        self.curr_file_index = 0
        self.curr_file: torch.tensor = self.read_file(filenames[0])
        self.data_index = 0

    def read_file(self, filename: Path) -> torch.tensor:
        print(f"Reading file: {filename}...")
        return pickle.load(open(filename, "rb"))

    def get_batch(self, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for _ in range(batch_size):
            start = self.data_index
            end = min(start + block_size + 1, len(self.curr_file))
            data_slice = self.curr_file[start:end].clone()
            remaining_to_read = (block_size + 1) - (end - start)
            assert remaining_to_read >= 0
            if remaining_to_read > 0:
                # Reached the end of the file, so read a new file
                self.curr_file_index = (self.curr_file_index + 1) % len(self.filenames)
                self.curr_file = self.read_file(self.filenames[self.curr_file_index])
                data_slice = torch.cat([data_slice, self.curr_file[:remaining_to_read]])
                self.data_index = remaining_to_read
            else:
                self.data_index += block_size

            assert len(data_slice) == block_size + 1
            xs.append(data_slice[:-1])
            ys.append(data_slice[1:])
        return torch.stack(xs).to(get_device()), torch.stack(ys).to(get_device())
