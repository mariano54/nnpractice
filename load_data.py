from typing import Tuple, List

import numpy as np
from numpy._typing import NDArray
from numpy import float32

lookup_table = {}
for i in range(256):
    lookup_table[i] = i/256.0

def load_data(img_filename: str, label_filename: str) -> Tuple[List[NDArray[float32]], List[int]]:
    with open(img_filename, "rb") as f:
        data = f.read()

    with open(label_filename, "rb") as f:
        labels_data = f.read()

    num_labels = int.from_bytes(data[4:8], byteorder="big")
    labels: List[int] = []
    for i in range(num_labels):
        labels.append(labels_data[8 + i])

    num_images = int.from_bytes(data[4:8], byteorder="big")
    rows = int.from_bytes(data[8:12], byteorder="big")
    cols = int.from_bytes(data[12:16], byteorder="big")

    total_bytes = 16 + num_images * rows * cols
    assert total_bytes == len(data)
    assert num_labels == num_images

    images_py = []
    for i in range(num_images):
        if i % 10000 == 0 and i!=0:
            print(f"Loaded {i} images")
        start_index = 16 + i * (rows * cols)
        image_bytes = data[start_index : start_index + rows * cols]
        image_arr = np.array([lookup_table[byte] for byte in image_bytes])
        images_py.append(image_arr)
    print("processed images")
    return images_py, labels
