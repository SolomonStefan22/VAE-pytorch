import numpy as np
import torch


def load_images(path):
    with open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise ValueError(f"{path} nu e un fisier de imagini IDX")
    n = int.from_bytes(data[4:8], "big")
    pixels = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, 784)
    return torch.from_numpy(pixels.astype(np.float32) / 255.0)


def load_labels(path):
    with open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise ValueError(f"{path} nu e un fisier de etichete IDX")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return torch.from_numpy(labels.astype(np.int64))