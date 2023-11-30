import numpy as np
import torch

import datasets.config as config
from my_utils.parser import parse_files
from my_utils.patches import load_patches
from my_utils.preprocessing import filter_by_occurrence, get_w2i_dictionary


def load_supervised_data(ds_name: str, min_occurence: int = 50) -> dict:
    # 1) Parse files
    X, Y = parse_files(ds_name=ds_name)

    # 2) Filter out samples with low occurrence
    X, Y = filter_by_occurrence(bboxes=X, labels=Y, min_occurence=min_occurence)

    # 3) Get w2i dictionary
    w2i = get_w2i_dictionary(tokens=Y)

    # 4) Preprocessing
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray([w2i[w] for w in Y], dtype=np.int64)

    return {"X": X, "Y": Y, "w2i": w2i}


def load_pretrain_data(
    ds_name: str,
    supervised: bool = False,
    num_random_patches: int = -1,
    kernel: tuple = (64, 64),
    stride: tuple = (32, 32),
    entropy_threshold: float = 0.8,
) -> torch.Tensor:
    # Get either bounding boxes or patches
    if supervised:
        # Bboxes
        X, _ = parse_files(ds_name=ds_name)
        X = np.asarray(X, dtype=np.int32)
        X = torch.from_numpy(X)
    else:
        # Patches
        patches_filepath = config.patches_dir / "patches_"
        patches_filepath += f"k{'x'.join(map(str, kernel))}_"
        patches_filepath += f"s{'x'.join(map(str, stride))}_"
        patches_filepath += f"et{entropy_threshold}.npy"
        X = load_patches(
            ds_name=ds_name,
            patches_filepath=str(patches_filepath),
            kernel=kernel,
            stride=stride,
            entropy_threshold=entropy_threshold,
        )
        if num_random_patches > 0:
            perm = torch.randperm(len(X))
            X = X[perm[:num_random_patches]]
    return X
