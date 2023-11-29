import random
from typing import Tuple

import numpy as np
import torch
from sklearn.utils import shuffle

from network.augmentation import AugmentStage


def pretrain_data_generator(
    images: torch.Tensor,  # bboxes or patches
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    random.shuffle(images)

    augment = AugmentStage()
    augment = augment.to(device)

    size = len(images)
    start = 0
    while True:
        end = min(start + batch_size, size)
        if end - start > 1:
            x = images[start:end].to(device)
            xa = augment(x)
            xb = augment(x)
            yield xa, xb

        if end == size:
            start = 0
            random.shuffle(images)
        else:
            start = end


def supervised_data_generator(
    images: np.ndarray, labels: np.ndarray, device: torch.device, batch_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = shuffle(images, labels, random_state=1)

    size = len(images)
    start = 0
    while True:
        end = min(start + batch_size, size)
        xi = torch.from_numpy(images[start:end])
        yi = torch.from_numpy(labels[start:end])
        yield xi.to(device), yi.to(device)

        if end == size:
            start = 0
            images, labels = shuffle(images, labels, random_state=1)
        else:
            start = end
