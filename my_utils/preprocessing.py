from collections import Counter
from operator import itemgetter
from typing import Tuple

import cv2
import numpy as np
import torch

import datasets.config as config


def preprocess_image(image: np.ndarray, resize: bool = True) -> torch.Tensor:
    if resize:
        image = cv2.resize(
            image, config.INPUT_SIZE, interpolation=cv2.INTER_AREA
        )  # Resize
    image = image / 255  # Normalize
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    return image


def get_w2i_dictionary(labels: list) -> dict:
    labels = sorted(set(labels))
    w2i = dict(zip(labels, range(len(labels))))
    return w2i


def filter_by_occurrence(
    bboxes: list, labels: list, min_noccurence: int = 10
) -> Tuple[list, list]:
    label_occurence_dict = Counter(labels)
    ids_kept = [
        id
        for id, label in enumerate(labels)
        if label_occurence_dict[label] >= min_noccurence
    ]
    filtered_bboxes = list(itemgetter(*ids_kept)(bboxes))
    filtered_labels = list(itemgetter(*ids_kept)(labels))
    print(f"Removing labels that appear less than {min_noccurence} times")
    print(f"Before: {len(labels)} samples, After: {len(filtered_labels)} samples")
    return filtered_bboxes, filtered_labels
