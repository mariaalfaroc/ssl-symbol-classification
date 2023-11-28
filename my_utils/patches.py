import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from skimage.filters import threshold_sauvola
from skimage.measure import shannon_entropy
from torchvision.transforms import InterpolationMode

import datasets.config as config
from my_utils.parser import load_img_pages


def load_patches(
    ds_name: str,
    kernel: Tuple = (64, 64),
    stride: Tuple = (32, 32),
    entropy_threshold: float = 0.8,
    entropy_threshold_staff_lines: float = 0.1,
    use_remove_staff_lines: bool = False,
) -> torch.Tensor:
    patches = None
    if os.path.exists(config.patches_file):
        print(f"Loading patches from {config.patches_file}")
        patches = np.load(config.patches_file, allow_pickle=True)
        patches = torch.from_numpy(patches)
    else:
        print(f"Creating patches and loading them to {config.patches_file}")
        images = load_img_pages(ds_name=ds_name)
        patches = create_patches(
            images=images,
            kernel=kernel,
            stride=stride,
            entropy_threshold=entropy_threshold,
            entropy_threshold_staff_lines=entropy_threshold_staff_lines,
            use_remove_staff_lines=use_remove_staff_lines,
        )
        # Save them
        np.save(config.patches_file, patches.numpy())

    return patches


###################################################################### PATCHES EXTRACTION AND FILTERING:


def create_patches(
    images: torch.Tensor,
    kernel: Tuple = (64, 64),
    stride: Tuple = (32, 32),
    entropy_threshold: float = 0.8,
    entropy_threshold_staff_lines: float = 0.1,
    use_remove_staff_lines: bool = False,
) -> torch.Tensor:
    all_patches = []
    for i in tqdm.tqdm(images, position=0, leave=True):
        patches = extract_patches(i, kernel, stride)
        for p in patches:
            useful = filter_patch(p, entropy_threshold)
            if useful:
                if use_remove_staff_lines:
                    useful = remove_staff_lines(p, entropy_threshold_staff_lines)
                if useful:
                    p = transforms.Resize(
                        config.INPUT_SIZE, interpolation=InterpolationMode.BICUBIC
                    )(p)
                    all_patches.append(p)
            else:
                continue
    all_patches = torch.stack(all_patches)
    return all_patches


def extract_patches(
    x: torch.Tensor, kernel: Tuple = (64, 64), stride: Tuple = (32, 32)
) -> torch.Tensor:
    # x.shape -> [1, 3, height, width]
    x = x.unsqueeze(0)
    # We do not want to create slices across dimension 1,
    # since those are the channels of the image (RGB in this case)
    # and they need to be kept as they are for all patches
    # The patches are only created across the height and width of an image
    patches = x.unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1])
    # The resulting tensor will have size
    # [1, 3, num_vertical_slices, num_horizontal_slices, kernel[0], kernel[1]]
    # Reshape it to combine the slices to get a list of patches,
    # i.e., size of [num_patches, 3, kernel[0], kernel[1]]
    patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel[0], kernel[1])
    return patches


def remove_staff_lines(image: torch.Tensor, entropy_threshold: float = 0.1) -> bool:
    useful = False  # Output flag

    # 1) BINARIZE IMAGE
    image_ = 255 * image.permute(1, 2, 0).numpy()
    image_ = np.array(image_, dtype=np.uint8)
    image_ = cv2.cvtColor(cv2.cvtColor(image_, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image_, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 2) REMOVE STAFF LINES ON BINARIZED IMAGE
    # https://stackoverflow.com/questions/46274961/removing-horizontal-lines-in-image-opencv-python-matplotlib
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 3)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    result = 0 - cv2.morphologyEx(
        0 - thresh, cv2.MORPH_CLOSE, repair_kernel, iterations=1
    )
    # 3) FILTER BY ENTROPY
    entropy = shannon_entropy(result)
    if entropy > entropy_threshold:
        useful = True
    return useful


def filter_patch(image: torch.Tensor, entropy_threshold: float = 0.8) -> bool:
    useful = False  # Output flag

    # 1) BINARIZE IMAGE
    image_ = 255 * image.permute(1, 2, 0).numpy()
    image_ = np.array(image_, dtype=np.uint8)
    image_ = cv2.cvtColor(cv2.cvtColor(image_, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    thresh_sauvola = threshold_sauvola(image_, window_size=25)
    image_ = image_ > thresh_sauvola

    # 2) FILTER BY ENTROPY
    entropy = shannon_entropy(image_)
    if entropy > entropy_threshold:
        useful = True
    return useful
