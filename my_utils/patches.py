import os

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
    patches_filepath: str,
    kernel: tuple = (64, 64),
    stride: tuple = (32, 32),
    entropy_threshold: float = 0.8,
) -> torch.Tensor:
    patches = None
    if os.path.exists(patches_filepath):
        print(f"Loading patches from {patches_filepath}")
        patches = np.load(patches_filepath, allow_pickle=True)
        patches = torch.from_numpy(patches)
    else:
        print(f"Creating patches and saving them to {patches_filepath}")
        images = load_img_pages()
        patches = create_patches(
            images=images,
            kernel=kernel,
            stride=stride,
            entropy_threshold=entropy_threshold,
        )
        # Save them
        np.save(patches_filepath, patches.numpy())

    return patches


###################################################################### PATCHES EXTRACTION AND FILTERING:


def create_patches(
    images: list,
    kernel: tuple = (64, 64),
    stride: tuple = (32, 32),
    entropy_threshold: float = 0.8,
) -> torch.Tensor:
    all_patches = []
    for i in tqdm.tqdm(images, position=0, leave=True):
        patches = extract_patches(torch.from_numpy(i), kernel, stride)
        for p in patches:
            useful = filter_patch(p, entropy_threshold)
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
    x: torch.Tensor, kernel: tuple = (64, 64), stride: tuple = (32, 32)
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
