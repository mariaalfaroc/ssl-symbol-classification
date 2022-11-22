import os, json, random
from typing import Tuple

import cv2
import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
from skimage.measure import shannon_entropy
from skimage.filters import threshold_sauvola
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image

import config
from augmentation import AugmentStage

def load_pages(filepaths: list) -> list:
    if "AidaMathB1" in str(config.base_dir):
        filepaths = get_aida_page_names()
        return preprocess_pages(filepaths=filepaths)
    return preprocess_pages(filepaths=filepaths)

def get_aida_page_names(num_pages: int = 500) -> list:
    json_file = list(config.json_dir.glob(f"*{config.json_extn}"))
    assert len(json_file) == 1, "There should be only ONE json file for the Aida Math dataset!"
    json_file = json_file[0]

    with open(json_file, "r") as json_file:
        data = json.load(json_file)

    images_names = []
    # NOTE: This is added to be able to fit it in memory (Turing's memory)
    # for sample in data:
    for page in range(num_pages):
        sample = data[page]
        images_names.append(f"{sample['filename'][:-4]}{config.image_extn}")
    return images_names

def preprocess_pages(filepaths: list) -> list:
    images = []
    for filepath in filepaths:
        image_path = config.images_dir / filepath
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            image = image / 255
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            images.append(image)
    return images

def extract_patches(x: torch.Tensor, kernel: Tuple = (64, 64), stride: Tuple = (32, 32)) -> torch.Tensor:
    # x.shape -> [1, 3, height, width]
    x =  x.unsqueeze(0)
    # We do not want to create slices across dimension 1,
    # since those are the channels of the image (RGB in this case)
    # and they need to be kept as they are for all patches
    # The patches are only created across the height and width of an image
    patches = x.unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1])
    # The resulting tensor will have size [1, 3, num_vertical_slices, num_horizontal_slices, kernel[0], kernel[1]]
    # Reshape it to combine the slices to get a list of patches i.e. size of [num_patches, 3, kernel[0], kernel[1]]
    patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel[0], kernel[1])
    return patches

def remove_lines(image: torch.Tensor) -> bool:
    # Output flag
    useful = False
    image_ = 255*image.permute(1, 2, 0).numpy()
    image_ = np.array(image_, dtype=np.uint8)
    image_ = cv2.cvtColor(cv2.cvtColor(image_, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image_, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # REMOVE STAFF LINES ON BINARIZED IMAGE
    # https://stackoverflow.com/questions/46274961/removing-horizontal-lines-in-image-opencv-python-matplotlib
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 3)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    result = 0 - cv2.morphologyEx(0 - thresh, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    # FILTER BY ENTROPY
    entropy = shannon_entropy(result)
    # print(entropy)
    # Threshold by trial and error
    if entropy > 0.1:
        useful = True
    return useful

def filter_patch(image: torch.Tensor) -> bool:
    # Output flag
    useful = False
    image_ = 255*image.permute(1, 2, 0).numpy()
    image_ = np.array(image_, dtype=np.uint8)
    image_ = cv2.cvtColor(cv2.cvtColor(image_, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    thresh_sauvola = threshold_sauvola(image_, window_size=25)
    image_ = image_ > thresh_sauvola
    entropy = shannon_entropy(image_)
    # print(entropy)
    # Threshold by trial and error
    if entropy > 0.8:
        useful = True
    return useful

def load_patches(patches_path: str, images: list, kernel: Tuple = (64, 64), stride: Tuple = (32, 32), use_remove_lines: bool = False) -> torch.Tensor:
    patches = None

    if os.path.isfile(patches_path):
        print(f"Loading patches from {patches_path}")
        patches = np.load(patches_path, allow_pickle=True)
        patches = torch.from_numpy(patches)
    else:
        print(f"Creating patches and loading them to {patches_path}")
        patches = create_patches(images=images, kernel=kernel, stride=stride, use_remove_lines=use_remove_lines)
        # Save them
        np.save(patches_path, patches.numpy())
    
    return patches

def create_patches(images: list, kernel: Tuple = (64, 64), stride: Tuple = (32, 32), use_remove_lines: bool = False) -> torch.Tensor:
    patches_acc = []
    for i in tqdm.tqdm(images, position=0, leave=True):
        patches = extract_patches(i, kernel, stride)
        for p in patches:
            if filter_patch(p):
                useful = True
                if use_remove_lines:
                    useful = remove_lines(p)
                if useful:
                    p = transforms.Resize(config.INPUT_SIZE, interpolation=InterpolationMode.BICUBIC)(p)
                    patches_acc.append(p)
            else:
                continue
    patches_acc = torch.stack(patches_acc)
    return patches_acc

def pretrain_data_generator(images: torch.Tensor, device: torch.device, batch_size: int = 32, add_crop: bool = True, crop_scale: Tuple = (0.5, 0.5)) -> Tuple[torch.Tensor, torch.Tensor]:
    augment = AugmentStage(add_crop=add_crop, crop_scale=crop_scale)
    augment.to(device)
    np.random.shuffle(images.numpy())
    size = len(images)
    start = 0
    while True:
        end = min(start + batch_size, size)
        xa = []
        xb = []
        if end - start > 1:
            for i in images[start:end]:
                i.to(device)
                xa.append(augment(i).cpu().detach())
                xb.append(augment(i).cpu().detach())
            yield torch.stack(xa).to(device), torch.stack(xb).to(device)
        if end == size:
            start = 0
            np.random.shuffle(images.numpy())
        else:
            start = end

if __name__ == "__main__":
    config.set_data_dirs(base_path="MTH1000")
    filepaths = [fname for fname in os.listdir(config.images_dir) if fname.endswith(config.image_extn)]
    images = load_pages(filepaths=filepaths)
    patches = load_patches(images = images, kernel= (64, 64), stride= (32, 32))
    train_size = patches.shape[0]
    print(f"Number of patches: {train_size}")
    save_image(patches[random.randint(0, train_size - 1)], "check_images/patch_sample.jpg")
    save_image(make_grid(patches[:20], nrow=4), "check_images/patches.jpg")
    # Generator
    gen = pretrain_data_generator(images=patches, device=torch.device("cpu"))
    xa, xb = next(gen)
    print(xa.shape, xb.shape)
    cv2.imwrite("check_images/xa0_p.jpg", cv2.cvtColor(255*xa[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite("check_images/xb0_p.jpg", cv2.cvtColor(255*xb[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
    save_image(xa[0], "check_images/xa0_p_torch.jpg")
    save_image(xb[0], "check_images/xb0_p_torch.jpg")
    save_image(make_grid(xa, nrow=4), "check_images/xa_p.jpg")
    save_image(make_grid(xb, nrow=4), "check_images/xb_p.jpg")
