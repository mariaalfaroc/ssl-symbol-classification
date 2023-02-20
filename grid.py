import argparse
from random import randint

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision.transforms import InterpolationMode

import config
from data import *
from pretrain import str2bool

def parse_arguments():
    parser = argparse.ArgumentParser(description="Transformation grid image arguments")
    parser.add_argument("--ds_path", type=str, default="b-59-850", choices=["Egyptian", "MTH1000", "MTH1200", "TKH", "b-59-850", "b-3-28", "b-50-747", "b-53-781"], help="Dataset's path")
    parser.add_argument("--add_crop", type=str2bool, default="True", help="Whether to add 'RandomResizeCrop' transform")
    parser.add_argument("--crop_scale", nargs="+", type=float, default=(0.5, 0.5), help="Scale (h, w) for random crop with respect to input image")
    args = parser.parse_args()
    return args

def main():
    CHECK_DIR = "CHECK_IMAGES"
    os.makedirs(CHECK_DIR, exist_ok=True)

    args = parse_arguments()
    print(args)
    
    config.set_data_dirs(base_path=args.ds_path)
    filepaths = [fname for fname in os.listdir(config.images_dir) if fname.endswith(config.image_extn)]
    bboxes = []
    if "json" in config.json_extn:
        bboxes = parse_files_json(filepaths=filepaths)[0]
    else:
        bboxes = parse_files_txt(filepaths=filepaths)[0]
    img = bboxes[randint(0, len(bboxes) - 1)]
    img = preprocess_image(img)
    grid = [img]
    nrow = 5
    if args.add_crop:
        nrow += 1
        grid.append(transforms.RandomResizedCrop(config.INPUT_SIZE, scale=args.crop_scale, interpolation=InterpolationMode.BICUBIC)(img))
    grid.append(transforms.RandomHorizontalFlip(p=1)(img))
    grid.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(img))
    grid.append(transforms.RandomGrayscale(p=1)(img))
    grid.append(transforms.GaussianBlur(kernel_size=23)(img))
    grid = torch.stack(grid)
    save_image(make_grid(grid, nrow=nrow), f"{CHECK_DIR}/transformations.jpg")
    pass

if __name__ == "__main__":
    main()
