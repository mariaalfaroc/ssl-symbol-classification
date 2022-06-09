import argparse, os, gc, random

import tqdm
import torch
import numpy as np
import pandas as pd

import config
from model import VICReg
from loss import vicreg_loss
from data import parse_files_json, parse_files_txt, pretrain_data_generator
from patches import load_pages, load_patches, pretrain_data_generator as patches_generator

def str2bool(v: str) -> bool:
    if v == "True":
        return True
    elif v == "False":
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="VICReg pretraining arguments")
    parser.add_argument("--ds_path", type=str, default="b-59-850", choices=["MTH1000", "MTH1200", "TKH", "b-59-850", "b-3-28", "b-50-747", "b-53-781"], help="Dataset's path")
    parser.add_argument("--crops_labelled", type=str2bool, default="True", help="Whether to use perfectly cropped symbol images")
    parser.add_argument("--add_crop", type=str2bool, default="True", help="Use RandomResizedCrop transform in the transform chain")
    parser.add_argument("--crop_scale", nargs="+", type=float, default=(0.5, 0.5), help="Crop scale for the RandomResizedCrop transform in the transform chain")
    parser.add_argument("--kernel", nargs="+", type=int, default=(64, 64), help="Size (height, width) of the patches")
    parser.add_argument("--stride", nargs="+", type=int, default=(32, 32), help="Size (height, width) of the stride used when creating patches")
    parser.add_argument("--remove_lines", type=str2bool, default="False", help="Whether to use a second filter that removes staff lines when creating patches")
    parser.add_argument("--encoder_features", type=int, default=1600, help="Encoder features dimension")
    parser.add_argument("--expander_features", type=int, default=1024, help="Expander features dimension")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--patience", type=int, default=150, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sim_loss_weight", type=float, default=1.0, help="Weight applied to the invariance loss")
    parser.add_argument("--var_loss_weight", type=float, default=1.0, help="Weight applied to the variance loss")
    parser.add_argument("--cov_loss_weight", type=float, default=1.0, help="Weight applied to the covariance loss")
    args = parser.parse_args()
    return args

def main():
    gc.collect()
    torch.cuda.empty_cache()

    # Run on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Device {device}")

    # Seed
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    args = parse_arguments()
    # Print experiment details
    print("Pre-training (VICReg approach) experiment")
    print(args)

    # Data
    config.set_data_dirs(base_path=args.ds_path)
    print(f"Data used {config.base_dir.stem}")
    filepaths = [fname for fname in os.listdir(config.images_dir) if fname.endswith(config.image_extn)]
    print(f"Number of pages: {len(filepaths)}")
    images = None
    size = None
    gen = None
    name = f"Labelled_{args.crops_labelled}_"
    if args.crops_labelled == True:
        # Perfectly cropped images
        if 'json' in config.json_extn:
            images = parse_files_json(filepaths=filepaths)[0]
        else:
            images = parse_files_txt(filepaths=filepaths)[0]
        size = len(images)
        print(f"Number of labelled patches: {size}")
        gen = pretrain_data_generator(images=images, device=device, batch_size=args.batch_size, add_crop=args.add_crop, crop_scale=args.crop_scale)
    else:
        # Patches
        images = load_pages(filepaths=filepaths)
        patches = load_patches(images=images, kernel=args.kernel, stride=args.stride, use_remove_lines=args.remove_lines)
        size = patches.shape[0]
        print(f"Number of unlabelled patches: {size}")
        gen = patches_generator(images=patches, device=device, batch_size=args.batch_size, add_crop=args.add_crop, crop_scale=args.crop_scale)
        name += f"kernel_{args.kernel}_stride_{args.stride}_delLines_{args.remove_lines}_"

    # Set filepaths outputs
    os.makedirs(config.output_dir, exist_ok=True)
    name += f"ENC_{args.encoder_features}_EXP_{args.expander_features}_s_{args.sim_loss_weight}_v_{args.var_loss_weight}_c{args.cov_loss_weight}_crop_{args.add_crop}_scale_{args.crop_scale}"
    name = name.strip()
    name = name.replace("'", "")
    encoder_filepath = config.output_dir / f"{name}.pt"
    log_path = config.output_dir / f"{name}.csv"

    # VICReg model
    model = VICReg(encoder_features=args.encoder_features, expander_features=args.expander_features)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Instantiate logs variables
    loss_acc = []
    sim_loss_acc = [] 
    var_loss_acc = []
    cov_loss_acc = []

    current_patience = args.patience
    # Train
    best_loss = np.Inf
    best_epoch = 0
    for epoch in range(args.epochs):
        print(f"--Epoch {epoch + 1}--")
        print("Training:")
        for _ in tqdm.tqdm(range(size // args.batch_size), position=0, leave=True):
            xa, xb = next(gen)
            optimizer.zero_grad()
            za, zb = model(xa), model(xb)
            loss, sim_loss, var_loss, cov_loss = vicreg_loss(za, zb, sim_loss_weight=args.sim_loss_weight, var_loss_weight=args.var_loss_weight, cov_loss_weight=args.cov_loss_weight)
            loss.backward()
            optimizer.step()
        loss_acc.append(loss.cpu().detach().item())
        sim_loss_acc.append(sim_loss.cpu().detach().item())
        var_loss_acc.append(var_loss.cpu().detach().item())
        cov_loss_acc.append(cov_loss.cpu().detach().item())
        print(f"Epoch {epoch + 1} ; Total loss: {loss_acc[-1]} -> Invariance Loss: {sim_loss_acc[-1]} | Variance Loss {var_loss_acc[-1]} | Covariance Loss {cov_loss_acc[-1]}")
        if loss_acc[-1] < best_loss:
            print(f"Loss improved from {best_loss} to {loss_acc[-1]}. Saving encoder's weights to {encoder_filepath}")
            best_loss = loss_acc[-1]
            best_epoch = epoch
            model.save(path=encoder_filepath)
            current_patience = args.patience
        else:
            # Early stopping
            current_patience -= 1
            if current_patience == 0:
                break
    print(f"Epoch {best_epoch + 1} achieved lowest loss value = {best_loss:.2f}")

    # Save logs
    logs = {"loss" : loss_acc, "sim_loss" : sim_loss_acc, "var_loss" : var_loss_acc, "cov_loss" : cov_loss_acc}
    logs = pd.DataFrame.from_dict(logs)
    logs.to_csv(log_path, index=False)

    pass

if __name__ == "__main__":
    main()
