import gc
import random

import fire
import numpy as np
import pandas as pd
import torch
import tqdm
from torchinfo import summary

import datasets.config as config
from datasets.loader import load_pretrain_data
from my_utils.generators import pretrain_data_generator
from network.loss import vicreg_loss
from network.model import VICReg

# Seed
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def run_pretrain(
    *,
    ds_name: str,
    supervised_data: bool = False,
    num_random_patches: int = -1,
    kernel: tuple = (64, 64),
    stride: tuple = (32, 32),
    entropy_threshold: float = 0.8,
    model_type: str = "CustomCNN",
    encoder_features_dim: int = 1600,
    expander_features_dim: int = 1024,
    epochs: int = 150,
    batch_size: int = 16,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------VICREG PRETRAINING EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"Supervised data: {supervised_data}")
    print(f"Number of random patches: {num_random_patches}")
    print(f"Kernel: {kernel}")
    print(f"Stride: {stride}")
    print(f"Entropy threshold: {entropy_threshold}")
    print(f"Model type: {model_type}")
    print(f"Encoder features dimension: {encoder_features_dim}")
    print(f"Expander features dimension: {expander_features_dim}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("----------------------------------------------------")

    # 1) LOAD DATA
    X = load_pretrain_data(
        ds_name=ds_name,
        supervised=supervised_data,
        num_random_patches=num_random_patches,
        kernel=kernel,
        stride=stride,
        entropy_threshold=entropy_threshold,
    )

    # 2) SET OUTPUT DIR
    output_dir = config.output_dir / ds_name / "VICReg"
    output_dir = output_dir / f"{model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = ""
    if supervised_data:
        model_name += "bboxes_"
    else:
        model_name = f"{num_random_patches}" if num_random_patches > 0 else "all"
        model_name += "patches_"
        model_name += f"k{'x'.join(map(str, kernel))}_"
        model_name += f"s{'x'.join(map(str, stride))}_"
        model_name += f"et{entropy_threshold}_"
    model_name += f"encdim{encoder_features_dim}_"
    model_name += f"expdim{expander_features_dim}_"
    model_name += f"bs{batch_size}_"
    model_name += f"ep{epochs}_"
    model_name += f"sw{sim_loss_weight}_"
    model_name += f"vw{var_loss_weight}_"
    model_name += f"cw{cov_loss_weight}"
    encoder_filepath = output_dir / f"{model_name}_encoder.pt"

    # 3) PRETRAINING
    logs_dict = pretrain_model(
        X=X,
        encoder_filepath=str(encoder_filepath),
        model_type=model_type,
        encoder_features_dim=encoder_features_dim,
        expander_features_dim=expander_features_dim,
        batch_size=batch_size,
        epochs=epochs,
        sim_loss_weight=sim_loss_weight,
        var_loss_weight=var_loss_weight,
        cov_loss_weight=cov_loss_weight,
    )

    # 4) SAVE RESULTS
    pd.DataFrame(logs_dict).to_csv(
        output_dir / f"{model_name}_train_logs.csv", index=False
    )


############################################# UTILS:


def pretrain_model(
    *,
    X: torch.Tensor,
    encoder_filepath: str,
    model_type: str = "CustomCNN",
    encoder_features_dim: int = 1600,
    expander_features_dim: int = 1024,
    batch_size: int = 16,
    epochs: int = 150,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    # 1) LOAD DATA
    train_steps = len(X) // batch_size
    train_gen = pretrain_data_generator(images=X, device=device, batch_size=batch_size)

    # 2) CREATE MODEL
    if model_type in ["CustomCNN", "Resnet34", "Vgg19"]:
        model = VICReg(
            base_model=model_type,
            encoder_features=encoder_features_dim,
            expander_features=expander_features_dim,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")
    model = model.to(device)
    summary(model, input_size=[(1,) + config.INPUT_SHAPE])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3) TRAINING
    best_epoch = 0
    best_losses = {
        "sim_loss": float("inf"),
        "var_loss": float("inf"),
        "cov_loss": float("inf"),
        "total_loss": float("inf"),
    }
    losses_acc = {
        "sim_loss": [],
        "var_loss": [],
        "cov_loss": [],
        "total_loss": [],
    }

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training
        broken_epoch = False
        for _ in tqdm.tqdm(range(train_steps), position=0, leave=True):
            xa, xb = next(train_gen)
            optimizer.zero_grad()
            za, zb = model(xa), model(xb)
            loss_dict = vicreg_loss(
                za=za,
                zb=zb,
                sim_loss_weight=sim_loss_weight,
                var_loss_weight=var_loss_weight,
                cov_loss_weight=cov_loss_weight,
            )
            loss = loss_dict["loss"]
            if torch.isnan(loss):
                broken_epoch = True
                break
            loss.backward()
            optimizer.step()

        if broken_epoch:
            for k, v in losses_acc.items():
                losses_acc[k].append(v[-1])
        else:
            for k, v in loss_dict.items():
                losses_acc[k].append(v.cpu().detach().item())

        print_info = []
        for k, v in losses_acc.items():
            print_info.append(f"{k}: {v[-1]:.4f}")
        print_info = " - ".join(print_info)
        print(print_info)

        # Save best model
        if losses_acc["loss"][-1] < best_losses["loss"]:
            print(
                f"Loss improved from {best_losses['loss']} to {losses_acc['loss'][-1]}. Saving encoder's weights to {encoder_filepath}"
            )
            best_losses = {
                "sim_loss": losses_acc["sim_loss"][-1],
                "var_loss": losses_acc["var_loss"][-1],
                "cov_loss": losses_acc["cov_loss"][-1],
                "total_loss": losses_acc["loss"][-1],
            }
            best_epoch = epoch
            model.save(path=encoder_filepath)

    # 4) PRINT BEST RESULTS
    print(
        f"Epoch {best_epoch + 1} achieved lowest loss value = {best_losses['loss']:.4f}"
    )

    return losses_acc


if __name__ == "__main__":
    fire.Fire(run_pretrain)
