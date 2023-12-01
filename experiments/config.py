import os

################################################### PRETRAIN HPARAMS:
# Common hyperparameters to all datasets
ENTROPY_THRESHOLD = 0.8
ENCODER_FEATURES_DIM = 1600
EXPANDER_FEATURES_DIM = 1024
EPOCHS = 150
BATCH_SIZE = 16
NUM_RANDOM_PATCHES = -1

# Hyperparameters for each dataset
DS_PRETRAIN_HPARAMS = {
    "b-59-850": {
        "kernel": (64, 64),
        "stride": (32, 32),
        "sim_loss_weight": 10,
        "var_loss_weight": 10,
        "cov_loss_weight": 1,
    },
    "Egyptian": {
        "kernel": (32, 32),
        "stride": (16, 16),
        "sim_loss_weight": 10,
        "var_loss_weight": 1,
        "cov_loss_weight": 1,
    },
    "TKH": {
        "kernel": (64, 64),
        "stride": (32, 32),
        "sim_loss_weight": 10,
        "var_loss_weight": 1,
        "cov_loss_weight": 1,
    },
    "Greek": {
        "kernel": (32, 32),
        "stride": (16, 16),
        "sim_loss_weight": 10,
        "var_loss_weight": 10,
        "cov_loss_weight": 1,
    },
}


# Add the common hyperparameters to each dataset
for k, v in DS_PRETRAIN_HPARAMS.items():
    v["entropy_threshold"] = ENTROPY_THRESHOLD
    v["encoder_features_dim"] = ENCODER_FEATURES_DIM
    v["expander_features_dim"] = EXPANDER_FEATURES_DIM
    v["epochs"] = EPOCHS
    v["batch_size"] = BATCH_SIZE
    v["num_random_patches"] = NUM_RANDOM_PATCHES


################################################### TEST HPARAMS:

DS_TEST_HPARAMS = {}


# IMPORTANT NOTE!!!
# This is done accordingly to the saving model structure followed in pretrain.py
# If you change the structure, you must change this function accordingly
def get_model_path(
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
    output_dir = f"datasets/{ds_name}/experiments"
    output_dir = os.path.join(output_dir, "VICReg")
    output_dir = os.path.join(output_dir, f"{model_type}")

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
    encoder_filepath = os.path.join(output_dir, f"{model_name}_encoder.pt")
    return encoder_filepath


for ds_name, pretrain_config in DS_PRETRAIN_HPARAMS.items():
    DS_TEST_HPARAMS[ds_name] = {}
    for model_type in ["CustomCNN", "Resnet34"]:
        DS_TEST_HPARAMS[ds_name][f"{model_type.lower()}_bboxes"] = get_model_path(
            ds_name=ds_name,
            supervised_data=True,
            model_type=model_type,
            **pretrain_config,
        )
        DS_TEST_HPARAMS[ds_name][f"{model_type.lower()}_patches"] = get_model_path(
            ds_name=ds_name,
            supervised_data=False,
            model_type=model_type,
            **pretrain_config,
        )
