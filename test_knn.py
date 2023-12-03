import gc
import random

import fire
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import datasets.config as config
from datasets.loader import load_supervised_data
from my_utils.train_utils import (
    train_test_split,
    write_plot_results,
    write_tsne_representation,
)
from network.model import (
    CustomCNN,
    ResnetEncoder,
    ResnetEncoderVICReg,
    VggEncoder,
    VggEncoderVICReg,
)

# Seed
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def run_bootstrap(
    *,
    ds_name: str,
    samples_per_class: int,
    min_occurence: int = 50,
    model_type: str = "CustomCNN",
    pretrained: bool = False,
    checkpoint_path: str = "",
    num_runs: int = 5,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------KNN CLASSIFICATION EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Min occurence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Pretrained: {pretrained}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Number of bootstrap runs: {num_runs}")
    print("----------------------------------------------------")

    # 1) LOAD DATA
    data_dict = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence)
    X, Y, w2i = data_dict["X"], data_dict["Y"], data_dict["w2i"]
    print(f"Dataset {ds_name} information:")
    print(f"\tTotal number of samples: {len(Y)}")
    print(f"\tNumber of classes: {len(w2i)}")
    print("----------------------------------------------------")

    # 2) SET OUTPUT DIR
    output_dir = config.output_dir / "knn"
    output_dir = output_dir / f"{model_type}-Pretrained{pretrained}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir = output_dir / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    # 3) RUN BOOTSTRAP
    results = []
    for run in range(num_runs):
        print(f"\t Bootstrap run {run + 1}/{num_runs}")
        # 3.1) Get samples
        XTrain, YTrain, XTest, YTest = train_test_split(
            X=X,
            Y=Y,
            samples_per_class=samples_per_class,
        )
        # 3.2) Train and test KNN classifier
        accuracy, XTrain_embedded = train_and_test_knn(
            XTrain=XTrain,
            YTrain=YTrain,
            XTest=XTest,
            YTest=YTest,
            model_type=model_type,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
        # NOTE: So far, only accuracy is saved
        results.append(accuracy)

        # 3) SAVE TSNE REPRESENTATION
        tsne_filepath = ""
        if checkpoint_path != "":
            tsne_filepath += checkpoint_path.split("/")[-1] + "-"
        tsne_filepath += (
            f"test_on_{samples_per_class}spc_with{min_occurence}-run{run}.dat"
        )
        tsne_filepath = tsne_dir / tsne_filepath
        write_tsne_representation(
            filepath=tsne_filepath,
            x=XTrain_embedded,
            y=YTrain,
            w2i=w2i,
        )

    # 4) SAVE RESULTS
    print("----------------------------------------------------")
    print("BOOTSTRAP SUMMARY:")
    print(f"\tSamples per class: {samples_per_class}")
    print(f"\tNumber of bootstrap runs: {num_runs}")
    print(f"\tMean accuracy: {np.mean(results):.2f}")
    print(f"\tStandard deviation: {np.std(results):.2f}")
    from_weights = checkpoint_path if pretrained else "-"
    from_weights = "imagenet" if pretrained and checkpoint_path == "" else from_weights
    write_plot_results(
        filepath=output_dir / "results.txt",
        from_weights=from_weights,
        epochs="-",
        batch_size="-",
        results=results,
        samples_per_class=samples_per_class,
    )


############################################# UTILS:


def get_images_representations(*, encoder, X, device):
    Y = []
    with torch.no_grad():
        for x in X:
            x = x.unsqueeze(0).to(device)
            y = encoder(x)[0].cpu().detach().numpy()
            Y.append(y)
    return np.asarray(Y).reshape(X.shape[0], -1)


def train_and_test_knn(
    *,
    XTrain: np.ndarray,
    YTrain: np.ndarray,
    XTest: np.ndarray,
    YTest: np.ndarray,
    model_type: str = "Flatten",
    pretrained: bool = False,
    checkpoint_path: str = "",
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    # 1) GET IMAGES REPRESENTATIONS
    if model_type == "Flatten" and not pretrained and checkpoint_path == "":
        # No model is used, just the flatten images
        XTrain = XTrain.reshape(XTrain.shape[0], -1)
        XTest = XTest.reshape(XTest.shape[0], -1)

    elif pretrained:
        # Pretrained with VICReg
        if model_type in ["CustomCNN", "Resnet34", "Vgg19"] and checkpoint_path != "":
            print(
                f"Using a VICReg-pretrained {model_type} to obtain images' representations"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if model_type == "CustomCNN":
                encoder = CustomCNN(encoder_features=checkpoint["encoder_features"])
            elif model_type == "Resnet34":
                encoder = ResnetEncoderVICReg(
                    encoder_features=checkpoint["encoder_features"]
                )
            elif model_type == "Vgg19":
                encoder = VggEncoderVICReg(
                    encoder_features=checkpoint["encoder_features"]
                )
            encoder.load_state_dict(checkpoint["encoder_state_dict"])

        # Pretrained with IMAGENET
        elif model_type in ["Resnet34", "Vgg19"] and checkpoint_path == "":
            print(
                f"Using a IMAGENET-pretrained {model_type} to obtain images' representations"
            )
            if model_type == "Resnet34":
                encoder = ResnetEncoder(pretrained=pretrained)
            elif model_type == "Vgg19":
                encoder = VggEncoder(pretrained=pretrained)

        encoder = encoder.to(device)
        encoder.eval()
        XTrain = get_images_representations(
            encoder=encoder, X=torch.from_numpy(XTrain), device=device
        )
        XTest = get_images_representations(
            encoder=encoder, X=torch.from_numpy(XTest), device=device
        )

    else:
        raise NotImplementedError(
            f"Model type {model_type} with pretrained={pretrained} and checkpoint_path={checkpoint_path} not supported"
        )

    # 2) TRAIN AND TEST KNN CLASSIFIER
    knnClassifier = KNeighborsClassifier(n_neighbors=1)
    knnClassifier.fit(XTrain, YTrain)
    predictions = knnClassifier.predict(XTest)
    class_rep = classification_report(
        y_true=YTest, y_pred=predictions, output_dict=True
    )
    # NOTE: So far, only accuracy is saved
    accuracy = 100 * class_rep["accuracy"]
    print(f"Accuracy: {accuracy:.2f}")

    return accuracy, XTrain


if __name__ == "__main__":
    fire.Fire(run_bootstrap)
