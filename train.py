import gc
import random

import fire
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import classification_report
from torchinfo import summary

import datasets.config as config
from datasets.loader import load_supervised_data
from my_utils.generators import supervised_data_generator
from my_utils.train_utils import train_test_split, write_plot_results
from network.model import (
    ResnetClassifier,
    SupervisedClassifier,
    VggClassifier,
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
    epochs: int = 150,
    batch_size: int = 16,
    num_runs: int = 5,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------SUPERVISED CNN CLASSIFICATION EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Min occurence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Pretrained: {pretrained}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
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
    output_dir = config.output_dir / ds_name / "supervised"
    output_dir = output_dir / f"{model_type}-Pretrained{pretrained}"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        # 3.2) Train and test model
        class_rep = train_and_test_model(
            XTrain=XTrain,
            YTrain=YTrain,
            XTest=XTest,
            YTest=YTest,
            num_classes=len(w2i),
            model_type=model_type,
            pretrained=pretrained,
            batch_size=batch_size,
            epochs=epochs,
        )
        # NOTE: So far, only accuracy is saved
        accuracy = 100 * class_rep["accuracy"]
        results.append(accuracy)

    # 4) SAVE RESULTS
    print("----------------------------------------------------")
    print("BOOTSTRAP SUMMARY:")
    print(f"\tSamples per class: {samples_per_class}")
    print(f"\tNumber of bootstrap runs: {num_runs}")
    print(f"\tMean accuracy: {np.mean(results):.2f}")
    print(f"\tStandard deviation: {np.std(results):.2f}")
    write_plot_results(
        filepath=output_dir / "results.txt",
        from_weights="-",
        epochs=epochs,
        batch_size=batch_size,
        results=results,
        samples_per_class=samples_per_class,
    )


############################################# UTILS:


def test_model(*, model, X, Y):
    YHAT = []

    model.eval()
    with torch.no_grad():
        for x in tqdm.tqdm(X, position=0, leave=True):
            x = x.unsqueeze(0).to(model.device)
            yhat = model(x)[0]
            yhat = yhat.softmax(dim=0)
            yhat = torch.argmax(yhat, dim=0)
            YHAT.extend(yhat.cpu().detach().numpy())

    class_rep = classification_report(y_true=Y, y_pred=YHAT, output_dict=True)
    accuracy = 100 * class_rep["accuracy"]

    return accuracy, class_rep


def train_and_test_model(
    *,
    XTrain: np.ndarray,
    YTrain: np.ndarray,
    XTest: np.ndarray,
    YTest: np.ndarray,
    num_classes: int,
    model_type: str = "CustomCNN",
    pretrained: bool = False,
    batch_size: int = 16,
    epochs: int = 150,
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    # 1) LOAD DATA
    train_steps = len(XTrain) // batch_size
    train_gen = supervised_data_generator(
        images=XTrain, labels=YTrain, device=device, batch_size=batch_size
    )
    XTest = torch.from_numpy(XTest)

    # 2) CREATE MODEL
    if model_type == "CustomCNN" and not pretrained:
        model = SupervisedClassifier(num_labels=num_classes)
    elif model_type == "Resnet34":
        model = ResnetClassifier(num_labels=num_classes, pretrained=pretrained)
    elif model_type == "Vgg19":
        model = VggClassifier(num_labels=num_classes, pretrained=pretrained)
    else:
        raise NotImplementedError(
            f"Model type {model_type} with pretrained={pretrained} not supported"
        )
    model = model.to(device)
    summary(model, input_size=[(1,) + config.INPUT_SHAPE])

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3) TRAINING
    best_accuracy = 0
    best_epoch = 0
    best_class_rep = None

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training
        for _ in tqdm.tqdm(range(train_steps), position=0, leave=True):
            x, y = next(train_gen)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
        # Testing
        test_accuracy, test_class_rep = test_model(model=model, X=XTest, Y=YTest)
        print(
            f"train_loss: {loss.cpu().detach().item():.4f} - test_accuracy: {test_accuracy:.2f}"
        )

        # Save best model
        if test_accuracy > best_accuracy:
            print(
                f"Test accuracy improved from {best_accuracy:.2f} to {test_accuracy:.2f}"
            )
            best_accuracy = test_accuracy
            best_epoch = epoch
            best_class_rep = test_class_rep

        # Get back to training mode
        model.train()

    # 4) PRINT BEST RESULTS
    print(
        f"Epoch {best_epoch + 1} achieved highest test accuracy value = {best_accuracy:.2f}"
    )

    return best_class_rep


if __name__ == "__main__":
    fire.Fire(run_bootstrap)
