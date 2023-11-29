import gc
import random
import time

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
from network.model import (
    ResnetClassifier,
    SupervisedClassifier,
    VggClassifier,
)
from train_utils.utils import train_test_split, write_plot_results

# Seed
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def run_bootstrap(
    *,
    ds_name: str,
    samples_per_class: int,
    model_type: str = "CustomCNN",
    pretrained: bool = False,
    epochs: int = 150,
    batch_size: int = 16,
    num_runs: int = 10,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------SUPERVISED CLASSIFICATION EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Model type: {model_type}")
    print(f"Pretrained: {pretrained}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Number of bootstrap runs: {num_runs}")
    print("----------------------------------------------------")

    # 1) LOAD DATA
    data_dict = load_supervised_data(ds_name=ds_name)
    X, Y, num_classes = data_dict["X"], data_dict["Y"], data_dict["num_classes"]
    print(f"Dataset {ds_name} information:")
    print(f"\tTotal number of samples: {len(Y)}")
    print(f"\tNumber of classes: {num_classes}")
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
            num_classes=num_classes,
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
        epochs=epochs,
        batch_size=batch_size,
        results=results,
        samples_per_class=samples_per_class,
    )


############################################# UTILS:


def test_model(*, model, data_gen, steps):
    Y = []
    YHAT = []

    model.eval()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(steps), position=0, leave=True):
            x, y = next(data_gen)
            yhat = model(x)
            yhat = yhat.softmax(dim=1)
            yhat = torch.argmax(yhat, dim=1)
            Y.extend(y.cpu().detach().numpy())
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
    test_steps = len(XTest) // batch_size
    test_gen = supervised_data_generator(
        images=XTest, labels=YTest, device=device, batch_size=batch_size
    )

    # 2) CREATE MODEL
    if model_type == "CustomCNN":
        if pretrained:
            print("CustomCNN model does not support pretrained weights")
            print("Using random weights instead")
        model = SupervisedClassifier(num_labels=num_classes)
    elif model_type == "Resnet34":
        model = ResnetClassifier(num_labels=num_classes, pretrained=pretrained)
    elif model_type == "Vgg19":
        model = VggClassifier(num_labels=num_classes, pretrained=pretrained)
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")
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

        start = time.time()
        # Training
        for _ in tqdm.tqdm(range(train_steps), position=0, leave=True):
            x, y = next(train_gen)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
        # Testing
        test_accuracy, test_class_rep = test_model(
            model=model, data_gen=test_gen, steps=test_steps
        )
        end = time.time()
        print(
            f"train_loss: {loss.cpu().detach().item():.4f} - test_accuracy: {test_accuracy:.2f} - {round(end-start)}s"
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
