from typing import Tuple

import numpy as np
from sklearn.manifold import TSNE


def write_tsne_representation(filepath, x, y, w2i):
    i2w = {v: k for k, v in w2i.items()}
    x_embedded = TSNE(n_components=2).fit_transform(x)
    with open(filepath, "w") as datfile:
        for i, components in enumerate(x_embedded):
            datfile.write(f"{components[0]}\t{components[1]}\t{y[i]}\t{i2w[y[i]]}\n")


def write_plot_results(
    filepath,
    from_weights,
    epochs,
    batch_size,
    results,
    samples_per_class,
):
    # NOTE: results is a list of accuracies!
    if not filepath.exists():
        with open(filepath, "w") as datfile:
            header = [
                "from_weights",
                "samples_per_class",
                "bootstrap_runs",
                "epochs",
                "batch_size",
                "mean_accuracy",
                "std_accuracy",
            ]
            datfile.write("\t".join(header) + "\n")
    with open(filepath, "a") as datfile:
        values = [
            from_weights,
            samples_per_class,
            len(results),
            epochs,
            batch_size,
            np.mean(results),
            np.std(results),
        ]
        datfile.write("\t".join([str(value) for value in values]) + "\n")


def train_test_split(
    X: np.ndarray, Y: np.ndarray, samples_per_class: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    XTrain, YTrain = [], []
    XTest, YTest = [], []

    unique_ys = np.unique(Y, axis=0)
    for unique_y in unique_ys:
        indices = np.argwhere(Y == unique_y).flatten()
        train_samples = np.random.choice(indices, samples_per_class, replace=False)
        # Sets cause any duplicated elements to be removed -> We have no repeated elements!
        test_samples = list(set(indices) - set(train_samples))
        XTrain.extend(X[train_samples])
        YTrain.extend(Y[train_samples])
        XTest.extend(X[test_samples])
        YTest.extend(Y[test_samples])

    return (
        np.asarray(XTrain, dtype=np.float32),
        np.asarray(YTrain, dtype=np.int64),
        np.asarray(XTest, dtype=np.float32),
        np.asarray(YTest, dtype=np.int64),
    )
