import argparse, os, gc, random

import cv2
import tqdm
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

import config
from model import CustomCNN, ResnetEncoder
from data import parse_files_json, parse_files_txt, filter_by_occurrence, preprocess_image, get_w2i_dictionary
from train import train_and_test_model as supervised_train_and_test

def parse_arguments():
    parser = argparse.ArgumentParser(description="KNN classification test arguments")
    parser.add_argument("--ds_path", type=str, default="b-59-850", choices=["Egyptian", "MTH1000", "MTH1200", "TKH", "b-59-850", "b-3-28", "b-50-747", "b-53-781"], help="Dataset's path")
    parser.add_argument("--min_noccurence", type=int, default=50, help="Minimum number of observations to take into account a class symbol")
    parser.add_argument("--model_name", type=str, default=None, help="Model name", required=True)
    parser.add_argument("--weights_path", type=str, default=None, help="Weights path to load")
    parser.add_argument("--encoder_features", type=int, default=1600, help="Encoder features dimension")
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of bootstrap iterations")
    parser.add_argument("--n_neighbors", type=int, default=1, help="Number of neighbors to use")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--patience", type=int, default=150, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    return args

def flatten_load(images):
    X = []
    for image in images:
        image = cv2.resize(image, config.INPUT_SIZE, interpolation=cv2.INTER_AREA)
        image = image / 255
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image.flatten()
        X.append(image)
    return np.asarray(X)

def cnn_load(images, encoder):
    X = []
    with torch.no_grad():
        for image in images:
            image = preprocess_image(image)
            image = image.unsqueeze(0)
            image_re = encoder(image).detach().numpy()
            image_re = image_re[0]
            X.append(image_re)
    return np.asarray(X)

def get_train_test_split(X, Y, samples_per_class, supervised):
    XTrain = []
    YTrain = []
    XTest = []
    YTest = []
    dtype = object if supervised else np.float32
    
    unique_ys = np.unique(Y, axis=0)
    for unique_y in unique_ys:
        indices = np.argwhere(Y==unique_y).flatten()
        train_samples = np.random.choice(indices, samples_per_class, replace=False)
        # Sets cause any duplicated elements to be removed -> We have no repeated elements!
        test_samples = list(set(indices) - set(train_samples))
        XTrain.extend(X[train_samples])
        YTrain.extend(Y[train_samples])
        XTest.extend(X[test_samples])
        YTest.extend(Y[test_samples])
    return np.asarray(XTrain, dtype=dtype), np.asarray(YTrain), np.asarray(XTest, dtype=dtype), np.asarray(YTest)

def writeTSNE_representation(plotfile, x, y):
    x_embedded = TSNE(n_components=2).fit_transform(x)
    with open(plotfile, "w") as datfile:
        for i, components in enumerate(x_embedded):
            datfile.write(f"{components[0]}\t{components[1]}\t{y[i]}\n")

def write_plot_results(plotfile, results):
    with open(plotfile, "w") as datfile:
        for samples, accuracies in results.items():
            datfile.write(f"{samples}\t{np.mean(accuracies)}\t{np.std(accuracies)}\n")

def main():
    gc.collect()
    torch.cuda.empty_cache()

    # Seed
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    args = parse_arguments()
    # Print experiment details
    print("KNN classification test experiment")
    print(args)

    # Data
    config.set_data_dirs(base_path=args.ds_path)
    print(f"Data used {config.base_dir.stem}")
    filepaths = [fname for fname in os.listdir(config.images_dir) if fname.endswith(config.image_extn)]
    print(f"Number of pages: {len(filepaths)}")
    # Perfectly cropped images
    if "json" in config.json_extn:
        images, labels = parse_files_json(filepaths=filepaths)
    else:
        images, labels = parse_files_txt(filepaths=filepaths)
    images, labels = filter_by_occurrence(bboxes=images, labels=labels, min_noccurence=args.min_noccurence)
    # Preprocessing!
    X = []
    Y = []
    supervised = False
    # 1) Convert labels to int
    w2i = get_w2i_dictionary(labels)
    print(f"Size of vocabulary: {len(w2i)}")
    Y = np.asarray([w2i[i] for i in labels])
    # 2) Process images by encoder
    if "Flatten" in args.model_name:
        print("No encoder; passing images as they are to the classifier")
        X = flatten_load(images=images)
    elif "Resnet" in args.model_name:
        print("Using a pretrained Resnet34 to obtain images'representations")
        encoder = ResnetEncoder()
        encoder.eval()
        X = cnn_load(images=images, encoder=encoder)
    elif "Supervised" in args.model_name:
        print("Training a custom CNN in a supervised way")
        X = np.asarray(images, dtype=object)
        supervised = True
    else:
        print("Using a VICReg-pretrained CNN to obtain images'representations")
        encoder = CustomCNN(encoder_features=args.encoder_features)
        encoder.load_state_dict(torch.load(args.weights_path, map_location="cpu"))
        encoder.eval()
        X = cnn_load(images=images, encoder=encoder)

    # Set dir output
    output_dir = config.output_dir / args.model_name
    os.makedirs(output_dir, exist_ok=True)
    TSNE_dir = output_dir / "TSNE"
    os.makedirs(TSNE_dir, exist_ok=True)
    
    # Test
    results = dict()
    samples_per_class = [1]
    samples_per_class += range(5, 35, 5)    
    for samples in samples_per_class:
        print(f"Training with {samples} samples per class")
        results[samples] = []
        # Boostrap
        for _ in range(args.n_iterations):
            XTrain, YTrain, XTest, YTest = get_train_test_split(X=X, Y=Y, samples_per_class=samples, supervised=supervised)
            print(f"Train size: {len(YTrain)}")
            print(f"Test size: {len(YTest)}")
            class_rep = None
            if "Supervised" in args.model_name:
                # Supervised
                class_rep = supervised_train_and_test(
                    data=(XTrain, YTrain, XTest, YTest),
                    w2i=w2i,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience
                )
            else:
                # KNN classifier
                knnClassifier = KNeighborsClassifier(n_neighbors=args.n_neighbors)
                knnClassifier.fit(XTrain, YTrain)
                predictions = knnClassifier.predict(XTest)
                class_rep = classification_report(y_true=YTest, y_pred=predictions, output_dict=True)
            accuracy = 100 * class_rep["accuracy"]
            print(f"Accuracy: {accuracy:.2f} - From {len(YTest)} samples")
            results[samples].append(accuracy)
        if "Supervised" not in args.model_name:
            # TSNE only if KNN classifier is used
            writeTSNE_representation(TSNE_dir / f"{samples}_train.dat", XTrain, YTrain)
    # Save bootstrap results
    write_plot_results(output_dir / f"{args.model_name}_bootstrap_{args.n_iterations}iter.dat", results)
    
    pass

if __name__ == "__main__":
    main()
