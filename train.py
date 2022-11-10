import os, gc, random
from model import CustomCNN

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from model import SupervisedClassifier

from data import train_data_generator, parse_files_txt, parse_files_json, get_w2i_dictionary
import config

def test_model(*, model, data_gen, steps):
    Y = []
    YHAT = []
    model.eval()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(steps), position=0, leave=True):
                x, y = next(data_gen)
                yhat = model(x)
                yhat = F.softmax(yhat, dim=1)
                yhat = torch.argmax(yhat, dim=1)
                Y.extend(y.cpu().detach().numpy())
                YHAT.extend(yhat.cpu().detach().numpy())
    class_rep = classification_report(y_true=Y, y_pred=YHAT, output_dict=True)
    accuracy = 100 * class_rep["accuracy"]
    print(f"Accuracy: {accuracy:.2f} - From {len(Y)} samples")
    return accuracy, class_rep

def train_and_test_model(*, data, w2i, batch_size, epochs, patience, model):
    torch.cuda.empty_cache()
    gc.collect()
    # Run on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    # Data
    XTrain, YTrain, XTest, YTest = data
    train_size = len(XTrain)
    train_gen = train_data_generator(images=XTrain, labels=YTrain, device=device, batch_size=batch_size)
    test_size = len(XTest)
    test_gen = train_data_generator(images=XTest, labels=YTest, device=device, batch_size=batch_size)

    # Model summary:
    model.to(device)
    summary(model, input_size=[(1,) + config.INPUT_SHAPE])

    # Optimizer and CrossEntropyLoss
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    epochs_acc = 0

    # Train and test
    best_accuracy = 0
    best_epoch = 0
    best_class_rep = None
    model.train()
    for epoch in range(epochs):
        print(f"--Epoch {epoch + 1}--")
        print("Training:")
        for _ in tqdm.tqdm(range(train_size // batch_size), position=0, leave=True):
            x, y = next(train_gen)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
        print(f"loss = {loss.cpu().detach().item()}")
        print("Testing:")
        test_accuracy, test_class_rep = test_model(model=model, data_gen=test_gen, steps=test_size // batch_size)
        print(f"test-accuracy = {test_accuracy}")
        # Early stopping
        if test_accuracy > best_accuracy:
            print(f"Test accuracy improved from {best_accuracy} to {test_accuracy}")
            best_accuracy = test_accuracy
            best_epoch = epoch
            best_class_rep = test_class_rep
            epochs_acc = 0
        elif epoch > 10:
            epochs_acc += 1
            if epochs_acc == patience:
                break
        # Get back to training mode
        model.train()      
    print(f"Epoch {best_epoch + 1} achieved highest test accuracy value = {best_accuracy:.2f}")
    return best_class_rep

if __name__ == "__main__":
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    # Data
    config.set_data_dirs(base_path="Egyptian")
    print(f"Data used {config.base_dir.stem}")
    filepaths = [fname for fname in os.listdir(config.images_dir) if fname.endswith(config.image_extn)]
    print(f"Number of pages: {len(filepaths)}")

    if 'json' in config.json_extn:
        images, labels = parse_files_json(filepaths=filepaths)
    else:
        images, labels = parse_files_txt(filepaths=filepaths)

    print(f"Number of samples: {len(labels)}")
    w2i = get_w2i_dictionary(labels)
    print(f"Size of vocabulary: {len(w2i)}")
    # Preprocessing
    X = np.asarray(images, dtype=object)
    Y = np.asarray([w2i[i] for i in labels], dtype=np.int64)
    # Create train and test set
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, random_state=1)
    print(f"Train size: {len(XTrain)}")
    print(f"Test size: {len(XTest)}")
    # Create model:
    model = SupervisedClassifier(num_labels=len(w2i))
    # Train and test
    train_and_test_model(
        data=(XTrain, YTrain, XTest, YTest),
        w2i=w2i,
        batch_size=32,
        epochs=150,
        patience=10,
        model = model
    )
