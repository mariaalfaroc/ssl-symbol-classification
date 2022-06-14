import os, json, statistics, random
from collections import Counter
from operator import itemgetter
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from sklearn.utils import shuffle

import config
from augmentation import AugmentStage

def parse_files_json(filepaths: list, return_position: bool = False) -> Tuple[list, list]:
    bboxes = []
    glyphs = []
    positions = []

    for filepath in filepaths:
        json_path = config.json_dir / f"{filepath + config.json_extn}"
        image_path = config.images_dir / filepath
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            with open(json_path) as json_file:  
                data = json.load(json_file)
                if "pages" in data:
                    for page in data["pages"]:
                        if "regions" in page:
                            for region in page["regions"]:
                                if region["type"] == "staff" and "symbols" in region:  
                                    symbols = region["symbols"]
                                    if len(symbols) > 0:
                                        if all([False for s in symbols if "bounding_box" not in s.keys()]):
                                            symbols.sort(key=lambda symbol: symbol["bounding_box"]["fromX"])
                                            r_top, r_left, r_bottom, r_right = region["bounding_box"]["fromY"], region["bounding_box"]["fromX"], region["bounding_box"]["toY"], region["bounding_box"]["toX"]
                                            if image[r_top:r_bottom, r_left:r_right] is not None:
                                                for s in symbols:
                                                    if "bounding_box" in s and "agnostic_symbol_type" in s and "position_in_staff" in s:
                                                        s_top, s_left, s_bottom, s_right = s["bounding_box"]["fromY"], s["bounding_box"]["fromX"], s["bounding_box"]["toY"], s["bounding_box"]["toX"]
                                                        glyphs.append(s["agnostic_symbol_type"])
                                                        positions.append(s["position_in_staff"])
                                                        if (s_bottom - s_top) != 0 and (s_right - s_left) != 0 and (r_bottom - r_top) != 0:
                                                            bboxes.append(image[s_top:s_bottom, s_left:s_right])
    if return_position:
        return bboxes, positions
    return bboxes, glyphs

def parse_files_txt(filepaths: list) -> Tuple[list, list]:
    bboxes = []
    glyphs = []

    for filepath in filepaths:
        label_path = config.json_dir / "{}{}".format(filepath.split(".")[0], config.json_extn)
        image_path = config.images_dir / filepath
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None: 
            with open(label_path) as f:
                FileRead = f.readlines()

            for line in FileRead:
                line_s = line.split()
                symbol = line_s[0]
                y_s, x_s, y_l, x_l = [int(float(u)) for u in line_s[1:]]
                bboxes.append(image[x_s:x_l, y_s:y_l])
                glyphs.append(symbol)

    return bboxes, glyphs

def filter_by_occurrence(bboxes: list, labels: list, min_noccurence: int = 10) -> Tuple[list, list]:
    label_occurence_dict = Counter(labels)
    ids_kept = [id for id, label in enumerate(labels) if label_occurence_dict[label] >= min_noccurence]
    filtered_bboxes = list(itemgetter(*ids_kept)(bboxes))
    filtered_labels = list(itemgetter(*ids_kept)(labels))
    print(f"Removing labels that appear less than {min_noccurence} times")
    print(f"Before: {len(labels)} samples, After: {len(filtered_labels)} samples")
    return filtered_bboxes, filtered_labels

def get_w2i_dictionary(tokens: list) -> dict:
    tokens = sorted(set(tokens))
    w2i = dict(zip(tokens, range(len(tokens))))
    return w2i

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = cv2.resize(image, config.INPUT_SIZE, interpolation=cv2.INTER_AREA)
    image = image / 255
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    return image

def pretrain_data_generator(images: list, device: torch.device, batch_size: int = 32, add_crop: bool = True, crop_scale: Tuple = (0.5, 0.5)) -> Tuple[torch.Tensor, torch.Tensor]:
    augment = AugmentStage(add_crop=add_crop, crop_scale=crop_scale)
    augment.to(device)
    random.shuffle(images)
    size = len(images)
    start = 0
    while True:
        end = min(start + batch_size, size)
        xa = []
        xb = []
        if end - start > 1:
            for i in images[start:end]:
                i = preprocess_image(i)
                i.to(device)
                xa.append(augment(i).cpu().detach())
                xb.append(augment(i).cpu().detach())
            yield torch.stack(xa).to(device), torch.stack(xb).to(device)
        if end == size:
            start = 0
            random.shuffle(images)
        else:
            start = end

def train_data_generator(images: np.ndarray, labels: np.ndarray, device: torch.device, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = shuffle(images, labels, random_state=1)
    size = len(images)
    start = 0
    while True:
        end = min(start + batch_size, size)
        xi = torch.stack([preprocess_image(i) for i in images[start:end]])
        # Labels are already in a int (np.int64) form
        yi = torch.from_numpy(labels[start:end])
        yield xi.to(device), yi.to(device)
        if end == size:
            start = 0
            images, labels = shuffle(images, labels, random_state=1)
        else:
            start = end

if __name__ == "__main__":
    config.set_data_dirs(base_path="TKH")
    filepaths = [fname for fname in os.listdir(config.images_dir) if fname.endswith(config.image_extn)]
    filepaths = filepaths[:2]
    if "json" in config.json_extn:
        bboxes, glyphs = parse_files_json(filepaths=filepaths)
    else:
        bboxes, glyphs = parse_files_txt(filepaths=filepaths)
    assert len(bboxes) == len(glyphs)
    print(f"Total number of symbols {len(glyphs)}")
    label_occurence_dict = Counter(glyphs)
    print(f"Average of occurences {statistics.mean(label_occurence_dict.values())}")
    print(f"Standard deviation of occurances {statistics.stdev(label_occurence_dict.values())}")
    s_lower_bound = min(label_occurence_dict, key=label_occurence_dict.get)
    s_upper_bound = max(label_occurence_dict, key=label_occurence_dict.get)
    print(f"Lower bound: glyph {s_lower_bound} with {label_occurence_dict[s_lower_bound]} occurences")
    print(f"Upper bound: glyph {s_upper_bound} with {label_occurence_dict[s_upper_bound]} occurences")
    # Self-supervised
    gen = pretrain_data_generator(images=bboxes, device=torch.device("cpu"), batch_size=16)
    xa, xb = next(gen)
    print(xa.shape, xb.shape)
    cv2.imwrite("check_images/xa0.jpg", cv2.cvtColor(255*xa[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite("check_images/xb0.jpg", cv2.cvtColor(255*xb[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
    save_image(xa[0], "check_images/xa0_torch.jpg")
    save_image(xb[0], "check_images/xb0_torch.jpg")
    save_image(make_grid(xa, nrow=4), "check_images/xa.jpg")
    save_image(make_grid(xb, nrow=4), "check_images/xb.jpg")
    # Supervised
    w2i = get_w2i_dictionary(tokens=glyphs)
    print(w2i)
    X = np.asarray(bboxes, dtype=object)
    Y = np.asarray([w2i[i] for i in glyphs], dtype=np.int64)
    svgen = train_data_generator(images=X, labels=Y, device=torch.device("cpu"), batch_size=16)
    xi, yi = next(svgen)
    print(xi.shape)
    print(yi.shape)
    cv2.imwrite("check_images/xi0.jpg", cv2.cvtColor(255*xi[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
    save_image(xi[0], "check_images/xi0_torch.jpg")
