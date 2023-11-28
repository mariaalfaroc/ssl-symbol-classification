from pathlib import Path

INPUT_SIZE = (40, 40)
INPUT_SHAPE = (3,) + INPUT_SIZE

# Datasets available:
# b-59-850, b-3-28, b-50-747, b-53-781, TKH, MTH1000, MTH1200, Egyptian, and Greek
# All these datasets follow the structure set in set_data_dirs()

MUSIC_DATASETS = ["b-59-850", "b-3-28", "b-50-747", "b-53-781"]
TEXT_DATASETS = ["TKH", "MTH1000", "MTH1200", "Egyptian", "Greek"]
DATASETS = MUSIC_DATASETS + TEXT_DATASETS


def set_data_dirs(ds_name: str):
    global base_dir
    global images_dir
    global labels_dir
    global folds_dir
    global output_dir
    global image_extn
    global label_extn
    global patches_file

    assert ds_name in DATASETS, f"Dataset {ds_name} not found"

    base_dir = Path(f"datasets/{ds_name}")
    if ds_name in MUSIC_DATASETS:
        images_dir = base_dir / "images"
        labels_dir = base_dir / "json"
        image_extn = ".JPG"
        label_extn = ".json"
    else:
        images_dir = base_dir / "img"
        labels_dir = base_dir / "label_char"
        image_extn = ".jpg"
        label_extn = ".txt"
    folds_dir = base_dir / "folds"
    output_dir = base_dir / "experiments"
    patches_file = base_dir / "patches.npy"
