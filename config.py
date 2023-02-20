from pathlib import Path

INPUT_SIZE = (40, 40)
INPUT_SHAPE = (3,) + INPUT_SIZE

# Datasets available: b-59-850, b-3-28, b-50-747, b-53-781, TKH, MTH1000, MTH1200, and Greek
# All these datasets follow the structure set in set_data_dirs()

def set_data_dirs(base_path: str):
    global base_dir
    global images_dir
    global json_dir
    global folds_dir
    global output_dir

    global image_extn
    global json_extn

    base_dir = Path(base_path)

    if base_path[0] == "b": # OMR data
        images_dir = base_dir / "images"
        json_dir = base_dir / "json"
        folds_dir = base_dir / "folds"
        output_dir = base_dir / "experiments"
        image_extn = ".JPG"
        json_extn = ".json"
    else:
        images_dir = base_dir / "img"
        json_dir = base_dir / "label_char"
        folds_dir = base_dir / "folds"
        output_dir = base_dir / "experiments"
        image_extn = ".jpg"
        json_extn = ".txt"
