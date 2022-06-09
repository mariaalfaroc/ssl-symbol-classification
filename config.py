import pathlib

INPUT_SIZE = (40, 40)
INPUT_SHAPE = (3,) + INPUT_SIZE

# Datasets available: b-59-850, b-3-28, b-50-747, and b-53-781
# All these datasets follow the structure set in set_data_dirs() and have the file extensions defined below
image_extn = ".JPG"
json_extn = ".json"

def set_data_dirs(base_path: str):
    global base_dir
    global images_dir
    global json_dir
    global folds_dir
    global output_dir
    base_dir = pathlib.Path(base_path)
    images_dir = base_dir / "images"
    json_dir = base_dir / "json"
    folds_dir = base_dir / "folds"
    output_dir = base_dir / "experiments"
