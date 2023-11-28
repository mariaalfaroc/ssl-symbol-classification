import numpy as np

from my_utils.parser import parse_files
from my_utils.preprocessing import filter_by_occurrence, get_w2i_dictionary


def load_supervised_data(ds_name: str, min_occurence: int = 50) -> dict:
    # 1) Parse files
    X, Y = parse_files(ds_name=ds_name)

    # 2) Filter out samples with low occurrence
    X, Y = filter_by_occurrence(bboxes=X, labels=Y, min_occurence=min_occurence)

    # 3) Get w2i dictionary
    w2i = get_w2i_dictionary(tokens=Y)

    # 4) Preprocessing
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray([w2i[w] for w in Y], dtype=np.int64)

    return {"X": X, "Y": Y, "num_classes": len(w2i)}
