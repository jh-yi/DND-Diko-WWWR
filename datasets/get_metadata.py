"""
Get dataset split & collect metadata.

Author: Jinhui Yi
Date: 2023.06.01
"""

import os
from collections import Counter
from configs.config import data_root
import yaml
import random

class_to_ind = {
        'unfertilized': 0,
        '_PKCa': 1,
        'N_KCa': 2,
        'NP_Ca': 3,
        'NPK_': 4,
        'NPKCa': 5,
        'NPKCa+m+s': 6,
    }

def get_metadata(data_root, croptype, split='train', verbose=True):
    data_path = os.path.join(data_root, croptype, 'images')
    print("Data root path: ".ljust(40), data_path)
    assert os.path.exists(data_path), "{} does not exist, please check your root_path".format(data_path)
    
    # load split
    split_path = os.path.join(data_root, croptype, split)+'.txt'
    with open(split_path, 'r') as f:
        file_names = f.read().splitlines()
        print("Loading split from: ".ljust(40), split_path)

    # load metadata
    labels = None
    if split != 'test':
        metadata_path = os.path.join(data_root, croptype, 'labels_trainval.yml')
        labels_trainval = yaml.safe_load(open(metadata_path, 'r')) # dict, e.g., {20200422_1.jpg: unfertilized, ...}
        print("Loading labels from: ".ljust(40), metadata_path)

        labels = [labels_trainval[file_name] for file_name in file_names]

    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    # date of data aquisition
    dates = [file_name.split('_')[0] for file_name in file_names] # e.g. 20200422_0.jpg -> 20200422

    if verbose:
        print("Num of images: ".ljust(40), len(file_paths))
        print("Num of labels: ".ljust(40), sum(Counter(labels).values()), len(Counter(labels).values()),Counter(labels))
        print("Num of dates: ".ljust(40), sum(Counter(dates).values()), len(Counter(dates).values()),Counter(dates))

    return file_paths, labels, class_to_ind

if __name__ == '__main__':
    croptype = 'WW2020' # WW2020, WR2021
    split = 'trainval' # trainval, test
    img_paths, label_names, class_to_ind = get_metadata(data_root, croptype, split, verbose=True)