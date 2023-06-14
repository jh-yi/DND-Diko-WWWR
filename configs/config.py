#!/usr/bin/env python
# Author: Jinhui Yi
# Created: 2023-06-01

from easydict import EasyDict
from pathlib import Path
import numpy as np
import os
import yaml
import argparse


project_root = Path(__file__).resolve().parents[1]
data_root = '/home/yij/remote_home/datasets/phenorob_cp2/Challenge/DND-Diko-WWWR' # path to dataset

cfg = EasyDict()

# COMMON CONFIGS
cfg.CROPTYPE = 'images'
cfg.SEED = 42                 # 1234, 3407, 42
cfg.NWORK = 4

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.OPTIM = 'sgd'         # sgd | adam | rmsprop
# Network params
cfg.TRAIN.LR = 1e-3             # 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 1e-2   # 0.0005， L2
cfg.TRAIN.POWER = 0.9
# cfg.TRAIN.CLIP = 5.0          # gradients will be clipped to have norm less than this
# Other params
cfg.TRAIN.MAX_EPOCH = 100


# TEST CONFIGS
cfg.TEST = EasyDict()
# Network params
# cfg.TEST.MODEL_WEIGHT = (1.0,)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)

    return cfg

def yaml_load(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
def yaml_dump(python_object, file_path):
    make_parent(file_path)
    with open(file_path, 'w') as f:
        yaml.dump(python_object, f, default_flow_style=False)
def make_parent(file_path):
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for ...")

    # action
    parser.add_argument('--is_test', action='store_true')
    parser.add_argument('--use_cache', dest='USE_CACHE', help='use preprocessed cache for acceleration', action='store_true')
    # numeric
    parser.add_argument('--p_interval', help='print during training', type=int, default=100)
    parser.add_argument('--ngpus', dest='NGPUS', type=int, default=1)
    parser.add_argument('--im_scale', dest='im_scale', type=int, default=896)
    parser.add_argument('--bsz', type=int, default=4)    
    parser.add_argument('--acc_bsz', type=int, default=1, help='batch size for gradient accumulation')  
    # str
    parser.add_argument('--cfg', type=str, default='./configs/WW2020.yml', required=True, help='optional config file', )   
    parser.add_argument('--exp_name', type=str, required=False, default="play", help='data config path')
    parser.add_argument('--restore_from', type=str, required=False, default="")
    parser.add_argument('--codalab_pred', type=str, required=False, default="test", help="save predictions for codalab. test for submission, val for validation")
    parser.add_argument('--train_set', type=str, required=False, default="trainval", help="trainval or train: trainval means no validation set needed")
  

    return parser.parse_args()