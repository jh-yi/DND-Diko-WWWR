#!/bin/bash

python main_dnd.py \
 --use_cache \
 --p_interval 100 \
 --cfg ./configs/WW2020.yml \
 --exp_name WW2020_swinv2s \
 --im_scale 896 \
 --bs 4 \
 --acc_bsz 4 \
 --codalab_pred test \
 --train_set trainval \