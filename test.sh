#!/bin/bash

python main_dnd.py \
 --use_cache \
 --p_interval 100 \
 --cfg ./configs/WW2020.yml \
 --exp_name WW2020_swinv2s \
 --im_scale 896 \
 --bs 1 \
 --acc_bsz 4 \
 --train_set trainval \
 --codalab_pred test \
 --is_test \
 --restore_from path_to_ckpt\