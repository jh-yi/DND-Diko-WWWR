#!/usr/bin/env python
import sys
import os
import os.path
import yaml

from configs.config import project_root
input_dir = os.path.join(project_root, 'codalab')
output_dir = os.path.join(project_root, 'codalab')
split = 'test'

submit_dir = os.path.join(input_dir, 'res'+'_'+split)
truth_dir = os.path.join(input_dir, 'ref'+'_'+split)

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labels_ww2020_path = os.path.join(truth_dir, "labels_WW2020.txt")
    labels_ww2020 = open(labels_ww2020_path).readlines()
    labels_wr2021_path = os.path.join(truth_dir, "labels_WR2021.txt")
    labels_wr2021 = open(labels_wr2021_path).readlines()

    pred_ww2020_path = os.path.join(submit_dir, "predictions_WW2020.txt")
    pred_ww2020 = open(pred_ww2020_path).readlines()
    pred_wr2021_path = os.path.join(submit_dir, "predictions_WR2021.txt")
    pred_wr2021 = open(pred_wr2021_path).readlines()

    sorted_truth0 = [item.split(' ')[1].strip() for item in sorted(labels_ww2020)]
    sorted_pred0 = [item.split(' ')[1].strip() for item in sorted(pred_ww2020)]
    accuracy0 = float(sum(1 for x,y in zip(sorted_truth0, sorted_pred0) if x == y)) / len(sorted_truth0)
    sorted_truth1 = [item.split(' ')[1].strip() for item in sorted(labels_wr2021)]
    sorted_pred1 = [item.split(' ')[1].strip() for item in sorted(pred_wr2021)]
    accuracy1 = float(sum(1 for x,y in zip(sorted_truth1, sorted_pred1) if x == y)) / len(sorted_truth1)

    res = {
            'accuracy_mean': round((accuracy0+accuracy1)/2*100, 1), 
            'accuracy_ww2020': round(accuracy0*100, 1),
            'accuracy_wr2021': round(accuracy1*100, 1),
            }
    output_path = os.path.join(output_dir, 'scores.txt')
    with open(output_path, 'w') as output_file:
        yaml.dump(res, output_file, default_flow_style=False)

    print(res)