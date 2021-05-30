#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/29 上午10:26
# @Author  : Lan Jiang
# @File    : run_ensemble.py

import pickle
from sklearn.metrics import f1_score
import argparse
import os
import numpy as np
import sys

sys.path.append("..")
from utils.text_utils import load_data


def load_ground_true(args, include_self):
    ground_true_list = []
    data = load_data(args, "test")
    label_list = data['label_group'].to_numpy()
    for i in range(len(label_list)):
        label = label_list[i]
        ground_true = 1 * (label_list == label)
        if not include_self:
            ground_true = np.delete(ground_true, i)
        ground_true_list.append(ground_true)

    return np.array(ground_true_list)


def load_single_model_pred(method, include_self=False):
    file_name = "%s_pred%s.pickle" % (method, "_include_self" if include_self else "")
    with open(os.path.join(args.save_dir, file_name), "rb") as f:
        pred_list = pickle.load(f)

    return np.array(pred_list).astype(int)


def vote(single_model_pred_list, ground_true, weights=[0.5, 0.5]):
    votes = np.sum(np.array(weights).reshape(2, 1, 1) * single_model_pred_list, axis=0)
    votes[votes < 0.5] = 0
    votes[votes >= 0.5] = 1
    f1_score_list = [f1_score(ground_true[i], votes[i]) for i in range(len(votes))]
    print("F1: {:.4f}".format(np.mean(f1_score_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-product-matching/split_data")
    parser.add_argument("--save_dir", type=str, default="../../tmp/shopee/")

    args = parser.parse_args()

    weights = [0.55, 0.45]
    print(weights)
    print("Vote [include self]")
    ground_true = load_ground_true(args, include_self=True)
    pred_list = []
    for method in ["bert", "resnet50"]:
        pred = load_single_model_pred(method, include_self=True)
        pred_list.append(pred)

    vote(pred_list, ground_true, weights=weights)

    print("Vote [exclude self]")
    ground_true = load_ground_true(args, include_self=False)
    pred_list = []
    for method in ["bert", "resnet50"]:
        pred = load_single_model_pred(method, include_self=False)
        pred_list.append(pred)

    vote(pred_list, ground_true, weights=weights)