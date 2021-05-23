#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 上午10:02
# @Author  : Lan Jiang
# @File    : split_data.py

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import argparse


def split_train_and_test(args):
    data = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    gss = GroupShuffleSplit(n_splits=args.n_splits if args.n_splits else 5,
                            train_size=args.train_size if args.train_size else 5,
                            random_state=42)
    gss.get_n_splits()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    split = 1
    for train_idx, test_idx in gss.split(data, groups=data['label_group'].tolist()):
        train, dev_and_test = data.iloc[train_idx, :], data.iloc[test_idx, :]
        # split dev from test
        dev = dev_and_test.sample(frac=0.5, random_state=42)
        test = dev_and_test.drop(dev.index)
        print("Split %d: train %d test %d dev %d" % (split, len(train), len(test), len(dev)))
        train.to_csv(os.path.join(args.save_dir, "train_split_%d.csv" % split), index=False)
        test.to_csv(os.path.join(args.save_dir, "test_split_%d.csv" % split), index=False)
        dev.to_csv(os.path.join(args.save_dir, "dev_split_%d.csv" % split), index=False)
        split += 1

    print("Split and save %d-fold train&test data over." % (split-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--train_size", type=float, default=0.8, help="Should be between 0 and 1.")
    parser.add_argument("--dev_size", type=float, default=0.1, help="Should be between 0 and 1.")
    args = parser.parse_args()

    split_train_and_test(args)


if __name__ == "__main__":
    main()