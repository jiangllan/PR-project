#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 上午10:02
# @Author  : Lan Jiang
# @File    : split_data.py

import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_datset(args):
    original_data = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tmp = original_data.groupby('label_group').posting_id.agg('unique').to_dict()
    original_data['target'] = original_data.label_group.map(tmp)
    train_idx, val_idx, test_idx = [], [], []
    split_group = []

    data = original_data

    for cur_data_id in range(data.shape[0]):
        cur_data = data.iloc[cur_data_id, :]
        group_sample_number = len(cur_data.target)
        if cur_data.label_group not in split_group:
            print('Processing the {}-th data'.format(cur_data_id))
            if group_sample_number < 6:
                print('\t group sample number = {} < 6'.format(group_sample_number))
                for j in range(group_sample_number):
                    train_idx.append(data[data.posting_id == cur_data.target[j]].index.tolist()[0])
                split_group.append(cur_data.label_group)
            elif 6 <= group_sample_number < 20:
                print('\t 6 <= group_sample_number = {} < 20'.format(group_sample_number))
                index_list = [i for i in range(group_sample_number)]
                random.Random(8).shuffle(index_list)
                for j in range(len(cur_data.target)):
                    if j < 2:
                        val_idx.append(
                            data[data.posting_id == cur_data.target[index_list[j]]].index.tolist()[0])
                    elif 2 <= j < 4:
                        test_idx.append(
                            data[data.posting_id == cur_data.target[index_list[j]]].index.tolist()[0])
                    else:
                        train_idx.append(
                            data[data.posting_id == cur_data.target[index_list[j]]].index.tolist()[0])
                split_group.append(cur_data.label_group)
            else:
                print('\t group_sample_number = {} >= 20'.format(group_sample_number))
                index_list = [i for i in range(group_sample_number)]
                random.Random(8).shuffle(index_list)
                for j in range(group_sample_number):
                    if j < int(group_sample_number * (1 - args.train_size) * 0.5):
                        val_idx.append(
                            data[data.posting_id == cur_data.target[index_list[j]]].index.tolist()[0])
                    elif int(group_sample_number * (1 - args.train_size) * 0.5) <= j < \
                            int(group_sample_number * (1 - args.train_size)):
                        test_idx.append(
                            data[data.posting_id == cur_data.target[index_list[j]]].index.tolist()[0])
                    else:
                        train_idx.append(
                            data[data.posting_id == cur_data.target[index_list[j]]].index.tolist()[0])
                split_group.append(cur_data.label_group)
        else:
            print('{}-th data already processed'.format(cur_data_id))

    assert len(train_idx) + len(val_idx) + len(test_idx) == data.shape[0]
    train, val, test = data.iloc[train_idx, :], data.iloc[val_idx, :], data.iloc[test_idx, :]
    print("train %d val %d test %d" % (len(train), len(val), len(test)))
    train.to_csv(os.path.join(args.save_dir, "new_train.csv"), index=False)
    val.to_csv(os.path.join(args.save_dir, "new_val.csv"), index=False)
    test.to_csv(os.path.join(args.save_dir, "new_test.csv"), index=False)


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
        print(train_idx)
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
    parser.add_argument("--data_dir", type=str, default="/cluster/home/hjjiang/PR-project/data")
    parser.add_argument("--save_dir", type=str, default="/cluster/home/hjjiang/PR-project/data/new_split_data/")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--train_size", type=float, default=0.6, help="Should be between 0 and 1.")
    parser.add_argument("--dev_size", type=float, default=0.1, help="Should be between 0 and 1.")
    args = parser.parse_args()

    split_datset(args)


if __name__ == "__main__":
    main()