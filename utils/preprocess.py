#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 上午10:01
# @Author  : Lan Jiang
# @File    : preprocess.py

import sys
sys.path.append("..")

import os
import pandas as pd
import pickle
import progressbar
from utils.text_utils import load_corpus
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--result_dir", type=str, default="../result/text")
    args = parser.parse_args()

    corpus = load_corpus(args)
    raw_data = pd.read_csv(os.path.join(args.data_dir, "train.csv"))

    raw_data['std_title'] = corpus
    raw_data.to_csv(os.path.join(args.data_dir, "train.csv"), index=False)
    print("Write std file over.")
