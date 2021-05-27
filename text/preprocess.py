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


if __name__ == '__main__':
    data_dir = "../../Dataset/shopee-product-matching/"
    with open(os.path.join(data_dir, "aux_files", "corpus.pickle"), "rb") as f:
        corpus = pickle.load(f)

    raw_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    p = progressbar.ProgressBar()
    for i in p(range(len(corpus))):
        std_title = corpus[i]
        assert (isinstance(std_title, str))

    raw_data['std_title'] = corpus
    raw_data.to_csv(os.path.join(data_dir, "std_train.csv"), index=False)
    print("Write std file over.")
