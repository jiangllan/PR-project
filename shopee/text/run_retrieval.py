#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 下午7:29
# @Author  : Lan Jiang
# @File    : run_retrieval.py

import sys,os
sys.path.append("..")
# print(sys.path)
import argparse
from text.bm25 import BM25
from utils.text_utils import load_corpus, preprocess_title
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-product-matching/")
    parser.add_argument("--cache_dir", type=str, default="../../Dataset/shopee-product-matching/aux_files")
    args = parser.parse_args()

    corpus = load_corpus(args)
    bm25_model = BM25(corpus)

    data = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    titles = data['title'].tolist()
    for i, title in enumerate(titles):
        if i < 5:
            pre_title = preprocess_title(title)
            res_index = bm25_model.search(pre_title, top_n=1)
            print(title, ": ", ' '.join(corpus[res_index[0]]))
