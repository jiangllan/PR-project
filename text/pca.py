#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/20 下午3:53
# @Author  : Lan Jiang
# @File    : pca.py

from sklearn.decomposition import PCA
import numpy as np
import pickle
import argparse
import os


def run_pca(args):
    with open(os.path.join(args.cache_dir, "%s_emb.pickle" % args.model_name), "rb") as f:
        features = pickle.load(f)
    features = np.array(features)
    pca = PCA(n_components=2)
    pca.fit(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="../../Dataset/shopee-product-matching/aux_files")
    parser.add_argument("--model_name", type=str, default="paraphrase-distilroberta-base-v1")
    parser.add_argument("--n_components", type=int, default=2)
    args = parser.parse_args()

