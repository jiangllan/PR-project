#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 下午7:29
# @Author  : Lan Jiang
# @File    : run_retrieval.py

import os
import sys

sys.path.append("..")
# print(sys.path)
import argparse
from text.bm25 import BM25
from utils.text_utils import load_corpus, preprocess_title
import pandas as pd
import numpy as np
from utils.metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.metrics import f1_score, accuracy_score
import progressbar

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-product-matching/")
    parser.add_argument("--cache_dir", type=str, default="../../Dataset/shopee-product-matching/aux_files")
    parser.add_argument("--threshold", type=int, default="20")
    parser.add_argument("--top_n", type=int, default="10")
    args = parser.parse_args()

    corpus = load_corpus(args)
    bm25_model = BM25(corpus)

    data = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    titles = data['title'].tolist()
    labels = data['label_group'].to_numpy()
    top_pred_list = []
    top_inds_list = []
    thre_pred_list = []
    f1_score_list = []
    p = progressbar.ProgressBar()
    for i in p(range(len(titles))):
        title = titles[i]
        pre_title = preprocess_title(title)
        scores = bm25_model.search(pre_title)
        top_inds = np.argsort(scores)[-args.top_n:][::-1]
        # # remove query itself
        # top_inds = top_inds[top_inds != i]
        # try:
        #     assert (len(top_inds) == 10)
        # except AssertionError:
        #     top_inds = top_inds[:10]
        top_labels = labels[top_inds]
        rs = 1 * (top_labels == labels[i])
        top_pred_list.append(rs)
        top_inds_list.append(top_inds)

        pred = 1 * (scores > args.threshold)
        ground_true = 1 * (labels == labels[i])
        pred = np.delete(pred, i)
        ground_true = np.delete(ground_true, i)
        thre_pred_list.append(pred)
        f1 = f1_score(ground_true, pred)
        f1_score_list.append(f1)

    mAP = mean_average_precision(top_pred_list)
    mrr = mean_reciprocal_rank(top_pred_list)
    print("mAP@10: {} MRR: {}".format(mAP, mrr))
    np.savetxt('result/top_10_index.out', top_inds_list, fmt='%s', delimiter=',')

    # acc = accuracy_score(np.array(ground_truth_list).flatten(), np.array(thre_pred_list).flatten())
    total_f1 = np.mean(f1_score_list)
    print("F1: {}".format(total_f1))

    # save result
    np.savetxt('result/thre_pred_list.out', thre_pred_list, fmt='%s', delimiter=',')
