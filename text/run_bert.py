#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 上午11:11
# @Author  : Lan Jiang
# @File    : run_bert.py

import sys
import os
sys.path.append("..")

import pickle
import pandas as pd
import numpy as np
from utils.text_utils import load_features
import argparse
from utils.metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util
import progressbar
import torch


def evaluate_iter(query, corpus, args):
    # top_n
    cos_scores = util.pytorch_cos_sim(query, corpus)[0]
    top_inds = torch.topk(cos_scores, k=args.top_n+1)[1].numpy()
    # top_inds = np.argsort(cos_scores)[-args.top_n-1:][::-1]
    cos_scores = cos_scores.numpy()
    top_inds = top_inds[cos_scores[top_inds] > args.threshold]

    # F1
    # print(cos_scores)
    thresh_preds = 1 * (cos_scores > args.threshold)

    return top_inds, thresh_preds


def evaluate(query_list, label_list, corpus, args):
    top_pred_list = []
    top_inds_list = []
    thre_pred_list = []
    f1_score_list = []
    p = progressbar.ProgressBar()
    for i in p(range(len(query_list))):
        query, label = query_list[i], label_list[i]
        top_inds, thresh_preds = evaluate_iter(query, corpus, args)
        ground_true = 1 * (label_list == label)
        if not args.include_self:
            # remove query itself from the corpus
            thresh_preds = np.delete(thresh_preds, i)
            ground_true = np.delete(ground_true, i)
            top_inds = top_inds[top_inds != i]
        try:
            assert (len(top_inds) == 10)
        except AssertionError:
            top_inds = top_inds[:10]
        # print(top_inds)

        # print(ground_true[:20], thresh_preds[:20])
        f1 = f1_score(ground_true, thresh_preds)
        top_labels = label_list[top_inds]
        # print(label)
        # print(top_labels)
        rs = 1 * (top_labels == label)

        top_pred_list.append(rs)
        top_inds_list.append(top_inds)
        f1_score_list.append(f1)
        thre_pred_list.append(thresh_preds)

    mAP = mean_average_precision(top_pred_list)
    mrr = mean_reciprocal_rank(top_pred_list)
    total_f1 = np.mean(f1_score_list)

    return mAP, mrr, total_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-product-matching/split_data")
    parser.add_argument("--model_name", type=str, default="bert")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument("--top_n", type=int, default="10")

    args = parser.parse_args()

    total_result = []
    for fold in range(1, 6):
        train_X, train_labels = load_features(args, "train", fold)
        test_X, test_labels = load_features(args, "test", fold)
        dev_X, dev_labels = load_features(args, "dev", fold)
        fold_result = []  # split * metric
        print("load data over.")

        # evaluate
        if args.include_self:
            print("Include query itself")
        for split, features, labels in zip(["train", "dev", "test"], [train_X, test_X, dev_X],
                                           [train_labels, test_labels, dev_labels]):
            if split == "train":
                continue
            # eval on test
            print("=" * 9, "Evaluation on %s set" % split, "=" * 9)
            mAP, mrr, F1 = evaluate(features, labels, features, args)
            print("F1: {} mAP@10: {} MRR: {}".format(F1, mAP, mrr))
            fold_result.append([F1, mAP, mrr])

        total_result.append(fold_result)  # fold * split * metric

    total_result = np.array(total_result)
    save_result = []
    # print("\nAverage performance of 5 folds")
    # print("\tF1\tmAP@10\tMRR")
    for i, split in enumerate(["dev", "test"]):
        orig = total_result[:, i, :]
        orig = np.append(orig, [np.mean(total_result[:, i, :], axis=0)], axis=0)
        orig = np.append([['1'], ['2'], ['3'], ['4'], ['5'], ['AVG']], orig, axis=1)
        orig = np.append([["fold", "F1", "mAP@10", "MRR"]], orig, axis=0)
        file_name = "bert-%s-%s-%s.txt" % (split, str(args.threshold), args.model_name)
        np.savetxt(os.path.join(args.save_dir, file_name), orig, fmt='%s', delimiter=',')

    print("Over.")
