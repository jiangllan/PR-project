#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/20 下午3:53
# @Author  : Lan Jiang
# @File    : run_pca.py

import sys

sys.path.append("..")
from utils.text_utils import load_features
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from utils.metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.decomposition import PCA
import numpy as np
import argparse
import torch
import progressbar


def train(features, args):
    model = PCA(n_components=args.n_components, whiten=args.whiten, random_state=0)
    print("training model...")
    model.fit(features)
    return model


def evaluate_iter(cos_scores, args):
    # top_n
    top_inds = np.argsort(cos_scores)[-args.top_n-1:][::-1]
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
    cos_scores_matrix = cosine_similarity(corpus)
    p = progressbar.ProgressBar()
    for i in p(range(len(query_list))):
        query, label = query_list[i], label_list[i]
        top_inds, thresh_preds = evaluate_iter(cos_scores_matrix[i, :], args)
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
    parser.add_argument("--model_name", type=str, default="glove")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument('--whiten', action='store_true')
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument("--top_n", type=int, default="10")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    # load features
    total_result = []
    for fold in range(1, 6):
        train_X, train_labels = load_features(args, "train", fold)
        test_X, test_labels = load_features(args, "test", fold)
        dev_X, dev_labels = load_features(args, "dev", fold)
        fold_result = []

        # train
        model = train(train_X, args)
        print("train model over.")

        # evaluate
        if args.include_self:
            print("Include query itself")
        for split, features, labels in zip(["train", "dev", "test"], [train_X, test_X, dev_X],
                                           [train_labels, test_labels, dev_labels]):
            # if split == "train": continue
            # eval on test
            print("=" * 9, "Evaluation on %s set" % split, "=" * 9)
            reduced_features = model.transform(features)
            # reduced_features = features
            mAP, mrr, F1 = evaluate(reduced_features, labels, reduced_features, args)
            print("F1: {} mAP@10: {} MRR: {}".format(F1, mAP, mrr))
            fold_result.append([F1, mAP, mrr])

        total_result.append(fold_result)

    total_result = np.array(total_result)
    print("\nAverage performance of 5 folds")
    print("\tF1\tmAP@10\tMRR")
    for i, split in enumerate(["train", "dev", "test"]):
        print("{} {:.4f} {:.4f} {:.4f}".format(
            split,
            np.mean(total_result[:, i, :], axis=0)[0],
            np.mean(total_result[:, i, :], axis=0)[1],
            np.mean(total_result[:, i, :], axis=0)[2]
        ))