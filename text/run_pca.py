#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/20 下午3:53
# @Author  : Lan Jiang
# @File    : run_pca.py

import os
import sys

sys.path.append("..")
from utils.text_utils import load_features
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from utils.metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.decomposition import PCA
import numpy as np
import argparse
import progressbar
import pickle
import time


def train(features, args):
    model = PCA(n_components=args.n_components, whiten=args.whiten, random_state=0)
    print("training model...")
    model.fit(features)
    return model


def evaluate_iter(cos_scores, args):
    # top_n
    top_inds = np.argsort(cos_scores)[-args.top_n - 1:][::-1]
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
    start = time.time()
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

        f1 = f1_score(ground_true, thresh_preds)
        top_labels = label_list[top_inds]
        rs = 1 * (top_labels == label)

        top_pred_list.append(rs)
        top_inds_list.append(top_inds)
        f1_score_list.append(f1)
        thre_pred_list.append(thresh_preds)

    duration = time.time() - start
    print("Execution time: {:.2f}ms".format(duration * 1000 / len(query_list)))
    mAP = mean_average_precision(top_pred_list)
    mrr = mean_reciprocal_rank(top_pred_list)
    total_f1 = np.mean(f1_score_list)

    if args.save_result:
        save_file_name = "pca_pred%s.pickle" % ("_include_self" if args.include_self else "")
        with open(os.path.join(args.save_dir, save_file_name), "wb") as f:
            pickle.dump(thre_pred_list, f)
        print("save pca prediction result over.")

    return mAP, mrr, total_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-product-matching/split_data")
    parser.add_argument("--model_dir", type=str, default="../../Dataset/shopee-product-matching/split_data")
    parser.add_argument("--save_dir", type=str, default="../../tmp/shopee/")
    parser.add_argument("--model_name", type=str, default="glove")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument('--whiten', action='store_true')
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument("--top_n", type=int, default="10")
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    # load features
    train_X, train_labels = load_features(args, "train")
    test_X, test_labels = load_features(args, "test")
    dev_X, dev_labels = load_features(args, "val")

    # train
    model = train(train_X, args)
    print("train model over.")

    # evaluate
    reduced_features = model.transform(test_X)
    # reduced_features = features
    mAP, mrr, F1 = evaluate(reduced_features, test_labels, reduced_features, args)
    # # include self
    # args.include_self = True
    # _mAP, _mrr, _F1 = evaluate(reduced_features, test_labels, reduced_features, args)
    # print("=" * 9, "Evaluation on test set", "=" * 9)
    # print("& {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}".format(_F1, _mAP, _mrr, F1, mAP, mrr))

    # total_result = np.array([F1, mAP, mrr])
    # save_result = np.append(["F1", "mAP@10", "MRR"], total_result, axis=0)
    # file_name = "pca-%s-%d%s.txt" % (str(args.threshold), args.n_components, "-whiten" if args.whiten else "")
    # np.savetxt(os.path.join(args.save_dir, file_name), save_result, fmt='%s', delimiter=',')
    # save_result = []
    # # print("\nAverage performance of 5 folds")
    # # print("\tF1\tmAP@10\tMRR")
    # for i, split in enumerate(["dev", "test"]):
    #     orig = total_result[:, i, :]
    #     orig = np.append(orig, [np.mean(total_result[:, i, :], axis=0)], axis=0)
    #     orig = np.append([['1'], ['2'], ['3'], ['4'], ['5'], ['AVG']], orig, axis=1)
    #     orig = np.append([["fold", "F1", "mAP@10", "MRR"]], orig, axis=0)
    #     file_name = "pca-%s-%s-%s-%d%s.txt" % (split, args.model_name, str(args.threshold), args.n_components, "-whiten" if args.whiten else "")
    #     np.savetxt(os.path.join(args.save_dir, file_name), orig, fmt='%s', delimiter=',')

    print("Over.")
