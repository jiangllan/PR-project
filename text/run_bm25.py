#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 下午7:29
# @Author  : Lan Jiang
# @File    : run_bm25.py

import os
import sys

sys.path.append("..")
# print(sys.path)
import argparse
from utils.text_utils import load_corpus, preprocess_title, load_data
import pandas as pd
import numpy as np
from utils.metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.metrics import f1_score, accuracy_score
import progressbar
from gensim.summarization import bm25


class BM25(object):
    def __init__(self, corpus, idf=None):
        self.corpus = corpus
        self.model = bm25.BM25(corpus)
        self.idf = sum(map(lambda k: float(self.model.idf[k]), self.model.idf.keys())) / len(self.model.idf.keys()) if idf is None else idf

    def search(self, query):
        tok_query = query.strip().split()
        scores = self.model.get_scores(tok_query)
        return np.array(scores)


def evaluate_iter(title, bm25_model, args):
    # top_n
    pre_title = preprocess_title(title)
    scores = bm25_model.search(pre_title)
    top_inds = np.argsort(scores)[-args.top_n - 1:][::-1]

    # F1
    thresh_preds = 1 * (scores > args.threshold)

    return top_inds, thresh_preds


def evaluate(query_list, label_list, bm25_model, args):
    top_pred_list = []
    top_inds_list = []
    thre_pred_list = []
    f1_score_list = []
    p = progressbar.ProgressBar()
    for i in p(range(len(query_list))):
        query, label = query_list[i], label_list[i]
        top_inds, thresh_preds = evaluate_iter(query, bm25_model, args)
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

        # print("ground_true: ", ground_true)
        # print("thresh_preds: ", thresh_preds)
        f1 = f1_score(ground_true, thresh_preds)
        top_labels = label_list[top_inds]
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
    parser.add_argument("--cache_dir", type=str, default="../../Dataset/shopee-product-matching/aux_files")
    parser.add_argument("--threshold", type=int, default="20")
    parser.add_argument("--top_n", type=int, default="10")
    parser.add_argument('--include_self', action='store_true')
    args = parser.parse_args()

    total_result = []
    for fold in range(1, 6):
        train = load_data(args, "train", fold)
        test = load_data(args, "test", fold)
        dev = load_data(args, "dev", fold)
        fold_result = []    # split * metric

        # evaluate
        if args.include_self:
            print("Include query itself")
        for split, data in zip(["train", "dev", "test"], [train, test, dev]):
            if split == "train":
                continue
            # eval on test
            corpus = [item.split() for item in data['std_title'].tolist()]
            bm25_model = BM25(corpus)
            print("=" * 9, "Evaluation on %s set" % split, "=" * 9)
            query_list = data['std_title'].to_numpy()
            label_list = data['label_group'].to_numpy()
            # reduced_features = features
            mAP, mrr, F1 = evaluate(query_list, label_list, bm25_model, args)
            print("F1: {} mAP@10: {} MRR: {}".format(F1, mAP, mrr))
            fold_result.append([F1, mAP, mrr])

        total_result.append(fold_result)    # fold * split * metric

    total_result = np.array(total_result)
    save_result = []
    # print("\nAverage performance of 5 folds")
    # print("\tF1\tmAP@10\tMRR")
    for i, split in enumerate(["dev", "test"]):
        orig = total_result[:, i, :]
        orig = np.append(orig, [np.mean(total_result[:, i, :], axis=0)], axis=0)
        orig = np.append([['1'], ['2'], ['3'], ['4'], ['5'], ['AVG']], orig, axis=1)
        orig = np.append([["fold", "F1", "mAP@10", "MRR"]], orig, axis=0)
        file_name = "bm25-%s-%s.txt" % (split, str(args.threshold))
        np.savetxt(os.path.join(args.save_dir, file_name), orig, fmt='%s', delimiter=',')

    print("Over.")
