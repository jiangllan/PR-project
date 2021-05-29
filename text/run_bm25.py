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
from utils.text_utils import load_data
import numpy as np
from utils.metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.metrics import f1_score
import progressbar
from gensim.summarization import bm25
import pickle
import time


class BM25(object):
    def __init__(self, corpus, idf=None):
        self.corpus = corpus
        self.model = bm25.BM25(corpus)
        self.idf = sum(map(lambda k: float(self.model.idf[k]), self.model.idf.keys())) / len(
            self.model.idf.keys()) if idf is None else idf

    def search(self, query):
        tok_query = query.strip().split()
        scores = self.model.get_scores(tok_query)
        return np.array(scores)


def evaluate_iter(title, bm25_model, args):
    # top_n
    # pre_title = preprocess_title(title)
    scores = bm25_model.search(title)
    top_inds = np.argsort(scores)[-args.top_n - 1:][::-1]
    top_inds = top_inds[scores[top_inds] > args.threshold]

    # F1
    thresh_preds = 1 * (scores > args.threshold)

    return top_inds, thresh_preds


def evaluate(query_list, label_list, bm25_model, corpus_label_list, args):
    top_pred_list = []
    top_inds_list = []
    thre_pred_list = []
    f1_score_list = []
    p = progressbar.ProgressBar()
    start = time.time()
    for i in p(range(len(query_list))):
        query, label = query_list[i], label_list[i]
        top_inds, thresh_preds = evaluate_iter(query, bm25_model, args)
        ground_true = 1 * (corpus_label_list == label)
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
        top_labels = corpus_label_list[top_inds]
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
        save_file_name = "bm25_pred%s.pickle" % ("_include_self" if args.include_self else "")
        with open(os.path.join(args.save_dir, save_file_name), "wb") as f:
            pickle.dump(thre_pred_list, f)
        print("save bm25 prediction result over.")

    return mAP, mrr, total_f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-product-matching/split_data")
    parser.add_argument("--cache_dir", type=str, default="../../Dataset/shopee-product-matching/aux_files")
    parser.add_argument("--threshold", type=int, default="20")
    parser.add_argument("--save_dir", type=str, default="../../tmp/shopee/")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    args = parser.parse_args()

    if args.include_self:
        data = load_data(args, "test")
    else:
        data = load_data(args, "train")

    corpus = [title.split() for title in data['std_title'].tolist()]
    corpus_label_list = data['label_group'].to_numpy()
    bm25_model = BM25(corpus)
    print("Build BM25 model over.")
    test = load_data(args, "test")
    query_list = test['std_title'].to_numpy()
    label_list = test['label_group'].to_numpy()

    # evaluate
    print("=" * 9, "Evaluation", "=" * 9)
    mAP, mrr, F1 = evaluate(query_list, label_list, bm25_model, corpus_label_list, args)
    print("F1: {:.4f} mAP@10: {:.4f} MRR: {:.4f}".format(F1, mAP, mrr))

    total_result = np.array([F1, mAP, mrr])
    save_result = np.append(["F1", "mAP@10", "MRR"], total_result, axis=0)
    file_name = "bm25-%s.txt" % str(args.threshold)
    np.savetxt(os.path.join(args.save_dir, file_name), save_result, fmt='%s', delimiter=',')

    # # print("\nAverage performance of 5 folds")
    # # print("\tF1\tmAP@10\tMRR")
    # for i, split in enumerate(["dev", "test"]):
    #     orig = total_result[:, :, i, :]
    #     orig = np.append(orig, [np.mean(total_result[:, i, :], axis=0)], axis=0)
    #     orig = np.append([['1'], ['2'], ['3'], ['4'], ['5'], ['AVG']], orig, axis=1)
    #
    #     file_name = "bm25-%s-%s.txt" % (split, str(args.threshold))
    #     np.savetxt(os.path.join(args.save_dir, file_name), orig, fmt='%s', delimiter=',')

    print("Over.")
