#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/20 下午3:53
# @Author  : Lan Jiang
# @File    : run_pca.py

import os
import sys
from text_utils import load_features
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from metrics import mean_average_precision, mean_reciprocal_rank
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
    # print("Execution time: {:.2f}ms".format(duration * 1000 / len(query_list)))
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
    parser.add_argument("--data_dir", type=str, default="../data/split_data/")
    parser.add_argument("--result_dir", type=str, default="../result/text")
    parser.add_argument("--model_name", type=str, default="glove")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument('--whiten', action='store_true')
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument("--top_n", type=int, default="10")
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()

    # load features
    if args.do_train:
        train_X, train_labels = load_features(args, "train")
        # train
        model = train(train_X, args)
        print("train model over.")

        with open(os.path.join(args.result_dir, "pca_model.pickle"), "wb") as f:
            pickle.dump(model, f)
        print("Train & save PCA model over.")

    # evaluate
    if args.do_eval:
        test_X, test_labels = load_features(args, "test")
        model_file = os.path.join(args.result_dir, "pca_model.pickle")
        if not os.path.exists(model_file):
            print("Please train model first")
            sys.exit()
        else:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

        reduced_features = model.transform(test_X)
        # reduced_features = features
        mAP, mrr, F1 = evaluate(reduced_features, test_labels, reduced_features, args)
        print("=" * 9, " Evaluation ", "=" * 9)
        print("F1: {:.4f} mAP@10: {:.4f} MRR: {:.4f}".format(F1, mAP, mrr))

        if args.save_result:
            total_result = np.array([F1, mAP, mrr])
            save_result = np.append(["F1", "mAP@10", "MRR"], total_result, axis=0)
            file_name = "pca-%s-%d%s.txt" % (str(args.threshold), args.n_components, "-whiten" if args.whiten else "")
            np.savetxt(os.path.join(args.result_dir, file_name), save_result, fmt='%s', delimiter=',')


