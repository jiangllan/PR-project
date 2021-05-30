#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 上午11:11
# @Author  : Lan Jiang
# @File    : run_bert.py

import os
import sys

sys.path.append("..")

import numpy as np
from text_utils import load_features, load_pairwise_data
import argparse
from metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util, losses, evaluation
import progressbar
import torch
import pickle
from torch.utils.data import DataLoader
import logging
import math
import time


def train(model, args):
    model_save_dir = os.path.join(args.result_dir, args.model_name)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    train_samples = load_pairwise_data(args, "train")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Development set: Measure correlation between cosine score and gold labels
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = load_pairwise_data(args, "val")
    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(dev_samples, name='val')

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=binary_acc_evaluator,
              epochs=args.num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              save_best_model=True,
              output_path=model_save_dir)

    # # test
    # model = SentenceTransformer(model_save_dir)
    # test_samples = load_pairwise_data(args, "test")
    # test_evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(test_samples, name='test')
    # test_evaluator(model, output_path=args.model_save_path)


def evaluate_iter(query, corpus, args):
    # top_n
    cos_scores = util.pytorch_cos_sim(query, corpus)[0]
    top_inds = torch.topk(cos_scores, k=args.top_n + 1)[1].numpy()
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
    start = time.time()
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
        save_file_name = "bert_pred%s.pickle" % ("_include_self" if args.include_self else "")
        with open(os.path.join(args.save_dir, save_file_name), "wb") as f:
            pickle.dump(thre_pred_list, f)
        print("save bert prediction result over.")

    return mAP, mrr, total_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/split_data")
    parser.add_argument("--result_dir", type=str, default="../result/text")
    parser.add_argument("--model_name", type=str, default="distilbert-base-indonesian")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument("--top_n", type=int, default="10")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)

    args = parser.parse_args()

    modelAbbr = {
        "distilbert-base-indonesian": "cahya/distilbert-base-indonesian",
        "xlm-100": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokenss",
        "xlm-multi": "sentence-transformers/stsb-xlm-r-multilingual"
    }

    if args.do_train:
        pre_trained_model = modelAbbr[args.model_name]
        model = SentenceTransformer(pre_trained_model)
        train(model, args)

    if args.do_eval:
        test_X, test_labels = load_features(args, "test")
        print("load data over.")
        # evaluate
        print("=" * 9, " Evaluation ", "=" * 9)
        mAP, mrr, F1 = evaluate(test_X, test_labels, test_X, args)
        print("F1: {:.4f} mAP@10: {:.4f} MRR: {:.4f}".format(F1, mAP, mrr))

        if args.save_result:
            total_result = np.array([F1, mAP, mrr])
            save_result = np.append(["F1", "mAP@10", "MRR"], total_result, axis=0)
            file_name = "bert-%s-%s.txt" % (str(args.threshold), args.model_name)
            np.savetxt(os.path.join(args.result_dir, file_name), save_result, fmt='%s', delimiter=',')
