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
from utils.text_utils import load_features, load_pairwise_data
import argparse
from utils.metrics import mean_average_precision, mean_reciprocal_rank
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util, SentencesDataset, losses, evaluation
import progressbar
import torch
from torch.utils.data import DataLoader
import logging
import math


def train(model, args):
    train_samples = load_pairwise_data(args, "train")
    # print(train_samples)
    # train_dataset = SentencesDataset(train_samples, model=model)
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
              output_path=os.path.join(args.model_save_path, args.model_name))

    # test
    model = SentenceTransformer(args.model_save_path)
    test_samples = load_pairwise_data(args, "test")
    test_evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(test_samples, name='test')
    test_evaluator(model, output_path=args.model_save_path)


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
    parser.add_argument("--model_dir", type=str, default="../../Dataset/shopee-product-matching/split_data/fine-tuning")
    parser.add_argument("--model_name", type=str, default="bert")
    parser.add_argument("--model_save_path", type=str, default="../../tmp/shopee/")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument("--top_n", type=int, default="10")
    parser.add_argument("--save_dir", type=str, default="../../tmp/shopee/")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)

    args = parser.parse_args()

    if args.do_train:
        model = SentenceTransformer(args.model_name)
        train(model, args)

    if args.do_eval:
        test_X, test_labels = load_features(args, "test")
        print("load data over.")
        # evaluate
        if args.include_self:
            print("Include query itself")
        print("=" * 9, "Evaluation on test set", "=" * 9)
        mAP, mrr, F1 = evaluate(test_X, test_labels, test_X, args)
        print("F1: {:.4f} mAP@10: {:.4f} MRR: {:.4f}".format(F1, mAP, mrr))

        total_result = np.array([F1, mAP, mrr])
        save_result = np.append(["F1", "mAP@10", "MRR"], total_result, axis=0)
        file_name = "bert-%s-%s.txt" % (str(args.threshold), args.model_name)
        np.savetxt(os.path.join(args.save_dir, file_name), save_result, fmt='%s', delimiter=',')
    # save_result = []
    # # print("\nAverage performance of 5 folds")
    # # print("\tF1\tmAP@10\tMRR")
    # for i, split in enumerate(["dev", "test"]):
    #     orig = total_result[:, i, :]
    #     orig = np.append(orig, [np.mean(total_result[:, i, :], axis=0)], axis=0)
    #     orig = np.append([['1'], ['2'], ['3'], ['4'], ['5'], ['AVG']], orig, axis=1)
    #     orig = np.append([["fold", "F1", "mAP@10", "MRR"]], orig, axis=0)
    #     file_name = "bert-%s-%s-%s.txt" % (split, str(args.threshold), args.model_name)
    #     np.savetxt(os.path.join(args.save_dir, file_name), orig, fmt='%s', delimiter=',')

        print("Over.")
