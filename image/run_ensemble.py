import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_ground_true(file_path, include_self=True):
    test = pd.read_csv(file_path)
    tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
    test['target'] = test.label_group.map(tmp)

    ground_truth = np.zeros((len(test), len(test)))
    for i in range(len(test)):
        for j in range(len(test.iloc[i, :].target)):
            target_index = test[test.posting_id == test.iloc[i, :].target[j]].index.tolist()[0]
            ground_truth[i, target_index] = 1

    if not include_self:
        mask = 1 - np.diag(np.ones(ground_truth.shape[0]))
        ground_truth *= mask

    return ground_truth


def load_single_model_pred(file_path, include_self=True):
    with open(os.path.join(file_path), "rb") as f:
        pred_list = np.array(pickle.load(f))

    if not include_self:
        mask = 1 - np.diag(np.ones(pred_list.shape[0]))
        pred_list *= mask

    return pred_list.astype(int)


def vote(single_model_pred_list, ground_true, weights=[0.5, 0.5]):
    model_num = len(single_model_pred_list)
    assert len(weights) == model_num
    votes = np.sum(np.array(weights).reshape(model_num, 1, 1) * single_model_pred_list, axis=0)
    votes[votes < 0.5] = 0
    votes[votes >= 0.5] = 1
    f1_score_list = [f1_score(ground_true[i], votes[i]) for i in range(len(votes))]
    print("F1: {:.4f}".format(np.mean(f1_score_list)))


if __name__ == '__main__':
    weights = [0.6, 0.1, 0.3]
    print(weights)

    test_csv = '../../data/new_split_data/new_test.csv'
    ground_true = load_ground_true(test_csv, include_self=True)

    pred_list = []
    pred_pickle_file = ['../../log/image-only/triplet/best_densenet201_top2.pickle',
                        '../../log/image-only/knn/best_manhattan_neighbor4.pickle',
                        '../../log/image-only/pca/best_512_f1threshold_95e-2.pickle',]
    for pickle_file in pred_pickle_file:
        pred = load_single_model_pred(pickle_file, include_self=True)
        pred_list.append(pred)

    pred_list = np.array(pred_list)
    print("Vote [include self]")
    vote(pred_list, ground_true, weights=weights)

    pred_list = []
    for pickle_file in pred_pickle_file:
        pred = load_single_model_pred(pickle_file, include_self=False)
        pred_list.append(pred)

    pred_list = np.array(pred_list)
    print("Vote [exclude self]")
    vote(pred_list, ground_true, weights=weights)
