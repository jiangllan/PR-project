import os
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from utils import *
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


def run_knn_only_image(args):
    n_neighbors = [4, 8, 16, 32]
    test_f1_wo_itself = []
    test_mAP_wo_itself = []
    test_mrr_wo_itself = []
    neighbor_num = []

    train = pd.read_csv(os.path.join(args.cache_dir, 'train.csv'))
    train['image'] = args.cache_dir + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    train_feature = np.load(
        os.path.join(args.cache_dir, 'train_{}_224_feature.npy'.format(args.model_name)))
    train_feature = normalize(train_feature)

    test = pd.read_csv(os.path.join(args.cache_dir, 'test.csv'))
    test['image'] = args.cache_dir + 'train_images/' + test['image']
    tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
    test['target'] = test.label_group.map(tmp)
    test_feature = np.load(
        os.path.join(args.cache_dir, 'test_{}_224_feature.npy'.format(args.model_name)))
    test_feature = normalize(test_feature)
    train_feature = test_feature
    train = test

    val = pd.read_csv(os.path.join(args.cache_dir, 'val.csv'))
    val['image'] = args.cache_dir + 'train_images/' + val['image']
    tmp = val.groupby('label_group').posting_id.agg('unique').to_dict()
    val['target'] = val.label_group.map(tmp)
    val_feature = np.load(os.path.join(args.cache_dir, 'val_{}_224_feature.npy'.format(args.model_name)))
    val_feature = normalize(val_feature)

    for n_neighbor in n_neighbors:
        print('Fiting the training data...')
        nbrs = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto', metric=args.distance_metric).fit(train_feature)
        distances, indices = nbrs.kneighbors(train_feature)
        f1, mAP, mrr = calculate_metric(indices, train, n_neighbor=n_neighbor)

        nbrs = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto', metric=args.distance_metric).fit(val_feature)
        distances, indices = nbrs.kneighbors(val_feature)
        f1, mAP, mrr = calculate_metric(indices, val, n_neighbor=n_neighbor)

        print('Predicting the test data...')
        nbrs = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto', metric=args.distance_metric).fit(test_feature)
        distances, indices = nbrs.kneighbors(test_feature)
        f1, mAP, mrr = calculate_metric(indices, test, n_neighbor=n_neighbor, drop_itself=args.drop_itself)
        print('\t test data with {} n_neighbor/ {} distance f1/mAP/mrr = {:.4f} & {:.4f} & {:.4f} '.format(
            n_neighbor, args.distance_metric, f1, mAP, mrr))

        test_f1_wo_itself.append(f1)
        test_mAP_wo_itself.append(mAP)
        test_mrr_wo_itself.append(mrr)
        neighbor_num.append(n_neighbor)

    df = pd.DataFrame(
        {'test_f1_wo_itself': test_f1_wo_itself, 'test_mAP_wo_itself': test_mAP_wo_itself,
         'test_mrr_wo_itself': test_mrr_wo_itself, 'neighbor_num': neighbor_num})

    log_file = os.path.join(args.log_dir, args.log_name)
    with open(log_file, "w") as f:
        df.to_csv(f)


def run_knn_image_and_text(args):
    n_neighbors = [4, 8, 16, 32]
    test_f1_wo_itself = []
    test_mAP_wo_itself = []
    test_mrr_wo_itself = []
    neighbor_num = []

    file_name = '../data/indo/test_features.pickle'
    with open(file_name, "rb") as f:
        test_text_feature = pickle.load(f)
    test_text_feature = normalize(test_text_feature)

    test = pd.read_csv(os.path.join(args.cache_dir, 'test.csv'))
    test['image'] = args.cache_dir + 'train_images/' + test['image']
    tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
    test['target'] = test.label_group.map(tmp)
    test_image_feature = np.load(
        os.path.join(args.cache_dir, 'test_{}_224_feature.npy'.format(args.model_name)))
    test_image_feature = normalize(test_image_feature)

    test_feature = np.concatenate((test_text_feature, test_image_feature), axis=1)
    test_feature = normalize(test_feature)

    for n_neighbor in n_neighbors:
        print('Fiting training data and predicting the test data...')
        nbrs = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto', metric=args.distance_metric).fit(test_feature)
        distances, indices = nbrs.kneighbors(test_feature)
        f1, mAP, mrr = calculate_metric(indices, test, n_neighbor=n_neighbor, drop_itself=args.drop_itself)
        print('\t test data with {} n_neighbor/ {} distance f1/mAP/mrr = {:.4f} & {:.4f} & {:.4f} '.format(
            n_neighbor, args.distance_metric, f1, mAP, mrr))

        test_f1_wo_itself.append(f1)
        test_mAP_wo_itself.append(mAP)
        test_mrr_wo_itself.append(mrr)
        neighbor_num.append(n_neighbor)

    df = pd.DataFrame(
        {'test_f1_wo_itself': test_f1_wo_itself, 'test_mAP_wo_itself': test_mAP_wo_itself,
         'test_mrr_wo_itself': test_mrr_wo_itself, 'neighbor_num': neighbor_num})

    log_file = os.path.join(args.log_dir, args.log_name)
    with open(log_file, "w") as f:
        df.to_csv(f)


def calculate_metric(indices, csv, n_neighbor, drop_itself=True):
    preds_f1 = []
    preds_mAP = []

    for i in range(len(indices)):
        IDX_f1_mAP = indices[i, :]

        if drop_itself:
            final_IDX_f1_mAP = []
            for index in IDX_f1_mAP:
                if index != i:
                    final_IDX_f1_mAP.append(index)
        else:
            final_IDX_f1_mAP = IDX_f1_mAP

        o = csv.iloc[final_IDX_f1_mAP].posting_id.values
        preds_f1.append(o)
        preds_mAP.append(o)

    csv['oof_cnn_f1'] = preds_f1
    csv['oof_cnn_mAP'] = preds_mAP

    csv['f1'] = csv.apply(getMetric('f1', 'oof_cnn_f1'), axis=1)
    csv['mAP'] = csv.apply(getMetric('mAP@10', 'oof_cnn_mAP'), axis=1)
    csv['mrr'] = csv.apply(getMetric('mrr', 'oof_cnn_mAP'), axis=1)

    return csv.f1.mean(), csv.mAP.mean(), csv.mrr.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="../data/split_data/")
    parser.add_argument("--log_dir", type=str, default="../log/image-only/knn/")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--n_neighbors", type=int, default=2)
    parser.add_argument("--distance_metric", type=str, default='euclidean')
    parser.add_argument("--drop_itself", action="store_true", default=False)
    parser.add_argument("--only_image", action="store_true", default=False)
    args = parser.parse_args()

    args.log_name = 'log_knn_use_all_neighbor_split_{}_dropitself-{}_distance-{}.txt'.format(
        args.model_name, args.drop_itself, args.distance_metric)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.only_image:
        run_knn_only_image(args)
    else:
        run_knn_image_and_text(args)
