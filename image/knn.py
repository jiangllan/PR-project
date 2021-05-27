import os
import torch
import argparse
import numpy as np
import pandas as pd
from metrics import getMetric
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


def run_knn(args):
    n_neighbors = [8, 16, 32]
    train_f1_wo_itself = []
    train_mAP_wo_itself = []
    train_mrr_wo_itself = []
    val_f1_wo_itself = []
    val_mAP_wo_itself = []
    val_mrr_wo_itself = []
    test_f1_wo_itself = []
    test_mAP_wo_itself = []
    test_mrr_wo_itself = []
    split_id = []
    for split_index in range(1, args.n_splits + 1):
        print('--------------- SPLIT {} ---------------'.format(split_index))
        train = pd.read_csv(os.path.join(args.cache_dir, 'train_split_{}.csv'.format(split_index)))
        train['image'] = args.cache_dir + 'train_images/' + train['image']
        tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
        train['target'] = train.label_group.map(tmp)
        train_feature = np.load(os.path.join(args.cache_dir, 'train_split_{}_r50_224_feature.npy'.format(split_index)))

        val = pd.read_csv(os.path.join(args.cache_dir, 'dev_split_{}.csv'.format(split_index)))
        val['image'] = args.cache_dir + 'train_images/' + val['image']
        tmp = val.groupby('label_group').posting_id.agg('unique').to_dict()
        val['target'] = val.label_group.map(tmp)
        val_feature = np.load(os.path.join(args.cache_dir, 'dev_split_{}_r50_224_feature.npy'.format(split_index)))

        test = pd.read_csv(os.path.join(args.cache_dir, 'test_split_{}.csv'.format(split_index)))
        test['image'] = args.cache_dir + 'train_images/' + test['image']
        tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
        test['target'] = test.label_group.map(tmp)
        test_feature = np.load(os.path.join(args.cache_dir, 'test_split_{}_r50_224_feature.npy'.format(split_index)))

        for n_neighbor in n_neighbors:
            print('Fiting the training data...')
            nbrs = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto', metric='euclidean').fit(train_feature[:2000])
            # distances, indices = nbrs.kneighbors(train_feature[:2000])
            # f1, mAP, mrr = calculate_metric(indices, train)
            # print('\t train data with {} n_neighbor f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_neighbor, f1, mAP, mrr))
            # print(indices.shape)
            distances, indices = nbrs.kneighbors(val_feature)
            f1, mAP, mrr = calculate_metric(indices, val)
            print('\t val data with {} n_neighbor f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_neighbor, f1, mAP, mrr))

            # train_features_reduce = normalize(pca.transfnorm(train_feature))
            # val_features_reduce = normalize(pca.transform(val_feature))
            #
            # train_features_reduce = torch.from_numpy(train_features_reduce)
            # train_features_reduce = train_features_reduce.cuda()
            # print('train_features_reduce size = {}'.format(train_features_reduce.size()))
            # val_features_reduce = torch.from_numpy(val_features_reduce)
            # val_features_reduce = val_features_reduce.cuda()
            # print('val_features_reduce size = {}'.format(val_features_reduce.size()))
            #
            # print('--------------- train data with {} components ------------------------'.format(n_components))
            # cos_similarity(train_features_reduce, train, args)
            # print('--------------- val data with {} components ------------------------'.format(n_components))
            # cos_similarity(val_features_reduce, val, args)


def calculate_metric(indices, csv):
    preds_f1 = []
    preds_mAP = []

    for i in range(len(indices)):
        # print(i)
        # IDX_f1 = np.where(distances[k, ] > 0.95)[0][:]
        # IDX_mAP = np.argsort(distances[k, ])[::-1][:10]  # Top 10, except itself
        IDX_f1 = indices[i, :10]
        IDX_mAP = indices[i, :10]
        # print(IDX_f1)
        o = csv.iloc[IDX_f1].posting_id.values
        # print('itself:', csv.iloc[i].posting_id)
        # print('neighbors:', o)
        # print('target:', csv.iloc[i].target)
        # print('------------------')
        preds_f1.append(o)
        o = csv.iloc[IDX_mAP].posting_id.values
        preds_mAP.append(o)

    csv['oof_cnn_f1'] = preds_f1
    csv['oof_cnn_mAP'] = preds_mAP

    csv['f1'] = csv.apply(getMetric('f1', 'oof_cnn_f1'), axis=1)
    # print('CV score for baseline =', csv.f1.mean())
    csv['mAP'] = csv.apply(getMetric('mAP@10', 'oof_cnn_mAP'), axis=1)
    # print('CV score for baseline =', csv.mAP.mean())
    csv['mrr'] = csv.apply(getMetric('mrr', 'oof_cnn_mAP'), axis=1)
    # print('CV score for baseline =', csv.mrr.mean())

    return csv.f1.mean(), csv.mAP.mean(), csv.mrr.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/cluster/home/hjjiang/PR-project/data/split_data/")
    parser.add_argument("--log_dir", type=str, default="/cluster/home/hjjiang/PR-project/log/image-only/pca/")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--n_neighbors", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    run_knn(args)

