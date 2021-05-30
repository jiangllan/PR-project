import os
import torch
import argparse
import numpy as np
import pandas as pd
from utils import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def run_pca(args):
    n_components = [512, 128, 32]
    train_f1_wo_itself = []
    train_mAP_wo_itself = []
    train_mrr_wo_itself = []
    val_f1_wo_itself = []
    val_mAP_wo_itself = []
    val_mrr_wo_itself = []
    test_f1_wo_itself = []
    test_mAP_wo_itself = []
    test_mrr_wo_itself = []
    component = []

    train = pd.read_csv(os.path.join(args.cache_dir, 'train.csv'))
    train['image'] = args.cache_dir + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    train_feature = np.load(os.path.join(args.cache_dir, 'train_image_tensor_{}.npy'.format(args.image_size)))
    train_feature = train_feature.reshape(train_feature.shape[0], -1)

    val = pd.read_csv(os.path.join(args.cache_dir, 'val.csv'))
    val['image'] = args.cache_dir + 'train_images/' + val['image']
    tmp = val.groupby('label_group').posting_id.agg('unique').to_dict()
    val['target'] = val.label_group.map(tmp)
    val_feature = np.load(os.path.join(args.cache_dir, 'val_image_tensor_{}.npy'.format(args.image_size)))
    val_feature = val_feature.reshape(val_feature.shape[0], -1)

    test = pd.read_csv(os.path.join(args.cache_dir, 'test.csv'))
    test['image'] = args.cache_dir + 'train_images/' + test['image']
    tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
    test['target'] = test.label_group.map(tmp)
    test_feature = np.load(os.path.join(args.cache_dir, 'test_image_tensor_{}.npy'.format(args.image_size)))
    test_feature = test_feature.reshape(test_feature.shape[0], -1)

    for n_component in n_components:
        print('Current component = {}'.format(n_component))
        pca = PCA(n_components=n_component).fit(train_feature)
        train_features_reduce = normalize(pca.transform(train_feature))
        val_features_reduce = normalize(pca.transform(val_feature))
        test_features_reduce = normalize(pca.transform(test_feature))

        train_features_reduce = torch.from_numpy(train_features_reduce)
        val_features_reduce = torch.from_numpy(val_features_reduce)
        test_features_reduce = torch.from_numpy(test_features_reduce)

        component.append(n_component)

        f1, mAP, mrr = cos_similarity(train_features_reduce, train, drop_itself=args.drop_itself,
                                      f1_threshold=args.f1_threshold, mAP_threshold=args.mAP_threshold)
        train_f1_wo_itself.append(f1)
        train_mAP_wo_itself.append(mAP)
        train_mrr_wo_itself.append(mrr)
        print('\t train data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP,
                                                                                           mrr))

        f1, mAP, mrr = cos_similarity(val_features_reduce, val, drop_itself=args.drop_itself,
                                      f1_threshold=args.f1_threshold, mAP_threshold=args.mAP_threshold)
        val_f1_wo_itself.append(f1)
        val_mAP_wo_itself.append(mAP)
        val_mrr_wo_itself.append(mrr)
        print('\t val data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP,
                                                                                         mrr))

        f1, mAP, mrr = cos_similarity(test_features_reduce, test, drop_itself=args.drop_itself,
                                      f1_threshold=args.f1_threshold, mAP_threshold=args.mAP_threshold)
        test_f1_wo_itself.append(f1)
        test_mAP_wo_itself.append(mAP)
        test_mrr_wo_itself.append(mrr)
        print('\t test data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP,
                                                                                          mrr))

        df = pd.DataFrame(
            {'train_f1_wo_itself': train_f1_wo_itself, 'train_mAP_wo_itself': train_mAP_wo_itself,
             'train_mrr_wo_itself': train_mrr_wo_itself, 'val_f1_wo_itself': val_f1_wo_itself,
             'val_mAP_wo_itself': val_mAP_wo_itself, 'val_mrr_wo_itself': val_mrr_wo_itself,
             'test_f1_wo_itself': test_f1_wo_itself, 'test_mAP_wo_itself': test_mAP_wo_itself,
             'test_mrr_wo_itself': test_mrr_wo_itself, 'component': component})

        log_file = os.path.join(args.log_dir, args.log_name)
        with open(log_file, "w") as f:
            df.to_csv(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="../data/split_data/")
    parser.add_argument("--log_dir", type=str, default="../result/final-image-only/pca/")
    parser.add_argument("--log_name", type=str, default="log.txt")
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--drop_itself", action="store_true", default=False)
    parser.add_argument("--f1_threshold", type=float, default=0.95)
    parser.add_argument("--mAP_threshold", type=float, default=0.95)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    args.log_name = 'log_pca_on_raw_image-{}_split_dropitself-{}_f1thres-{}_mAPthres-{}.txt'.format(
        args.image_size, args.drop_itself, args.f1_threshold, args.mAP_threshold)

    run_pca(args)
