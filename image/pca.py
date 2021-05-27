import os
import torch
import argparse
import numpy as np
import pandas as pd
from metrics import getMetric
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def run_pca_wo_multiple_split(args):
    n_components = [2048, 1024, 512, 256, 128, 64, 32, 16]
    # n_components = [512, 256, 128, 64, 32, 16]
    use_prev_n_components = []
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
    prev_n_component = []

    train = pd.read_csv(os.path.join(args.cache_dir, 'new_train.csv'))
    train['image'] = args.cache_dir + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    train_feature = np.load(os.path.join(args.cache_dir, 'train_split_{}_{}_224_feature.npy'.format(split_index, args.model_name)))

    val = pd.read_csv(os.path.join(args.cache_dir, 'new_val.csv'))
    val['image'] = args.cache_dir + 'train_images/' + val['image']
    tmp = val.groupby('label_group').posting_id.agg('unique').to_dict()
    val['target'] = val.label_group.map(tmp)
    val_feature = np.load(os.path.join(args.cache_dir, 'dev_split_{}_{}_224_feature.npy'.format(split_index, args.model_name)))

    test = pd.read_csv(os.path.join(args.cache_dir, 'new_test.csv'))
    test['image'] = args.cache_dir + 'train_images/' + test['image']
    tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
    test['target'] = test.label_group.map(tmp)
    test_feature = np.load(os.path.join(args.cache_dir, 'test_split_{}_{}_224_feature.npy'.format(split_index, args.model_name)))

    for n_component in n_components:
        pca = PCA(n_components=n_component).fit(train_feature)
        train_features_reduce = normalize(pca.transform(train_feature))
        val_features_reduce = normalize(pca.transform(val_feature))
        test_features_reduce = normalize(pca.transform(test_feature))

        train_features_reduce = torch.from_numpy(train_features_reduce)
        train_features_reduce = train_features_reduce.cuda()
        print('train_features_reduce size = {}'.format(train_features_reduce.size()))
        val_features_reduce = torch.from_numpy(val_features_reduce)
        val_features_reduce = val_features_reduce.cuda()
        print('val_features_reduce size = {}'.format(val_features_reduce.size()))
        test_features_reduce = torch.from_numpy(test_features_reduce)
        test_features_reduce = test_features_reduce.cuda()
        print('test_features_reduce size = {}'.format(val_features_reduce.size()))

        if len(use_prev_n_components) == 0:
            split_id.append(split_index)
            component.append(n_component)

            f1, mAP, mrr = cos_similarity(train_features_reduce, train)
            train_f1_wo_itself.append(f1)
            train_mAP_wo_itself.append(mAP)
            train_mrr_wo_itself.append(mrr)
            print('\t train data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP,
                                                                                               mrr))
            f1, mAP, mrr = cos_similarity(val_features_reduce, val)
            val_f1_wo_itself.append(f1)
            val_mAP_wo_itself.append(mAP)
            val_mrr_wo_itself.append(mrr)
            print('\t val data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP,
                                                                                             mrr))
            f1, mAP, mrr = cos_similarity(test_features_reduce, test)
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
        else:
            for use_prev_n_component in use_prev_n_components:
                component.append(n_component)
                prev_n_component.append(use_prev_n_component)

                f1, mAP, mrr = cos_similarity(train_features_reduce[:, :use_prev_n_component], train)
                train_f1_wo_itself.append(f1)
                train_mAP_wo_itself.append(mAP)
                train_mrr_wo_itself.append(mrr)
                print('\t train data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP, mrr))
                f1, mAP, mrr = cos_similarity(val_features_reduce[:, :use_prev_n_component], val)
                val_f1_wo_itself.append(f1)
                val_mAP_wo_itself.append(mAP)
                val_mrr_wo_itself.append(mrr)
                print('\t val data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP, mrr))
                f1, mAP, mrr = cos_similarity(test_features_reduce[:, :use_prev_n_component], test)
                test_f1_wo_itself.append(f1)
                test_mAP_wo_itself.append(mAP)
                test_mrr_wo_itself.append(mrr)
                print('\t test data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP, mrr))

                df = pd.DataFrame(
                    {'train_f1_wo_itself': train_f1_wo_itself, 'train_mAP_wo_itself': train_mAP_wo_itself,
                     'train_mrr_wo_itself': train_mrr_wo_itself, 'val_f1_wo_itself': val_f1_wo_itself,
                     'val_mAP_wo_itself': val_mAP_wo_itself, 'val_mrr_wo_itself': val_mrr_wo_itself,
                     'test_f1_wo_itself': test_f1_wo_itself, 'test_mAP_wo_itself': test_mAP_wo_itself,
                     'test_mrr_wo_itself': test_mrr_wo_itself, 'component': component,
                     'prev_n_component': prev_n_component})

                log_file = os.path.join(args.log_dir, args.log_name)
                with open(log_file, "w") as f:
                    df.to_csv(f)



def run_pca(args):
    # n_components = [2048, 1024, 512, 256, 128, 64, 32, 16]
    n_components = [512, 256, 128, 64, 32, 16]
    use_prev_n_components = []
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
    prev_n_component = []
    split_id = []

    for split_index in range(1, args.n_splits + 1):
        print('--------------- SPLIT {} ---------------'.format(split_index))
        train = pd.read_csv(os.path.join(args.cache_dir, 'train_split_{}.csv'.format(split_index)))
        train['image'] = args.cache_dir + 'train_images/' + train['image']
        tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
        train['target'] = train.label_group.map(tmp)
        train_feature = np.load(os.path.join(args.cache_dir, 'train_split_{}_{}_224_feature.npy'.format(split_index, args.model_name)))

        val = pd.read_csv(os.path.join(args.cache_dir, 'dev_split_{}.csv'.format(split_index)))
        val['image'] = args.cache_dir + 'train_images/' + val['image']
        tmp = val.groupby('label_group').posting_id.agg('unique').to_dict()
        val['target'] = val.label_group.map(tmp)
        val_feature = np.load(os.path.join(args.cache_dir, 'dev_split_{}_{}_224_feature.npy'.format(split_index, args.model_name)))

        test = pd.read_csv(os.path.join(args.cache_dir, 'test_split_{}.csv'.format(split_index)))
        test['image'] = args.cache_dir + 'train_images/' + test['image']
        tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
        test['target'] = test.label_group.map(tmp)
        test_feature = np.load(os.path.join(args.cache_dir, 'test_split_{}_{}_224_feature.npy'.format(split_index, args.model_name)))

        for n_component in n_components:
            pca = PCA(n_components=n_component).fit(train_feature)
            train_features_reduce = normalize(pca.transform(train_feature))
            val_features_reduce = normalize(pca.transform(val_feature))
            test_features_reduce = normalize(pca.transform(test_feature))

            train_features_reduce = torch.from_numpy(train_features_reduce)
            train_features_reduce = train_features_reduce.cuda()
            print('train_features_reduce size = {}'.format(train_features_reduce.size()))
            val_features_reduce = torch.from_numpy(val_features_reduce)
            val_features_reduce = val_features_reduce.cuda()
            print('val_features_reduce size = {}'.format(val_features_reduce.size()))
            test_features_reduce = torch.from_numpy(test_features_reduce)
            test_features_reduce = test_features_reduce.cuda()
            print('test_features_reduce size = {}'.format(val_features_reduce.size()))

            if len(use_prev_n_components) == 0:
                split_id.append(split_index)
                component.append(n_component)

                f1, mAP, mrr = cos_similarity(train_features_reduce, train)
                train_f1_wo_itself.append(f1)
                train_mAP_wo_itself.append(mAP)
                train_mrr_wo_itself.append(mrr)
                print('\t train data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP,
                                                                                                   mrr))
                f1, mAP, mrr = cos_similarity(val_features_reduce, val)
                val_f1_wo_itself.append(f1)
                val_mAP_wo_itself.append(mAP)
                val_mrr_wo_itself.append(mrr)
                print('\t val data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP,
                                                                                                 mrr))
                f1, mAP, mrr = cos_similarity(test_features_reduce, test)
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
                     'test_mrr_wo_itself': test_mrr_wo_itself, 'split_id': split_id, 'component': component})

                log_file = os.path.join(args.log_dir, args.log_name)
                with open(log_file, "w") as f:
                    df.to_csv(f)
            else:
                for use_prev_n_component in use_prev_n_components:
                    split_id.append(split_index)
                    component.append(n_component)
                    prev_n_component.append(use_prev_n_component)

                    f1, mAP, mrr = cos_similarity(train_features_reduce[:, :use_prev_n_component], train)
                    train_f1_wo_itself.append(f1)
                    train_mAP_wo_itself.append(mAP)
                    train_mrr_wo_itself.append(mrr)
                    print('\t train data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP, mrr))
                    f1, mAP, mrr = cos_similarity(val_features_reduce[:, :use_prev_n_component], val)
                    val_f1_wo_itself.append(f1)
                    val_mAP_wo_itself.append(mAP)
                    val_mrr_wo_itself.append(mrr)
                    print('\t val data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP, mrr))
                    f1, mAP, mrr = cos_similarity(test_features_reduce[:, :use_prev_n_component], test)
                    test_f1_wo_itself.append(f1)
                    test_mAP_wo_itself.append(mAP)
                    test_mrr_wo_itself.append(mrr)
                    print('\t test data with {} components f1/mAP/mrr = {:.4f}/{:.4f}/{:.4f} '.format(n_component, f1, mAP, mrr))

                    df = pd.DataFrame(
                        {'train_f1_wo_itself': train_f1_wo_itself, 'train_mAP_wo_itself': train_mAP_wo_itself,
                         'train_mrr_wo_itself': train_mrr_wo_itself, 'val_f1_wo_itself': val_f1_wo_itself,
                         'val_mAP_wo_itself': val_mAP_wo_itself, 'val_mrr_wo_itself': val_mrr_wo_itself,
                         'test_f1_wo_itself': test_f1_wo_itself, 'test_mAP_wo_itself': test_mAP_wo_itself,
                         'test_mrr_wo_itself': test_mrr_wo_itself, 'split_id': split_id, 'component': component,
                         'prev_n_component': prev_n_component})

                    log_file = os.path.join(args.log_dir, args.log_name)
                    with open(log_file, "w") as f:
                        df.to_csv(f)


def cos_similarity(feature, csv):
    preds_f1 = []
    preds_mAP = []
    CHUNK = 4096

    # print('Finding similar images...')
    CTS = len(feature) // CHUNK
    if len(feature) % CHUNK != 0:
        CTS += 1

    all_distances = []
    for j in range(CTS):
        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(feature))
        # print('chunk', a, 'to', b)

        distances = torch.matmul(feature, feature[a:b].T).T
        distances = distances.data.cpu().numpy()
        all_distances.append(distances)

    all_distances = np.vstack(all_distances)
    # np.save(os.path.join(OUTPUT_PATH, 'distance_matrix_resnet50_224.npy'), all_distances)

    mask = 1 - np.diag(np.ones(all_distances.shape[0]))
    all_distances *= mask
    # print(np.diagonal(all_distances))

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(feature))
        # print('chunk', a, 'to', b)

        distances = all_distances[a:b]

        for k in range(b - a):
            IDX_f1 = np.where(distances[k, ] > 0.95)[0][:]
            # print(type(IDX_f1))
            # print(IDX_f1.shape)
            IDX_mAP = np.argsort(distances[k, ])[::-1][:10]  # Top 10, except itself
            o = csv.iloc[IDX_f1].posting_id.values
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
    parser.add_argument("--log_name", type=str, default="log.txt")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    run_pca(args)
