DATA_PATH = '/cluster/home/hjjiang/PR-project/data/'
OUTPUT_PATH = '/cluster/home/hjjiang/PR-project/log/'

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from dataset import ShopeeImageDataset

from model import ShopeeImageEmbeddingNet

from sklearn.preprocessing import normalize

from metrics import average_precision


def getMetric(metric, col):
    if metric == 'f1':
        def f1score(row):
            print('row[col]:', row[col])
            print('row.target:', row.target)
            n = len(np.intersect1d(row.target, row[col]))
            print('f1 score:', 2 * n / (len(row.target) + len(row[col])))
            return 2 * n / (len(row.target) + len(row[col]))

        return f1score
    elif metric == 'mAP@10':
        def mAP_top_10(row):
            relevance = []
            # print('row[col]:', row[col])
            # print('row.target:', row.target)
            for i in range(len(row[col])):
                if row[col][i] in row.target:
                    relevance.append(1)
                else:
                    relevance.append(0)
            # print('relevance:', relevance)
            return average_precision(relevance)

        return mAP_top_10
    elif metric == 'mrr':
        def mean_reciprocal_rank(row):
            for i in range(len(row[col])):
                if row[col][i] in row.target:
                    return 1 / (i + 1)
            return 0

        return mean_reciprocal_rank
    else:
        raise NotImplementedError


def main():
    train = pd.read_csv(DATA_PATH + 'train.csv')
    train['image'] = DATA_PATH + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)

    preds_f1 = []
    preds_mAP = []
    CHUNK = 4096

    all_distances = np.load(os.path.join(OUTPUT_PATH, 'distance_matrix_resnet50_224.npy'))
    mask = 1 - np.diag(np.ones(all_distances.shape[0]))
    all_distances *= mask
    print(np.diagonal(all_distances))
    print('Loading pre-calculated distance matrix...')
    CTS = all_distances.shape[0] // CHUNK
    if all_distances.shape[0] % CHUNK != 0:
        CTS += 1

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, all_distances.shape[0])
        print('chunk', a, 'to', b)

        distances = all_distances[a:b]

        for k in range(b - a):
            IDX_f1 = np.where(distances[k, ] > 0.96)[0][:]
            IDX_mAP = np.argsort(distances[k, ])[::-1][:10]  # Top 10, except itself
            o = train.iloc[IDX_f1].posting_id.values
            preds_f1.append(o)
            o = train.iloc[IDX_mAP].posting_id.values
            preds_mAP.append(o)

    train['oof_cnn_f1'] = preds_f1
    train['oof_cnn_mAP'] = preds_mAP

    train['f1'] = train.apply(getMetric('f1', 'oof_cnn_f1'), axis=1)
    print('f1 score for baseline =', train.f1.mean())
    train['mAP'] = train.apply(getMetric('mAP@10', 'oof_cnn_mAP'), axis=1)
    print('mAP@10 score for baseline =', train.mAP.mean())
    train['mrr'] = train.apply(getMetric('mrr', 'oof_cnn_mAP'), axis=1)
    print('mrr@10 score for baseline =', train.mrr.mean())


if __name__ == '__main__':
    main()
