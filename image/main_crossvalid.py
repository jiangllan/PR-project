import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from metrics import average_precision
from dataset import ShopeeImageDataset
from model import ShopeeImageEmbeddingNet
from sklearn.preprocessing import normalize


def getMetric(metric, col):
    if metric == 'f1':
        def f1score(row):
            n = len(np.intersect1d(row.target, row[col]))
            return 2 * n / (len(row.target) + len(row[col]))
        return f1score
    elif metric == 'mAP@10':
        def mAP_top_10(row):
            relevance = []
            for i in range(len(row[col])):
                if row[col][i] in row.target:
                    relevance.append(1)
                else:
                    relevance.append(0)
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

    print('train shape is', train.shape)
    print(train.head())
    print(train.loc[34249])

    # tmp = train.groupby('image_phash').posting_id.agg('unique').to_dict()
    # train['oof_hash'] = train.image_phash.map(tmp)
    #
    # train['f1'] = train.apply(getMetric('oof_hash'), axis=1)
    # print('CV score for baseline =', train.f1.mean())

    dataset = ShopeeImageDataset(
        train['image'].values,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16
    )

    model = ShopeeImageEmbeddingNet()
    model = model.cuda()

    cnn_feature = []
    count = 0
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.cuda()
            feat = model(data)
            feat = feat.reshape(feat.shape[0], feat.shape[1])
            feat = feat.data.cpu().numpy()
            cnn_feature.append(feat)
            count += 1

    # l2 norm to kill all the sim in 0-1
    cnn_feature = np.vstack(cnn_feature)
    np.save(os.path.join(OUTPUT_PATH, 'resnet50_224_feature.npy'), cnn_feature)
    cnn_feature = normalize(cnn_feature)
    cnn_feature = torch.from_numpy(cnn_feature)
    cnn_feature = cnn_feature.cuda()

    preds_f1 = []
    preds_mAP = []
    CHUNK = 4096

    print('Finding similar images...')
    CTS = len(cnn_feature) // CHUNK
    if len(cnn_feature) % CHUNK != 0:
        CTS += 1

    all_distances = []
    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(cnn_feature))
        print('chunk', a, 'to', b)

        distances = torch.matmul(cnn_feature, cnn_feature[a:b].T).T
        distances = distances.data.cpu().numpy()
        all_distances.append(distances)

    all_distances = np.vstack(all_distances)
    np.save(os.path.join(OUTPUT_PATH, 'distance_matrix_resnet50_224.npy'), all_distances)

    mask = 1 - np.diag(np.ones(all_distances.shape[0]))
    all_distances *= mask
    print(np.diagonal(all_distances))

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(cnn_feature))
        print('chunk', a, 'to', b)

        distances = all_distances[a:b]

        for k in range(b - a):
            IDX_f1 = np.where(distances[k, ] > 0.95)[0][:]
            IDX_mAP = np.argsort(distances[k, ])[::-1][:10]  # Top 10, except itself
            o = train.iloc[IDX_f1].posting_id.values
            preds_f1.append(o)
            o = train.iloc[IDX_mAP].posting_id.values
            preds_mAP.append(o)

    del cnn_feature, model

    train['oof_cnn_f1'] = preds_f1
    train['oof_cnn_mAP'] = preds_mAP

    train['f1'] = train.apply(getMetric('f1', 'oof_cnn_f1'), axis=1)
    print('CV score for baseline =', train.f1.mean())
    train['mAP'] = train.apply(getMetric('mAP@10', 'oof_cnn_mAP'), axis=1)
    print('CV score for baseline =', train.mAP.mean())
    train['mrr'] = train.apply(getMetric('mrr', 'oof_cnn_mAP'), axis=1)
    print('CV score for baseline =', train.mrr.mean())


if __name__ == '__main__':

    main()
