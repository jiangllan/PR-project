import os
import argparse
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from dataset import ShopeeImageDataset
from model import ShopeeImageEmbeddingNet


def main_wo_multiple_split(args):

    model = ShopeeImageEmbeddingNet(args.model_name)
    model = model.cuda()

    for split_name in ['train', 'val', 'test']:
        args.csv_file = 'new_{}.csv'.format(split_name)

        data = pd.read_csv(os.path.join(args.cache_dir, args.csv_file))
        data['image'] = args.img_dir + 'train_images/' + data['image']
        tmp = data.groupby('label_group').posting_id.agg('unique').to_dict()
        data['target'] = data.label_group.map(tmp)

        dataset = ShopeeImageDataset(
            data['image'].values,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=32
        )

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
        np.save(os.path.join(args.cache_dir, 'new_{}_{}_224_feature.npy'.format(split_name, args.model_name)), cnn_feature)


def main(args):

    model = ShopeeImageEmbeddingNet(args.model_name)
    model = model.cuda()

    for split_index in range(1, args.n_splits + 1):
        for split_name in ['train', 'test', 'dev']:
            args.csv_file = '{}_split_{}.csv'.format(split_name, split_index)

            data = pd.read_csv(args.cache_dir + args.csv_file)
            data['image'] = args.img_dir + 'train_images/' + data['image']
            tmp = data.groupby('label_group').posting_id.agg('unique').to_dict()
            data['target'] = data.label_group.map(tmp)

            dataset = ShopeeImageDataset(
                data['image'].values,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]))

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                num_workers=32
            )

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
            np.save(os.path.join(args.cache_dir, '{}_split_{}_{}_224_feature.npy'.format(split_name, split_index, args.model_name)), cnn_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train_split_1.csv")
    parser.add_argument("--img_dir", type=str, default="/cluster/home/hjjiang/PR-project/data/")
    parser.add_argument("--cache_dir", type=str, default="/cluster/home/hjjiang/PR-project/data/split_data/")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    # main(args)
    main_wo_multiple_split(args)
