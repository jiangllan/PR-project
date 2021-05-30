import os
import argparse
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from dataset import ShopeeImageDataset
from model import ShopeeImageEmbeddingNet


def get_image_tensor(args):

    for split_name in ['train', 'val', 'test']:
        args.csv_file = '{}.csv'.format(split_name)

        data = pd.read_csv(os.path.join(args.cache_dir, args.csv_file))
        data['image'] = args.img_dir + 'train_images/' + data['image']
        tmp = data.groupby('label_group').posting_id.agg('unique').to_dict()
        data['target'] = data.label_group.map(tmp)

        dataset = ShopeeImageDataset(
            data['image'].values,
            transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=1
        )

        image_tensor = []
        count = 0
        with torch.no_grad():
            for data in tqdm(loader):
                image_tensor.append(data)
                count += 1

        image_tensor = np.vstack(image_tensor)
        np.save(os.path.join(args.cache_dir, '{}_image_tensor_64.npy'.format(split_name)), image_tensor)


def get_feature(args):

    model = ShopeeImageEmbeddingNet(args.model_name)
    model = model.cuda()

    for split_name in ['train', 'val', 'test']:
        args.csv_file = '{}.csv'.format(split_name)

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
            num_workers=1
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
        np.save(os.path.join(args.cache_dir, '{}_{}_224_feature.npy'.format(split_name, args.model_name)), cnn_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--img_dir", type=str, default="../data/")
    parser.add_argument("--cache_dir", type=str, default="../data/split_data/")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--get_feature", action='store_true')
    args = parser.parse_args()
    if args.get_feature:
        get_feature(args)
    else:
        get_image_tensor(args)
