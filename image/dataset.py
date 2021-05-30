import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset


class ShopeeImageDataset(Dataset):
    def __init__(self, img_path, transform):
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path)


class TripletShopeeImageDataset(Dataset):
    def __init__(self, csv, transform, train=True, init_distance=None):
        self.csv = csv
        self.transform = transform
        self.train = train
        if init_distance is not None:
            self.init_distanc

    def __getitem__(self, index):
        if self.train:
            anchor = Image.open(self.csv.iloc[index, ].image).convert('RGB')
            anchor = self.transform(anchor)

            anchor_target = self.csv.iloc[index, ].target
            positive_posting_id = np.random.choice(list(set(anchor_target) - set(self.csv.iloc[index, ].posting_id)))
            positive = Image.open(self.csv[self.csv.posting_id == positive_posting_id].image.values[0]).convert('RGB')
            positive = self.transform(positive)

            negative_posting_id = self.csv.iloc[index, ].posting_id
            while negative_posting_id in anchor_target:
                negative_posting_id = np.random.choice(self.csv.posting_id)

            negative = Image.open(self.csv[self.csv.posting_id == negative_posting_id].image.values[0]).convert('RGB')
            negative = self.transform(negative)
            return anchor, positive, negative
        else:
            img = Image.open(self.csv.iloc[index, ].image).convert('RGB')
            img = self.transform(img)
            return img

    def __len__(self):
        return len(self.csv)


if __name__ == '__main__':
    pass
    # train_csv = pd.read_csv(os.path.join('/cluster/home/hjjiang/PR-project/data/', 'new_split_data', 'new_train.csv'))
    # train_csv['image'] = '/cluster/home/hjjiang/PR-project/data/' + 'train_images/' + train_csv['image']
    # tmp = train_csv.groupby('label_group').posting_id.agg('unique').to_dict()
    # train_csv['target'] = train_csv.label_group.map(tmp)
    #
    # train_dataset = TripletShopeeImageDataset(
    #     train_csv,
    #     transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     train=True)
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True
    # )
    #
    # for i, (anchor, positive, negative) in enumerate(train_loader):
    #     print(anchor)
