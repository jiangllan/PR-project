#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/27 上午11:12
# @Author  : Lan Jiang
# @File    : pairwise_generate.py

# -*- coding: utf-8 -*-

import random
import os

import numpy as np
import pandas as pd
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_similarity(t1, t2):
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [t1, t2]
    vectors = cv.fit_transform(corpus).toarray()
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

def generate(data_dir, split):
    data = pd.read_csv(os.path.join(data_dir, "%s.csv" % split))
    dic = {}

    for index, row in data.iterrows():
        if row['label_group'] not in dic.keys():
            dic[row['label_group']] = []
        dic[row['label_group']].append(row['std_title'])
    print("generate dic successfully")

    '''generate positive sentences pairs'''
    print("begin generate positive pairs...")
    pairs = []
    for i, label in enumerate(dic):
        size = len(dic[label])
        positive_list = random.Random(8).sample(range(0, size), int(size))
        for item1 in positive_list:
            for item2 in positive_list:
                if item1 < item2:
                    pairs.append([dic[label][item1], dic[label][item2], 1])

    pairs = np.array(pairs)
    sen_pair = pd.DataFrame({"title_1": pairs[:, 0], "title_2": pairs[:, 1], "label": pairs[:, 2]})
    print("# Positive pairs: ", len(pairs))
    print("write positive pairs to csv...")
    sen_pair.to_csv(os.path.join(data_dir, "pairwise_pos_%s.csv" % split), index=False)
    print("generate %s positive pairs successfully." % split)


if __name__ == '__main__':
    data_dir = "../data/split_data"
    for split in ['train', 'val', 'test']:
        generate(data_dir, split)