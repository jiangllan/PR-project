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


if __name__ == '__main__':
    data_dir = "../data/split_data"
    data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    dic = {}
    P_Num = 0

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
    sen_pair.to_csv(os.path.join(data_dir, "pairwise_pos_train.csv"), index=False)
    print("generate positive pairs successfully.")

    # '''generate nagetive sentences pairs'''
    # print("begin generate negative pairs...")
    # total = 43724
    # topic_num = len(dic)
    # dic_v = list(dic.values())
    # negative_list = random.Random(8).sample(range(0, topic_num), int(topic_num - 1))
    # for topicIndex1 in negative_list:
    #     for topicIndex2 in negative_list:
    #         if total < 0:
    #             break
    #         if topicIndex1 < topicIndex2:
    #             # print(topicIndex1, topicIndex2)
    #             size1 = len(dic_v[topicIndex1])
    #             size2 = len(dic_v[topicIndex2])
    #             sen1Index1_list = random.Random(8).sample(range(0, size1 - 1), size1 - 1)
    #             sen1Index2_list = random.Random(8).sample(range(0, size2 - 1), size2 - 1)
    #             for sen1Index1 in sen1Index1_list:
    #                 for sen1Index2 in sen1Index2_list:
    #                     t_s1 = dic_v[topicIndex1][sen1Index1]
    #                     t_s2 = dic_v[topicIndex2][sen1Index2]
    #                     sim_score = tfidf_similarity(t_s1, t_s2)
    #                     if t_s1 != '' and t_s2 != '' and sim_score > 0.2:
    #                         print(t_s1, " ", t_s2)
    #                         add_data = [t_s1, t_s2, 0]
    #                         if total < 2:
    #                             print("Pos sample: ", add_data)
    #                         insertRow = pd.DataFrame([add_data])
    #                         sen_pair = sen_pair.append(insertRow, ignore_index=True)
    #                         total -= 1
    #                         print(total)
    # print("generate negative pairs successfully")

    # print("write to csv...")
    # sen_pair = sen_pair.sample(frac=1)
    # sen_pair.to_csv(os.path.join(data_dir, "pairwise_train.csv"), index=False)
