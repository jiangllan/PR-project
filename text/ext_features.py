#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/14 下午12:15
# @Author  : Lan Jiang
# @File    : ext_features.py

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer


class GloveEncoder(object):
    def __init__(self, args):
        self.model = KeyedVectors.load_word2vec_format(
            os.path.join(args.result_dir, "mymodel.bin"), binary=True)

    def encode(self, titles):
        embedding_corpus = []
        for sentence in titles:
            embedding_sentences = [self.model[word] if word in self.model else self.model['UNK'] for word in
                                   sentence.split()]
            mean_embedding_sentences = np.mean(embedding_sentences, axis=0)
            max_embedding_sentences = np.max(embedding_sentences, axis=0)
            cat_embedding_sentences = np.concatenate((max_embedding_sentences, mean_embedding_sentences), axis=0)
            print(mean_embedding_sentences.shape, cat_embedding_sentences.shape)

            embedding_corpus.append(cat_embedding_sentences)
        return embedding_corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/split_data")
    parser.add_argument("--result_dir", type=str, default="../result/text")
    parser.add_argument("--model_name", type=str, default="glove")
    args = parser.parse_args()

    # load model
    if args.model_name == "glove":
        model = GloveEncoder(args)
    else:
        model = SentenceTransformer(args.model_name)
    print("load model over.")

    save_dir = os.path.join(args.result_dir, args.model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for split in ['train', 'test', 'val']:
        data = pd.read_csv(os.path.join(args.data_dir, "%s.csv" % split))
        titles = data['std_title'].tolist()
        # for j, item in enumerate(titles):
        #     try:
        #         assert (isinstance(item, str))
        #     except AssertionError:
        #         print(j, " ", item)
        titles_embeddings = model.encode(titles)

        with open(os.path.join(save_dir, "%s_features.pickle" % split), "wb") as f:
            pickle.dump(titles_embeddings, f)
        print("save %s embeddings produced by %s over." % (split, args.model_name))

    # # save
    # cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    # top_results = torch.topk(cos_scores, k=top_k)
