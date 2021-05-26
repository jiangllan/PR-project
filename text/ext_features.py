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
            os.path.join(args.data_dir, "../../", "mymodel.bin"), binary=True)
        self.pool = args.pool

    def encode(self, titles):
        embedding_corpus = []
        for sentence in titles:
            embedding_sentences = [self.model[word] if word in self.model else self.model['UNK'] for word in
                                   sentence.split()]
            if self.pool == "avg":
                embedding_sentences = np.mean(embedding_sentences, axis=0)
            elif self.pool == "max":
                embedding_sentences = np.max(embedding_sentences, axis=0)
            else:
                pass
            embedding_corpus.append(embedding_sentences)
        return embedding_corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-product-matching/split_data")
    parser.add_argument("--cache_dir", type=str, default="../../Dataset/shopee-product-matching/aux_files")
    parser.add_argument("--model_name", type=str, default="xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    parser.add_argument("--pool", type=str, default="max")
    parser.add_argument("--method", type=str, default="bert")
    args = parser.parse_args()

    # load model
    if args.method == "bert":
        model = SentenceTransformer(args.model_name)
    elif args.method == "glove":
        model = GloveEncoder(args)
    else:
        model = None
    print("load model over.")

    for i in range(5):
        for split in ['train', 'test', 'dev']:
            file_name = "%s_split_%d" % (split, i + 1)
            data = pd.read_csv(os.path.join(args.data_dir, "%s.csv" % file_name))
            titles = data['std_title'].tolist()
            titles_embeddings = None
            for j, item in enumerate(titles):
                try:
                    assert (isinstance(item, str))
                except AssertionError:
                    print(j, " ", item)

            titles_embeddings = model.encode(titles)

            # save to cache files
            with open(os.path.join(args.data_dir, args.method, "%s_features.pickle" % file_name), "wb") as f:
                pickle.dump(titles_embeddings, f)
            print("save %s embeddings produced by %s over." % (file_name, args.model_name))

    # # save
    # cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    # top_results = torch.topk(cos_scores, k=top_k)
