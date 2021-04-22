#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 下午7:09
# @Author  : Lan Jiang
# @File    : bm25.py

from gensim.summarization import bm25
from utils.text_utils import load_corpus
import numpy as np
import argparse


class BM25(object):
    def __init__(self, corpus, idf=None):
        self.corpus = corpus
        self.model = bm25.BM25(corpus)
        self.idf = sum(map(lambda k: float(self.model.idf[k]), self.model.idf.keys())) / len(self.model.idf.keys()) if idf is None else idf

    def search(self, query, top_n=10):
        tok_query = query.strip().split()
        scores = self.model.get_scores(tok_query, self.idf)
        result_inds = np.argpartition(scores,-top_n)[-top_n:]
        return result_inds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../Dataset/shopee-price-match/")
    parser.add_argument("--cache_dir", type=str, default="../../Dataset/shopee-price-match/aux_files")
    args = parser.parse_args()

    corpus = load_corpus(args)
    bm25_model = BM25(corpus)
