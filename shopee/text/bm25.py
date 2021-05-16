#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 下午7:09
# @Author  : Lan Jiang
# @File    : bm25.py

from gensim.summarization import bm25
import numpy as np


class BM25(object):
    def __init__(self, corpus, idf=None):
        self.corpus = corpus
        self.model = bm25.BM25(corpus)
        self.idf = sum(map(lambda k: float(self.model.idf[k]), self.model.idf.keys())) / len(self.model.idf.keys()) if idf is None else idf

    def search(self, query):
        tok_query = query.strip().split()
        scores = self.model.get_scores(tok_query)
        return np.array(scores)
