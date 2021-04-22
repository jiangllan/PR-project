#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 下午7:09
# @Author  : Lan Jiang
# @File    : text_utils.py

import pandas as pd
import os
import pickle
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize


def preprocess_title(title):
    title = title.lower()
    # Remove Punctuation
    title = title.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespaces
    title = title.strip()
    # Tokenize
    tokens_title = word_tokenize(title)
    # Remove stopwords
    tokens_title = [word for word in tokens_title if not word in stopwords.words()]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemm_text = [lemmatizer.lemmatize(word) for word in tokens_title]
    prepped_title = ' '.join(lemm_text)
    return prepped_title


def build_corpus(data_dir):
    titles = pd.read_csv(os.path.join(data_dir, "train.csv"))['title'].tolist()
    std_titles = [preprocess_title(title) for title in titles]
    return std_titles


def load_corpus(args):
    aux_file = os.path.join(args.cache_dir, "corpus.pickle")
    if not os.path.exists(aux_file):
        print("building corpus matrix from raw data...")
        corpus = build_corpus(args.data_dir)
        with open(aux_file, "wb") as f:
            pickle.dump(corpus, f)
    else:
        print("load weights matrix from cached file: ", aux_file)
        with open(aux_file, "rb") as f:
            corpus = pickle.load(f)
        print("load over.")
    return corpus