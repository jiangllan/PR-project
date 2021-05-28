#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 下午7:09
# @Author  : Lan Jiang
# @File    : text_utils.py

import pandas as pd
import numpy as np
import os
import pickle
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import progressbar
from sentence_transformers.readers import InputExample


def preprocess_title(title):
    title = title.lower()
    # Remove Punctuation
    title = title.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespaces and tokenize
    tokens_title = title.strip().split()
    # Remove stopwords
    tokens_title = [word for word in tokens_title if not word in stopwords.words()]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemm_text = [lemmatizer.lemmatize(word) for word in tokens_title]
    prepped_title = ' '.join(lemm_text)
    return prepped_title


def build_corpus(data_dir):
    titles = pd.read_csv(os.path.join(data_dir, "train.csv"))['title'].tolist()
    std_titles = []
    p = progressbar.ProgressBar()
    for i in p(range(len(titles))):
        title = titles[i]
        std_titles.append(preprocess_title(title))
    return std_titles


def load_corpus(args):
    aux_file = os.path.join(args.cache_dir, "tok_corpus.pickle")
    if not os.path.exists(aux_file):
        print("building corpus matrix from raw data...")
        corpus = build_corpus(args.data_dir)
        if not os.path.exists(args.cache_dir):
            os.mkdir(args.cache_dir)
        with open(aux_file, "wb") as f:
            pickle.dump(corpus, f)
        print("building corpus over.")
    else:
        print("load weights matrix from cached file: ", aux_file)
        with open(aux_file, "rb") as f:
            corpus = pickle.load(f)
        print("load over.")
    return corpus


def load_features(args, split):
    file_name = "new_%s" % split
    with open(os.path.join(args.model_dir, args.model_name, "%s_features.pickle" % file_name), "rb") as f:
        features = pickle.load(f)

    data = pd.read_csv(os.path.join(args.data_dir, "%s.csv" % file_name))
    labels = data['label_group'].to_numpy()
    features = np.array(features)
    return features, labels


def load_data(args, split):
    file_name = "new_%s" % split
    data = pd.read_csv(os.path.join(args.data_dir, "%s.csv" % file_name))
    return data


def load_pairwise_data(args, split):
    data = pd.read_csv(os.path.join(args.data_dir, "pairwise_pos_%s.csv" % split))
    train_samples = []
    for index, row in data.iterrows():
        train_samples.append(InputExample(texts=[row['title_1'], row['title_2']], label=1))
        train_samples.append(InputExample(texts=[row['title_2'], row['title_1']], label=1))

    return train_samples