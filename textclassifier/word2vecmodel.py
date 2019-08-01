#!/usr/bin/env python
# coding=utf-8

import time
import gensim
import pandas as pd
import multiprocessing as mp
from utils import split_text


def train_word2vec(text_array, sg, size, window, min_count, num_threads, iterations):
    """
    a function to train Word2vec

    INPUT:
        text_array: list/numpy array/pandas Series of strings
        sg: Boolean --True for Skip-gram and False for CBOW
        size: Word2vec vector dimension
        window: maximum distance between the current and predicted word within a sentence.
        min_count: ignores all words with total frequency lower than this.
        num_threads: use these many worker threads to train the model
        iterations: number of iterations (epochs) over the corpus.
    OUTPUT:
        model: gensim Word2vec model object.
        df_vectors: a pandas dataframe which each column is the trained Word2vec vector of the token, each column name
                    is the corresponding token.
    """
    # split the word into array of words
    with mp.Pool(processes=num_threads) as pool:
        token_list = pool.map(split_text, text_array)

    # text[0:3]
    print("Number of documents imported: " + str(len(token_list)))

    # time recording
    start_time = time.time()

    # train Skip-gram model
    model = gensim.models.Word2Vec(sg=int(sg),
                                   size=size,
                                   window=window,
                                   min_count=min_count,
                                   workers=num_threads,
                                   iter=iterations)

    model.build_vocab(token_list)

    print('Start training models...')
    model.train(token_list, total_examples=len(token_list), epochs=model.iter)

    # save the model trained
    vocab = [z for z in model.wv.vocab]
    vectors = [model.wv[y] for y in vocab]

    df_vectors = pd.DataFrame(vectors, index=vocab)
    df_vectors = df_vectors.T

    end_time = int(time.time() - start_time)
    print(
        'Completed! Time taken: {:02d}:{:02d}:{:02d}'.format(end_time // 3600, (end_time % 3600 // 60), end_time % 60))

    return model, df_vectors
