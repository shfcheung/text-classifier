#!/usr/bin/env python
# coding=utf-8

"""Helper functions"""

import re
import bert
import jieba
import logging
import feather
import operator
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
from functools import partial
from dbfread import DBF
from opencc import OpenCC
from bert import tokenization
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# disable warnings in jieba packages
jieba.setLogLevel(logging.CRITICAL)


def normalize_full_width(text):
    """
    a function to normalize full width characters
    """
    return unicodedata.normalize('NFKC', text)


def sep_chi_eng(text):
    """
    a function to separate adjacent Chinese and English words from each other with a space character
    e.g. sep_chi_eng("apple手機iphone") = "apple 手機 iphone"
    """
    text = re.sub("([a-z])([\u2E80-\u2FD5\u3400-\u4DBF\u4E00-\u9FCC])", "\\1 \\2", text)
    text = re.sub("([\u2E80-\u2FD5\u3400-\u4DBF\u4E00-\u9FCC])([a-z])", "\\1 \\2", text)
    return text


def segment_chi(text):
    """
    a function to segment Chinese text using jieba
    e.g. segment_chi("蘋果手機") = "蘋果 手機"
    """
    return ' '.join(jieba.cut(text, cut_all=False))


def case_lower(text):
    """
    a function to convert English letters to lower case
    """
    return text.lower()


def remove_punct(text):
    """
    a function to remove punctuation
    """
    return re.sub("[^\w\s]", " ", text)


def detect_chi(text):
    """
    a function to return True if text containing at least 1 Chinese character, False otherwise.
    """
    return bool(re.search(u"[\u2E80-\u2FD5\u3400-\u4DBF\u4E00-\u9FCC]", text))


class ModifiedWordPieceTokenizer(object):
    """
    a text tokenizer for performing modified Word Piece segmentation. The tokenizer is modified based on BERT's.
    It preserves the segmented Chinese instead of converting some of them to the single and special [UKN] token.

    Package required: bert, re
    """

    def __init__(self, bert_vocab_file, do_lower_case):
        self.bert_vocab_file = bert_vocab_file
        self.do_lower_case = do_lower_case
        self.tokenizer = bert.tokenization.FullTokenizer(vocab_file=self.bert_vocab_file,
                                                         do_lower_case=self.do_lower_case)

    def tokenize(self, text):
        """
        tokenize the input text

        INPUT:
            text: string
        OUTPUT:
            string
        """
        wordPieces = self.tokenizer.tokenize(text)
        tokens = text.lower().split()

        i = 0  # index for tokens
        j = 0  # index for wordPieces
        string = ''  # for storing the concatenation of wordPieces
        output = []  # empty list to store the English wordPieces and concatenated Chinese vocabularies

        for wordPiece in wordPieces:
            wordPiece = re.sub('##', '',
                               wordPiece)  # substitute the special "##" symbol for concatenation of wordPieces later on
            if wordPiece == '[UNK]':
                wordPiece = '難'  # substitute the [UNK] character for Chinese to one single arbitray Chinese word 難
            string += wordPiece  # add the wordPiece to string

            # match base on comparing length of token and string but not at textual level due to presence of [UKN]
            # character/ the arbitray character is created such that the unknown Chinese word still remain a word
            # length of 1
            if len(string) == len(tokens[i]):
                if bool(re.search(u"[\u2E80-\u2FD5\u3400-\u4DBF\u4E00-\u9FCC]|[UNK]", string)):
                    # append token instead of string due to possible presence of unknown
                    # Chinese character in string
                    output.append(tokens[i])
                else:
                    # append English wordPieces
                    output.append(wordPieces[j])
                i += 1  # after matching 1 token, increase the token index by 1
                string = ''  # reset the string for comparison in the next token
            elif not bool(re.search(u"[\u2E80-\u2FD5\u3400-\u4DBF\u4E00-\u9FCC]", wordPiece)):
                # append English wordPieces
                output.append(wordPieces[j])

            j += 1  # next wordPieces

        return ' '.join(output)


def text_cleaning(text_array, word_tokenization, bert_vocab_file, do_lower_case, num_threads=1):
    """
    a function to clean text, including
    - Normalize full width characters
    - Convert all English letters to lower case
    - Translate Chinese simplified characters to traditional Chinese
    - Separate Chinese and English tokens from each other
    - Segment Chinese text
    - remove punctuation
    - perform the special Word Piece Segmentation if specified by the argument "word_tokenization"

    INPUT:
        text_array: list/numpy array/pandas Series
        word_tokenization: method of word segmentation, either split by "space" or "word_piece" tokenization
        bert_vocab_file: string --path to the .txt file of vocabularies for modified Word Piece Tokenizer
        do_lower_case: boolean --whether the modified Word Piece Tokenizer converts English letters to lower case or not
        num_threads: int --number of CPU processors for performing the text cleaning
    OUTPUT:
        text_array: a list of string (cleaned text)
    """

    with mp.Pool(processes=num_threads) as pool:

        print("   Normalizing full-width characters...")
        text_array = pool.map(normalize_full_width, text_array)

        print("   Converting English letters to lower case...")
        text_array = pool.map(case_lower, text_array)

        print("   Translating Simplified Chinese to Traditional...")
        bool_contain_chi = pool.map(detect_chi, text_array)
        text_array = np.array(text_array)
        cc = OpenCC('s2t')
        text_array[bool_contain_chi] = list(map(lambda x: cc.convert(x), text_array[bool_contain_chi]))

        print("   Separating Chinese amd English word tokens from each other...")
        text_array = pool.map(sep_chi_eng, text_array)

        print("   Segmenting Chinese vocabularies...")
        text_array = np.array(text_array)
        dtype_0 = str(text_array.dtype)
        dtype_len_0 = int(re.findall("[0-9]+", dtype_0)[0])

        if sum(bool_contain_chi):
            replacement = pool.map(segment_chi, text_array[bool_contain_chi])
            replacement = np.array(replacement)
            dtype_1 = str(replacement.dtype)
            dtype_len_1 = int(re.findall("[0-9]+", dtype_1)[0])

            # change the type of text_array to that of the replacement
            # in order to avoid some of the characters in the segmented text being trimmed
            if dtype_len_0 > dtype_len_1:
                text_array = text_array.astype(dtype_0)
            else:
                text_array = text_array.astype(dtype_1)

            text_array[bool_contain_chi] = replacement

        print("   Removing punctuation...")
        text_array = pool.map(remove_punct, text_array)


        if word_tokenization == "word_piece":
            print("   Word Piece Segmentation...")
            tokenizer = ModifiedWordPieceTokenizer(bert_vocab_file=bert_vocab_file,
                                                   do_lower_case=do_lower_case)
            text_array = pool.map(tokenizer.tokenize, text_array)
        elif word_tokenization == "space":
            return text_array

    return text_array


def tokens_to_index(token_list, vocab_dict):
    """
    convert word tokens to number index

    INPUT:
        token_list: a list of string tokens
        vocab_dict: a dictionary with tokens(key)-integer(value) pairs
    OUTPUT:
        index_list: a list of integer indices
    """
    index_list = []
    for x in token_list:
        try:
            index_list.append(vocab_dict[x])
        except:
            index_list.append(1)  # index for oov tokens
    return index_list


def padding(index_list, max_len):
    """
    pad 0 to the index_list

    INPUT:
        index_list: an integer list with length <= max_len
        max_len: integer --maximum number of entries in the list
    OUTPUT:
        an integer list with length == max_len
    """
    return [0] * (max_len - len(index_list)) + index_list


def truncate_text(token_list, max_len):
    """
    cutting tokens that exceeds the max_len for each input sentence

    INPUT:
        token_list: a list of string with variable length
        max_len: maximum number of tokens in the text
    OUTPUT:
        a list of string with length <= max_len
    """
    return token_list[:max_len]


def get_one_hot_encoding(label_lists, labels):
    """
    a function to return class labels in one hot encoding format
    INPUT:
        label_lists: list of integer labels for the samples
        labels: a list of all possible classification classes
    OUTPUT:
        y: 2-dimensional list, specifically a one-hot matrix -- shape of [N, len(labels)]
    """
    le = LabelEncoder()
    le = le.fit(labels)
    y = le.transform(label_lists)
    y = to_categorical(y, num_classes=len(labels))
    return y


def split_text(text):
    """a function to split sentence into a list of tokens based on space characters."""
    return text.split()


def is_valid_label(label, all_labels):
    """a function to return whether the input "label" exist in the all_labels, which is a list of string"""
    return label in all_labels


def label_validation(df, col_label, all_labels, num_threads=1):
    """
    a function to separate a data frame into two, one contains records with valid labels, one without

    INPUT:
        df: a pandas dataframe which the column "TEXT" and "LABEL" must exist
        col_label: name of label column in the dataframe
        labels: a list of all available labels of the classification task
        num_threads: number of CPU processors for performing the subtasks
    OUTPUT:
        df_valid: a pandas dataframe which all records have valid labels
        df_invalid: a pandas dataframe which all records have invalid labels
    """
    with mp.Pool(processes=num_threads) as pool:
        bool_valid_label = pool.map(partial(is_valid_label, all_labels=all_labels), df[col_label])
        bool_invalid_label = pool.map(operator.not_, bool_valid_label)

        df_valid = df.loc[bool_valid_label, :]
        df_invalid = df.loc[bool_invalid_label, :]

    return df_valid, df_invalid


def is_uninformative(index_list, max_len):
    """
    a boolean function to determine whether an input text is informative or not.
    Since the index 0 is used for padding and the index 1 is used for out-of-vocabulary tokens, when all token indices
    for an input text is either 0 or 1 only (equivalently small than or equal to 1), the text is
    unlikely to be reasonably predicted by the model.

    INPUT:
        index_list: a list of indices
        max_len:  maximum number of tokens in each text
    OUTPUT:
        True or False

    """
    return bool(sum([e <= 1 for e in index_list]) == max_len)


def get_wxy(des_lists, vocab_dict, max_len, labels, num_threads=1, label_lists=None, label_to_weight_map=None):
    """
    a function to get the x, y of input data frame, weight and boolean lists

    INPUT:
        des_lists: a list/numpy array/pandas Series of N texts
        vocab_dict: a dictionary with tokens(key)-integer(value) pairs
        max_len: maximum number of tokens in each text
        labels: a list of all possible classification classes
        num_threads: int --number of CPU processors for performing the subtasks
        label_lists: a list/numpy array/pandas Series of labels each input text belongs to
        label_weight_map: a map with label as the key and the corresponding weight as value

    OUTPUT:
        x_out_valid: 2-dimensional numpy array -- shape of [N-U, max_len], where U is the number of uninformative
                                               -- samples. The uninformative samples are excluded for training.
        y_out_valid: 2-dimensional numpy array, specifically a one-hot matrix -- shape of [N-U, len(labels)]
        weight_valid: an numpy array of weight for informative samples only, length of array is N-U
        bool_informative: a boolean list of length N, True means informative and vice versa
        bool_uninformative: a boolean list of length N, True means uninformative and vice versa
    """

    with mp.Pool(processes=num_threads) as pool:

        token_lists = pool.map(split_text, des_lists)
        truncated_token_lists = pool.map(partial(truncate_text, max_len=max_len), token_lists)
        truncated_index_lists = pool.map(partial(tokens_to_index, vocab_dict=vocab_dict), truncated_token_lists)
        x = pool.map(partial(padding, max_len=max_len), truncated_index_lists)
        x_out = np.array(x)

        bool_uninformative = pool.map(partial(is_uninformative, max_len=max_len), x_out)
        bool_informative = pool.map(operator.not_, bool_uninformative)

        x_out_valid = x_out[bool_informative]

        if label_lists is not None:
            y = get_one_hot_encoding(label_lists, labels)
            y_out = np.array(y)
            y_out_valid = y_out[bool_informative]

        else:
            y_out_valid = None

        if label_to_weight_map is not None:
            informative_labels = np.array(label_lists)[bool_informative]
            weight_valid = np.array(list(map(lambda x: label_to_weight_map[x], informative_labels)))
        else:
            weight_valid = None

    return x_out_valid, y_out_valid, weight_valid, bool_informative, bool_uninformative


def focal_loss(gamma=2., alpha=0.25):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        INPUT:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {0.25})

        OUTPUT:
            [tensor] -- focal loss value.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed


def load_data_file(file_path, FLAGS):
    """
    a function load the feather file and perform basic validation checking

    INPUT:
        file_path: string --path to the data file
        FLAGS: model flags
    OUTPUT:
        df: a pandas data frame which the columns "TEXT" and "LABEL" must exist.
    """

    # import the data file as pandas dataframe
    if bool(re.search(".feather", file_path.lower())):
        df = feather.read_dataframe(file_path)
    elif bool(re.search(".csv", file_path.lower())):
        df = pd.read_csv(file_path, dtype={'TEXT': str, 'LABEL': str})
    elif bool(re.search(".xlsx", file_path.lower())):
        df = pd.read_excel(file_path, dtype={'TEXT': str, 'LABEL': str})
    elif bool(re.search(".dbf", file_path.lower())):
        text = DBF(file_path, char_decode_errors="ignore")  # set "ignore_missing_memofile=True" in absence of memo file
        df = pd.DataFrame(iter(text))
    else:
        raise ValueError("Fail to open the file {}, the acceptable data file format \
        include .feather, .csv, .txt, .xlsx and .dbf only".format(file_path))

    # basic validation checking
    if "TEXT" not in df.columns:
        raise ValueError("The column `TEXT` is missing from the input data file: {}.".format(file_path))
    # only in "do_train" or "do_eval" mode the column "LABEL" is required
    if FLAGS.do_train or FLAGS.do_eval:
        if "LABEL" not in df.columns:
            raise ValueError("The column `LABEL` is missing from the input data file: {}.".format(file_path))

    return df


def get_cr_distr(prediction, col_label, col_pred, col_probit):
    """
    get the distribution of classification rate

    INPUT:
        prediction: a pandas dataframe with at least three columns, one for the groud truth label of the prediction,
                    another for the predicted label as well as one for the predicted probability of the predicted label
        col_label: column name of the ground truth label
        col_pred:  column name of the predicted label
        col_predict: column name of the predicted probability of the predicted label
    OUTPUT:
        distr_table: a pandas dataframe that shows the distribution of the classification rate and the accuracy
    """

    bins = np.arange(0, 1.1, 0.1)
    prediction["Probability"] = pd.cut(prediction[col_probit], bins)
    prediction["True/False"] = prediction[col_label] == prediction[col_pred]
    prediction["Proportion (%)"] = 1

    distr_table = prediction.loc[:, ["Probability", 'True/False', 'Proportion (%)']].groupby(["Probability", 'True/False'])
    distr_table = (distr_table.count()/prediction.shape[0])*100
    distr_table = distr_table.unstack()
    distr_table["Proportion (%)", "Total"] = distr_table["Proportion (%)"][True] + distr_table["Proportion (%)"][False]
    distr_table["Classification rate (%)"] = 100 * (distr_table["Proportion (%)"][True] / \
                                             (distr_table["Proportion (%)"][True] + distr_table["Proportion (%)"][False]))
    return distr_table



