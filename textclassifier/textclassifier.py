#!/usr/bin/env python
# coding=utf-8

"""Main program of a text classifier"""

# prevent printing "Using TensorFlow backend", causing minor formatting issues though
import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras

sys.stderr = stderr

import time
import json
import jieba
import feather
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
from keras.models import load_model
from kerasmodel import GCNN
from word2vecmodel import train_word2vec
from validatearguments import validate_flags_or_throw
from sklearn.metrics import classification_report
from utils import load_data_file, text_cleaning, get_wxy, focal_loss, label_validation, get_cr_distr

flags = tf.flags

FLAGS = flags.FLAGS

# parameters
flags.DEFINE_string("model_name", None,
                    "The name of the Neural Network Model")

flags.DEFINE_string("train_file", None,
                    "The path for .feather file that contains the training data.")

flags.DEFINE_string("valid_file", None,
                    "The path for .feather file that contains the validation data.")

flags.DEFINE_string("predict_file", None,
                    "The path for .feather file that contains the testing data.")

flags.DEFINE_string("train_word2vec_file", None,
                    "The path for .feather file that contains the training corpus for word2vec model.")

flags.DEFINE_string("eval_file", None,
                    "The path for .feather file that contains the data (contain "
                    "both description and code) for evaluation.")

flags.DEFINE_string("output_dir", None,
                    "The directory for storing the trained Word2vec/Neural Network model.")

flags.DEFINE_string("class_file", None,
                    "The path for .csv file that contains the classes of the classification task.")

flags.DEFINE_string("emb_matrix_file", None,
                    "The path for .feather file that contains word embedding matrix.")

flags.DEFINE_string("jieba_user_dict", None,
                    "The path for additional vocabulary file (.txt) for Chinese segmentation")

flags.DEFINE_string("bert_vocab_file", None,
                    "The BERT vocabulary file (.txt) for tokenization, must exist if `word_segmentation=word_piece`.")

flags.DEFINE_string("keras_model_file", None,
                    "The path for pre-trained keras neural network model.")

flags.DEFINE_string("keras_model_flags", None,
                    "The path for model flags of trained keras neural network model.")

flags.DEFINE_string("emb_layer_vocab", None,
                    "The path for vocabularies used in the embedding layer of keras neural network model.")

flags.DEFINE_integer("num_threads", 1,
                     "The number of threads for multiprocessing.")

flags.DEFINE_bool("do_word2vec", False, "Whether to training Word2vec vectors.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run evaluation on the testing dataset.")

flags.DEFINE_bool("do_predict", False, "Whether to run prediction on the input dataset.")

flags.DEFINE_bool("do_single_predict", False, "Whether to run prediction on single entry on command line.")

flags.DEFINE_bool("train_with_weights", True, "Whether to apply weights to loss function during training.")

flags.DEFINE_string("word_tokenization", None, "Which word tokenization method used in text cleaning, "
                                               "either `word_piece` or `space`.")

# flags for word2vec
flags.DEFINE_bool("word2vec_sg", None, "True for training skip-gram model and False for CBOW.")

flags.DEFINE_integer("word2vec_size", None, "Dimension of word2vec vectors.")

flags.DEFINE_integer("word2vec_window", None, "Maximum distance between the current and predicted word within "
                                              "a sentence.")

flags.DEFINE_integer("word2vec_min_count", None, "Ignores all words with total frequency lower than this in word2vec "
                                                 "training.")

flags.DEFINE_integer("word2vec_iterations", None, "Number of iterations (epochs) over the corpus in word2vec training.")

# flags for Neural Network Model
flags.DEFINE_integer("max_len", 256, "Maximum length of token sequence.")

flags.DEFINE_integer("batch_size", 128, "Training batch size.")

flags.DEFINE_integer("num_filter_1", 256, "The number of convolutional filters in branch 1"
                                          "of GCNN.")

flags.DEFINE_integer("kernel_size_1", 5, "The length of the 1D convolution window in branch 1.")

flags.DEFINE_integer("num_stride_1", 1, "The stride length of the convolution in branch 1.")

flags.DEFINE_integer("num_filter_2", 256, "The number of convolutional filters in branch 2"
                                          "of GCNN.")

flags.DEFINE_integer("kernel_size_2", 3, "The length of the 1D convolution window in branch 2.")

flags.DEFINE_integer("num_stride_2", 1, "The stride length of the convolution in branch 2.")

flags.DEFINE_integer("hidden_dims_1", 512, "The number of neurons in the 1st "
                                           "feed-forward neural network layer.")

flags.DEFINE_integer("hidden_dims_2", 256, "The number of neurons in the 2nd "
                                           "feed-forward neural network layer.")

flags.DEFINE_integer("hidden_dims_3", 128, "The number of neurons in the 3rd "
                                           "feed-forward neural network layer.")

flags.DEFINE_integer("hidden_dims_4", 64, "The number of neurons in the 4th "
                                          "feed-forward neural network layer.")

flags.DEFINE_integer("epochs", 100, "The number of training epochs to perform.")

flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")

flags.DEFINE_integer("es_patience", 10, "The number of epochs with no improvement after "
                                        "which training will be stopped.")

flags.DEFINE_float("es_min_delta", 0.0001, "The threshold for measuring the new optimum, "
                                           "to only focus on significant changes.")

flags.DEFINE_integer("reduce_lr_patience", 2, "If no improvement is seen for a `patience` "
                                              "number of epochs, the learning rate is reduced.")

flags.DEFINE_float("reduce_lr_factor", 0.5, "Factor by which the learning rate will be reduced. "
                                            "new_lr = lr * factor")

flags.DEFINE_float("dropout_rate", 0.5, "The fraction of input units setting at 0 at each update "
                                        "during training time, which helps prevent overfitting.")

flags.DEFINE_string("loss_function", None,
                    "Loss function for training Neural Network, must only be either `categorical_crossentropy` or"
                    "`focal_loss`.")


def main(_):
    """main program"""

    main_start = time.time()
    # set random seed to get reproducible results
    seed(3000)
    set_random_seed(3000)

    # disable TensorFlow default messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    # validate the flags
    validate_flags_or_throw(FLAGS)

    # user defined dictionary for better Chinese token segmentation
    if FLAGS.jieba_user_dict is not None:
        jieba.load_userdict(FLAGS.jieba_user_dict)

    # create directory if it doesn't exist
    if not FLAGS.do_single_predict:
        if not os.path.isdir(FLAGS.output_dir):
            os.mkdir(FLAGS.output_dir)

    if FLAGS.do_word2vec:

        word2vec_flags = {
            "sg": FLAGS.word2vec_sg,
            "size": FLAGS.word2vec_size,
            "window": FLAGS.word2vec_window,
            "min_count": FLAGS.word2vec_min_count,
            "number_of_threads": FLAGS.num_threads,
            "iterations": FLAGS.word2vec_iterations, }

        with open(os.path.join(FLAGS.output_dir, "WORD2VEC_FLAGS.json"), 'w') as json_file:
            json.dump(word2vec_flags, json_file, indent=4)

        train_corpus = load_data_file(FLAGS.train_word2vec_file, FLAGS)
        print("Cleaning training corpus...")
        train_corpus["TEXT"] = text_cleaning(text_array=train_corpus["TEXT"],
                                             bert_vocab_file=FLAGS.bert_vocab_file,
                                             do_lower_case=False,
                                             word_tokenization=FLAGS.word_tokenization,
                                             num_threads=FLAGS.num_threads)

        word2vec_model, df_vectors = train_word2vec(text_array=train_corpus['TEXT'],
                                                    sg=FLAGS.word2vec_sg,
                                                    size=FLAGS.word2vec_size,
                                                    window=FLAGS.word2vec_window,
                                                    min_count=FLAGS.word2vec_min_count,
                                                    num_threads=FLAGS.num_threads,
                                                    iterations=FLAGS.word2vec_iterations)

        print('Saving model and embedding vectors...')
        word2vec_model.save(os.path.join(FLAGS.output_dir, 'Word2vec.model'))
        feather.write_dataframe(df_vectors, os.path.join(FLAGS.output_dir, 'EmbeddingVectors.feather'))

    else:

        emb_matrix = None
        pretrained_model = None

        # list of labels and several associate dictionaries
        class_table = pd.read_csv(FLAGS.class_file, dtype={'LABEL': str, 'DESCRIPTION': str})
        labels = list(class_table["LABEL"])
        label_to_desc_map = dict(zip(class_table["LABEL"], class_table["DESCRIPTION"]))
        index_to_label_map = dict(enumerate(labels))
        if FLAGS.train_with_weights:
            if "WEIGHT" not in class_table.columns:
                raise ValueError("The column `WEIGHT` is missing in the {}.".format(FLAGS.class_file))
            else:
                label_to_weight_map = dict(zip(class_table["LABEL"], class_table["WEIGHT"]))

        if FLAGS.do_train:
            # embedding matrix
            emb_matrix = feather.read_dataframe(FLAGS.emb_matrix_file)
            vocab = list(emb_matrix.columns)
            vocab_dict = dict(zip(vocab, range(2, len(vocab) + 2)))

            emb_matrix = emb_matrix.T.values  # transpose the matrix then obtain the values
            emb_matrix = np.insert(emb_matrix,
                                   0,
                                   np.random.normal(size=[1, emb_matrix.shape[1]]),
                                   axis=0)  # embedding vectors for oov symbol
            emb_matrix = np.insert(emb_matrix,
                                   0,
                                   np.random.normal(size=[1, emb_matrix.shape[1]]),
                                   axis=0)  # embedding vectors for padding symbol

        else:
            # available in "do_eval", "do_predict", "do_single_predict" mode
            with open(FLAGS.emb_layer_vocab, "r") as f:
                vocab = [line.rstrip('\n') for line in f]
                vocab_dict = dict(zip(vocab, range(2, len(vocab) + 2)))

            with open(FLAGS.keras_model_flags, "r") as f:
                keras_model_flags = json.load(f)

            # bert_vocab_file is needed if word piece tokenization is used in training the model
            if keras_model_flags["word_tokenization"] == "word_piece" and FLAGS.bert_vocab_file is None:
                raise ValueError("The model uses Word Piece tokenization but `bert_vocab_file` is not supplied.")

            print("\nLoading trained model...")
            if keras_model_flags["loss_function"] == 'categorical_crossentropy':
                pretrained_model = load_model(FLAGS.keras_model_file)
            elif keras_model_flags["loss_function"] == 'focal_loss':
                pretrained_model = load_model(FLAGS.keras_model_file,
                                              custom_objects={'focal_loss(alpha=2)': focal_loss(alpha=2),
                                                              'focal_loss_fixed': focal_loss()})

        # instantiate a GCNN model object
        model = GCNN(num_classes=len(labels),
                     max_len=FLAGS.max_len,
                     emb_matrix=emb_matrix,
                     num_filter_1=FLAGS.num_filter_1,
                     kernel_size_1=FLAGS.kernel_size_1,
                     num_stride_1=FLAGS.num_stride_1,
                     num_filter_2=FLAGS.num_filter_2,
                     kernel_size_2=FLAGS.kernel_size_2,
                     num_stride_2=FLAGS.num_stride_2,
                     hidden_dims_1=FLAGS.hidden_dims_1,
                     hidden_dims_2=FLAGS.hidden_dims_2,
                     hidden_dims_3=FLAGS.hidden_dims_3,
                     hidden_dims_4=FLAGS.hidden_dims_4,
                     dropout_rate=FLAGS.dropout_rate,
                     pretrained_model=pretrained_model)

    if FLAGS.do_train:

        # save flags and hyperparameters to a json file
        print("Saving model flags and hyperparameters to {}".format(os.path.join(FLAGS.output_dir,
                                                                                 FLAGS.model_name + "_FLAGS.json")))

        neural_network_flags = {"model_name": FLAGS.model_name,
                                "num_classes": len(labels),
                                "max_len": FLAGS.max_len,
                                "batch_size": FLAGS.batch_size,
                                "num_filter_1": FLAGS.num_filter_1,
                                "kernel_size_1": FLAGS.kernel_size_1,
                                "num_stride_1": FLAGS.num_stride_1,
                                "num_filter_2": FLAGS.num_filter_2,
                                "kernel_size_2": FLAGS.kernel_size_2,
                                "num_stride_2": FLAGS.num_stride_2,
                                "hidden_dims_1": FLAGS.hidden_dims_1,
                                "hidden_dims_2": FLAGS.hidden_dims_2,
                                "hidden_dims_3": FLAGS.hidden_dims_3,
                                "hidden_dims_4": FLAGS.hidden_dims_4,
                                "epochs": FLAGS.epochs,
                                "learning_rate": FLAGS.learning_rate,
                                "es_patience": FLAGS.es_patience,
                                "es_min_delta": FLAGS.es_min_delta,
                                "reduce_lr_patience": FLAGS.reduce_lr_patience,
                                "reduce_lr_factor": FLAGS.reduce_lr_factor,
                                "dropout_rate": FLAGS.dropout_rate,
                                "loss_function": FLAGS.loss_function,
                                "train_with_weights": FLAGS.train_with_weights,
                                "word_tokenization": FLAGS.word_tokenization}

        with open(os.path.join(FLAGS.output_dir, FLAGS.model_name + "_flags.json"), 'w') as json_file:
            json.dump(neural_network_flags, json_file, indent=4)

        with open(os.path.join(FLAGS.output_dir, FLAGS.model_name + "_vocab.txt"), "w") as f:
            f.writelines('\n'.join(vocab))

        # import data and text tokenization
        print("Importing data...")
        train = load_data_file(os.path.join(FLAGS.train_file), FLAGS)
        valid = load_data_file(os.path.join(FLAGS.valid_file), FLAGS)

        # separate records with valid and invalid labels into 2 dataframes
        train, invalid_label_train = label_validation(train, "LABEL", labels, num_threads=FLAGS.num_threads)
        valid, invalid_label_valid = label_validation(valid, "LABEL", labels, num_threads=FLAGS.num_threads)

        # save the records with invalid labels
        if invalid_label_train.shape[0] > 0:
            print("{} train samples have invalid labels.".format(invalid_label_train.shape[0]))
            feather.write_dataframe(invalid_label_train, os.path.join(FLAGS.output_dir, "invalid_label_train.feather"))
        if invalid_label_valid.shape[0] > 0:
            print("{} validation samples have invalid labels.".format(invalid_label_train.shape[0]))
            feather.write_dataframe(invalid_label_valid, os.path.join(FLAGS.output_dir, "invalid_label_valid.feather"))

        print("Cleaning TRAINING data...")
        train["TEXT"] = text_cleaning(text_array=train["TEXT"],
                                      bert_vocab_file=FLAGS.bert_vocab_file,
                                      do_lower_case=False,
                                      word_tokenization=FLAGS.word_tokenization,
                                      num_threads=FLAGS.num_threads)

        print("Cleaning VALIDATION data...")
        valid["TEXT"] = text_cleaning(text_array=valid["TEXT"],
                                      bert_vocab_file=FLAGS.bert_vocab_file,
                                      do_lower_case=False,
                                      word_tokenization=FLAGS.word_tokenization,
                                      num_threads=FLAGS.num_threads)

        # Convert tokenized text to numerical matrices for Neural Network training
        print("Transforming data...")
        x_train, y_train, weight_train, informative_train, \
        uninformative_train = get_wxy(des_lists=train["TEXT"],
                                      label_lists=train["LABEL"],
                                      vocab_dict=vocab_dict,
                                      max_len=FLAGS.max_len,
                                      num_threads=FLAGS.num_threads,
                                      labels=labels,
                                      label_to_weight_map=label_to_weight_map)

        x_valid, y_valid, weight_valid, informative_valid, \
        uninformative_valid = get_wxy(des_lists=valid["TEXT"],
                                      label_lists=valid["LABEL"],
                                      vocab_dict=vocab_dict,
                                      max_len=FLAGS.max_len,
                                      num_threads=FLAGS.num_threads,
                                      labels=labels,
                                      label_to_weight_map=label_to_weight_map)

        # save records with uninformative description
        if sum(uninformative_train) > 0:
            print(
                "{} train samples are considered uninformative and are not included in model training".format(
                    sum(uninformative_train)))
            feather.write_dataframe(train.loc[uninformative_train, :],
                                    os.path.join(FLAGS.output_dir, "uninformative_train.feather"))

        if sum(uninformative_valid) > 0:
            print(
                "{} validation samples are considered uninformative and are not included in model training".format(
                    sum(uninformative_valid)))
            feather.write_dataframe(valid.loc[uninformative_valid, :],
                                    os.path.join(FLAGS.output_dir, "uninformative_valid.feather"))

        print("Building Neural Network...")
        # Build the GCNN model
        model.build()
        print("Structure of the Neural Network:")
        model.show_summary()
        print("Training begins...")
        model.train(x_train, y_train, x_valid, y_valid, FLAGS.learning_rate,
                    FLAGS.es_patience, FLAGS.es_min_delta, FLAGS.reduce_lr_patience,
                    FLAGS.reduce_lr_factor, FLAGS.batch_size, FLAGS.epochs, FLAGS.loss_function,
                    weight_train, weight_valid)

        # Evaluating model performance on training and validation data
        print("Evaluating model performance...")
        print("TRAINING DATA:")
        model.evaluate(x_train, y_train)
        print("VALIDATION DATA:")
        model.evaluate(x_valid, y_valid)

        # save GCNN model
        print("Training completed! Saving model to {}".format(os.path.join(FLAGS.output_dir,
                                                                           FLAGS.model_name + ".h5")))
        model.save(os.path.join(FLAGS.output_dir, FLAGS.model_name + ".h5"))  # save model

    if FLAGS.do_eval:
        print("Begin evaluating model performance...")

        # import data and tokenize text
        print("Importing data...")
        data_eval = load_data_file(FLAGS.eval_file, FLAGS)

        # separate records with valid and invalid labels into 2 dataframes
        data_eval, invalid_label_data_eval = label_validation(data_eval, "LABEL", labels, num_threads=FLAGS.num_threads)

        # save the records with invalid labels
        if invalid_label_data_eval.shape[0] > 0:
            print("{} evaluation samples have invalid labels.".format(invalid_label_data_eval.shape[0]))
            feather.write_dataframe(invalid_label_data_eval, os.path.join(FLAGS.output_dir, "invalid_label_eval.feather"))

        print("Cleaning data...")
        cleaned_text_array = text_cleaning(text_array=data_eval["TEXT"],
                                           bert_vocab_file=FLAGS.bert_vocab_file,
                                           do_lower_case=False,
                                           word_tokenization=keras_model_flags["word_tokenization"],
                                           num_threads=FLAGS.num_threads)

        # import data and text tokenization
        print("Transforming data...")
        x_eval, y_eval, _, informative_eval, uninformative_eval = get_wxy(des_lists=cleaned_text_array,
                                                                          label_lists=data_eval["LABEL"],
                                                                          vocab_dict=vocab_dict,
                                                                          max_len=keras_model_flags["max_len"],
                                                                          labels=labels,
                                                                          num_threads=FLAGS.num_threads)
        
        # save records with uninformative description
        if sum(uninformative_eval) > 0:
            print("{} evaluation samples are considered uninformative and are not included in model evaluation.".format(
                sum(uninformative_eval)))
            feather.write_dataframe(data_eval.loc[uninformative_eval, :],
                                    os.path.join(FLAGS.output_dir, "uninformative_eval.feather"))

        print("Prediction in progress...")
        pred_probits_eval, pred_labels_eval = model.predict(x_eval, index_to_label_map)
        print("Prediction completed.")

        data_eval = data_eval.loc[informative_eval, :]
        data_eval["PREDICTION"] = pred_labels_eval
        data_eval["PRED_PROBIT"] = np.max(pred_probits_eval, axis=1)
        data_eval = data_eval.join(other=pd.DataFrame(pred_probits_eval, columns=labels))

        feather.write_dataframe(data_eval, os.path.join(FLAGS.output_dir, "evaluation.feather"))

        # compute precision, recall and f-1 score
        data_report = classification_report(data_eval["LABEL"], pred_labels_eval, output_dict=True)
        data_report = pd.DataFrame(data_report).transpose()
        data_report.to_csv(os.path.join(FLAGS.output_dir, "eval_metrics.csv"))

        # compute distribution and classification rates of the highest prediction probability
        distr_cr = get_cr_distr(data_eval,
                                col_label="LABEL",
                                col_pred="PREDICTION",
                                col_probit="PRED_PROBIT")
        distr_cr.to_csv(os.path.join(FLAGS.output_dir, "distribution_stat.csv"))

        print("\nPerformance Metrics:")
        print(data_report)
        print("\nDistribution of classification rate: ")
        print(distr_cr)

    if FLAGS.do_predict:
        print("Prediction on input dataset...")

        # import data and tokenize text
        print("Importing data...")
        data_prediction = load_data_file(FLAGS.predict_file, FLAGS)

        print("Cleaning input data...")
        cleaned_text_array = text_cleaning(text_array=data_prediction["TEXT"],
                                           bert_vocab_file=FLAGS.bert_vocab_file,
                                           do_lower_case=False,
                                           word_tokenization=keras_model_flags["word_tokenization"],
                                           num_threads=FLAGS.num_threads)

        # Convert tokenized text to numerical matrices for Neural Network training
        print("Transforming data...")
        x_test, _, _, informative_test, uninformative_predict = get_wxy(des_lists=cleaned_text_array,
                                                                        vocab_dict=vocab_dict,
                                                                        max_len=keras_model_flags["max_len"],
                                                                        labels=labels,
                                                                        num_threads=FLAGS.num_threads)

        if sum(uninformative_predict) > 0:
            print("{} prediction samples are considered uninformative and are not included in prediction.".format(
                sum(uninformative_predict)))
            
            feather.write_dataframe(data_prediction.loc[uninformative_predict, :],
                                    os.path.join(FLAGS.output_dir, "uninformative_predict.feather"))

        # prediction
        print("Prediction in progress...")
        pred_probits, pred_labels = model.predict(x_test, index_to_label_map)
        print("Prediction completed.")

        data_prediction = data_prediction.loc[informative_test, :]
        data_prediction["PREDICTION"] = pred_labels
        data_prediction["PRED_PROBIT"] = np.max(pred_probits, axis=1)
        data_prediction = data_prediction.join(other=pd.DataFrame(pred_probits, columns=labels))

        feather.write_dataframe(data_prediction, os.path.join(FLAGS.output_dir, "prediction.feather"))

    if FLAGS.do_single_predict:

        print("Time taken for loading the model: {:.2f}".format(time.time() - main_start))
        do_next = True  # boolean variable of whether to do next prediction.

        while do_next:  # while loop to enable users to do multiple queries
            input_text = input("\nInput text here: ")
            start = time.time()
            input_text = text_cleaning(text_array=[input_text],
                                            bert_vocab_file=FLAGS.bert_vocab_file,
                                            do_lower_case=False,
                                            word_tokenization=keras_model_flags["word_tokenization"],
                                            num_threads=FLAGS.num_threads)
            x_data, _, _, _, uninformative_data = get_wxy(des_lists=input_text, vocab_dict=vocab_dict,
                                                          max_len=keras_model_flags["max_len"], labels=labels)

            if uninformative_data[0]:
                print("\nThe text is either uninformative or never learnt by model before...")
            else:
                print("Prediction in progress...")
                pred_probits, pred_labels = model.predict(x_data, index_to_label_map)

                # sort the pairs of label and prediction probability in descending order
                output = dict(zip(labels, pred_probits[0]))
                sorted_output = sorted(output.items(), key=lambda kv: kv[1], reverse=True)

                print("\nPREDICTIONS:")
                # print the top 3 predictions and probabilities
                index = 1
                for (code, probit) in sorted_output[0:3]:
                    code_name = label_to_desc_map[code]
                    print(str(index) + ". " + code_name)
                    print("   {:.2%}".format(probit))
                    index += 1

            print("Time taken for prediction: {:.2f}".format(time.time() - start))
            del input_text

            # ask user whether he/she would like to continue prediction
            reply = input("\nDo you want to start a new prediction (Y/N)? ")
            while reply != "Y" and reply != "N":
                reply = input("Please input either Y or N to proceed: ")

            # end the program if the user answer "N".
            if reply == "N":
                do_next = False
                print("****************End of program****************")


if __name__ == "__main__":
    tf.app.run()
