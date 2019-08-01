#!/usr/bin/env python
# coding=utf-8

"""Input argument validation"""

import multiprocessing as mp


def raise_error_if_flag_na(mode):
    def raise_error_if_flag_na_specific_mode(flag, flag_name):
        """raise ValueError when the argument is None"""
        if not flag:
            raise ValueError("If `{}` is True, then `{}` must be specified.".format(mode, flag_name))
    return raise_error_if_flag_na_specific_mode


def validate_flags_or_throw(FLAGS):
    """Validate the input FLAGS or throw an exception."""

    if (bool(FLAGS.do_word2vec) + bool(FLAGS.do_train) + bool(FLAGS.do_eval) +
        bool(FLAGS.do_predict) + bool(FLAGS.do_single_predict)) != 1:
        raise ValueError(
            "One of `do_word2vec` or `do_train` or `do_eval` or `do_predict` or `do_single_predict` must be True."
        )

    if FLAGS.do_word2vec:
        raise_error_if_flag_na_do_word2vec = raise_error_if_flag_na("do_word2vec")

        raise_error_if_flag_na_do_word2vec(FLAGS.output_dir, "output_dir")
        raise_error_if_flag_na_do_word2vec(FLAGS.train_word2vec_file, "train_word2vec_file")
    if FLAGS.do_train:
        raise_error_if_flag_na_do_train = raise_error_if_flag_na("do_train")

        raise_error_if_flag_na_do_train(FLAGS.output_dir, "output_dir")
        raise_error_if_flag_na_do_train(FLAGS.train_file, "train_file")
        raise_error_if_flag_na_do_train(FLAGS.valid_file, "valid_file")
        raise_error_if_flag_na_do_train(FLAGS.class_file, "class_file")
        raise_error_if_flag_na_do_train(FLAGS.emb_matrix_file, "emb_matrix_file")
        raise_error_if_flag_na_do_train(FLAGS.loss_function, "loss_function")

        if FLAGS.loss_function not in ["categorical_crossentropy", "focal_loss"]:
            raise ValueError("Input to `loss_function` must only be `categorical_crossentropy` or `focal_loss`.")
        if FLAGS.word_tokenization not in ["space", "word_piece"]:
            raise ValueError("Input to `word_tokenization` must only be `space` or `word_piece`.")
        if not FLAGS.bert_vocab_file and FLAGS.word_tokenization == "word_piece":
            raise ValueError(
                "`bert_vocab_file` must be supplied if the `word_tokenization` is indicated as `word_piece`."
            )

    if FLAGS.do_eval:
        raise_error_if_flag_na_do_eval = raise_error_if_flag_na("do_eval")

        raise_error_if_flag_na_do_eval(FLAGS.output_dir, "output_dir")
        raise_error_if_flag_na_do_eval(FLAGS.eval_file, "eval_file")
        raise_error_if_flag_na_do_eval(FLAGS.class_file, "class_file")
        raise_error_if_flag_na_do_eval(FLAGS.keras_model_file, "keras_model_file")
        raise_error_if_flag_na_do_eval(FLAGS.keras_model_flags, "keras_model_flags")
        raise_error_if_flag_na_do_eval(FLAGS.emb_layer_vocab, "emb_layer_vocab")
    if FLAGS.do_predict:
        raise_error_if_flag_na_do_predict = raise_error_if_flag_na("do_predict")

        raise_error_if_flag_na_do_predict(FLAGS.output_dir, "output_dir")
        raise_error_if_flag_na_do_predict(FLAGS.predict_file, "predict_file")
        raise_error_if_flag_na_do_predict(FLAGS.class_file, "class_file")
        raise_error_if_flag_na_do_predict(FLAGS.keras_model_file, "keras_model_file")
        raise_error_if_flag_na_do_predict(FLAGS.keras_model_flags, "keras_model_flags")
        raise_error_if_flag_na_do_predict(FLAGS.emb_layer_vocab, "emb_layer_vocab")
    if FLAGS.do_single_predict:
        raise_error_if_flag_na_do_single_predict = raise_error_if_flag_na("do_single_predict")

        raise_error_if_flag_na_do_single_predict(FLAGS.class_file, "class_file")
        raise_error_if_flag_na_do_single_predict(FLAGS.keras_model_file, "keras_model_file")
        raise_error_if_flag_na_do_single_predict(FLAGS.keras_model_flags, "keras_model_flags")
        raise_error_if_flag_na_do_single_predict(FLAGS.emb_layer_vocab, "emb_layer_vocab")
    if FLAGS.num_threads > mp.cpu_count():
        raise ValueError("Number of cpu cores specified exceeds upper limit")
