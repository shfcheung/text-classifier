#!/usr/bin/env python
# coding=utf-8

"""Keras GCNN model"""


import abc
import numpy as np
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.layers import Embedding
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Dropout, Multiply
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import focal_loss


class AbstractKerasModel(object):
    """
    Abtract class for KERAS model
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def build(self, **kwargs):
        pass

    @abc.abstractmethod
    def show_summary(self, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, **kwargs):
        pass


class GCNN(AbstractKerasModel):
    """a class for building Gated Convolutional Neural Network"""

    def __init__(self, num_classes=None, max_len=None, emb_matrix=None, num_filter_1=None,
                 kernel_size_1=None, num_stride_1=None, num_filter_2=None, kernel_size_2=None,
                 num_stride_2=None, hidden_dims_1=None, hidden_dims_2=None, hidden_dims_3=None,
                 hidden_dims_4=None, dropout_rate=None, pretrained_model=None):

        self.num_classes = num_classes
        self.max_len = max_len
        self.emb_matrix = emb_matrix
        self.num_filter_1 = num_filter_1
        self.kernel_size_1 = kernel_size_1
        self.num_stride_1 = num_stride_1
        self.num_filter_2 = num_filter_2
        self.kernel_size_2 = kernel_size_2
        self.num_stride_2 = num_stride_2
        self.hidden_dims_1 = hidden_dims_1
        self.hidden_dims_2 = hidden_dims_2
        self.hidden_dims_3 = hidden_dims_3
        self.hidden_dims_4 = hidden_dims_4
        self.dropout_rate = dropout_rate
        self.model = pretrained_model

        if self.emb_matrix is not None:
            self.num_vocab = len(self.emb_matrix)
            self.emb_dim = len(self.emb_matrix[0])

    def build(self):
        """construct model architecture"""
        input_context = Input(shape=(self.max_len,), dtype='int32')
        embeddings = Embedding(self.num_vocab + 2,
                               self.emb_dim,
                               embeddings_initializer=Constant(self.emb_matrix),
                               input_length=self.max_len,
                               trainable=True)(input_context)

        branch1 = Dropout(self.dropout_rate)(embeddings)
        branch1 = Conv1D(self.num_filter_1, self.kernel_size_1,
                         padding='same', activation='relu',
                         strides=self.num_stride_1)(branch1)

        branch2 = Dropout(self.dropout_rate)(embeddings)
        branch2 = Conv1D(self.num_filter_2, self.kernel_size_2,
                         padding='same', activation='tanh',
                         strides=self.num_stride_2)(branch2)

        merged = Multiply()([branch1, branch2])  # elementwise multiplication
        merged = GlobalMaxPooling1D()(merged)

        # feed-forward neural network
        preds = Dense(self.hidden_dims_1, activation='selu')(merged)
        preds = Dropout(self.dropout_rate)(preds)

        preds = Dense(self.hidden_dims_2, activation='selu')(preds)
        preds = Dropout(self.dropout_rate)(preds)

        preds = Dense(self.hidden_dims_3, activation='selu')(preds)
        preds = Dense(self.hidden_dims_4, activation='relu')(preds)
        preds = Dense(self.num_classes, activation='softmax')(preds)

        self.model = Model([input_context], preds)

    def show_summary(self):
        """show model summary"""
        self.model.summary()

    def train(self, x_train, y_train, x_valid, y_valid,
              learning_rate, es_patience, es_min_delta,
              reduce_lr_patience, reduce_lr_factor, batch_size,
              epochs, loss_function, train_sample_weight=None, valid_sample_weight=None):
        """model training"""

        # loss function setting
        if loss_function == 'categorical_crossentropy':
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(lr=learning_rate),
                               metrics=['accuracy'])
        elif loss_function == 'focal_loss':
            self.model.compile(loss=focal_loss(alpha=2),
                               optimizer=Adam(lr=learning_rate),
                               metrics=['accuracy'])

        # set early stopping criteria and reduce learning rate
        es = EarlyStopping(monitor='val_loss', patience=es_patience, min_delta=es_min_delta, mode='min')
        reduce_lr = ReduceLROnPlateau(patience=reduce_lr_patience, verbose=1, factor=reduce_lr_factor)

        # format validation data into a tuple
        if valid_sample_weight is not None:
            validation_data_tuple = (x_valid, y_valid, valid_sample_weight)
        else:
            validation_data_tuple = (x_valid, y_valid)

        # train GCNN model
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       shuffle=True,
                       epochs=epochs,
                       sample_weight=train_sample_weight,
                       validation_data=validation_data_tuple,
                       callbacks=[es, reduce_lr])

    def evaluate(self, x_input, y_input):
        """evaluate the prediction accuracy and loss"""
        loss, accuracy = self.model.evaluate(x_input, y_input, verbose=1)
        print("Prediction accuracy: {0:.3f}\nLoss: {1:.5f}".format(accuracy, loss))

    def predict(self, x_input, index_to_label_map):
        """
        predict a batch of x and y

        INPUT:
            x_input: a 2-dimensional integer array of shape [N, max_len]
            index_to_label_map: a dictionary with key as integer index, value as the label
        OUTPUT:
            pred_probits: a 2-dimensional array of shape [N, len(index_tot_label_map)]
            pred_labels: list of predicted labels
        """
        pred_probits = self.model.predict(x_input, verbose=1)  # verbose=1: show progress
        pred_indices = np.argmax(pred_probits, axis=1)
        pred_labels = [index_to_label_map[pred_index] for pred_index in pred_indices]

        return pred_probits, pred_labels

    def save(self, save_dir):
        """save trained model"""
        self.model.save(save_dir)
