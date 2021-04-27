"""
library of character-based RNN
@Author Chanwoo Gwon, Yonsei Univ. Researcher, since 2020.05. ~
@Date 2020.10.22
"""

import tensorflow as tf
import os
import codecs
import numpy as np
import random

from model.core import ModelCore, LOSS, Net


class CharRNN(ModelCore):
    """
    RNN Library Class

    1. Train and Predict using RNN based on Character
    2. Could be used like the encoder
    3. Can many-to-many or many-to-one RNN
    """
    def __init__(self, data_path, emb=100, loss=LOSS.SPARSE_CATEGORICAL_CROSSENTROPY):

        self._time_step = 0
        self._text_set = None
        self._emb = emb
        self._char_set = []
        self._char2idx = None
        self._idx2char = None
        self._text_as_int = None
        self._last_dim = 0

        super().__init__(data_path, loss=loss)

    def read_data(self):
        char_path = os.path.join(self._data_path, 'data.txt')

        self._text_set = open(char_path, 'rb').read().decode(encoding='utf-8')

        self._make_word_world()

    def _make_word_world(self):
        self._set_char()

    def _set_char(self, system_word_list=None):
        self._char_set = sorted(set(self._text_set))

        if system_word_list is not None:
            self._char_set = self._char_set + system_word_list

        self._char2idx = {u: i for i, u in enumerate(self._char_set)}
        self._idx2char = np.array(self._char_set)

    def build_model(self):
        super().build_model()

    def to_vector(self, sequence, add_margin=False):
        if add_margin:
            if len(sequence) < self._time_step:
                sequence = "{0}{1}".format(sequence, ''.join([' '] * (self._time_step - len(sequence))))
        return [self._char2idx[c] for c in sequence]

    def to_text(self, sequence):
        return repr("".join(self._idx2char[sequence]))


class TypoClassifier(CharRNN):
    def __init__(self, data_path):
        self._label_dic = {}
        super().__init__(data_path, loss=LOSS.CATEGORICAL_CROSSENTROPY)

    def _make_word_world(self):
        data_temp = []
        char_world = ''
        max_length = -1
        for line in self._text_set.split('\n'):
            split_data = line.strip().split('\t')

            typo = split_data[0]
            answer = split_data[1]

            data_temp.append([typo, answer])
            char_world = "{0}{1}".format(char_world, split_data[0])
            max_length = max(max_length, len(typo))

            if answer not in self._label_dic:
                self._label_dic[answer] = len(self._label_dic)

        self._text_set = char_world
        self._time_step = max_length
        self._last_dim = len(self._label_dic)
        # make char set using typo only
        self._set_char([' '])

        data_all = []
        for item in data_temp:
            item_one = item[0]
            zero = [0] * len(self._label_dic)
            zero[self._label_dic[item[1]]] = 1
            if len(item_one) < max_length:
                item_one = "{0}{1}".format(item_one, ''.join([' '] * (max_length - len(item_one))))
            data_all.append([self.to_vector(item_one), zero])

        random.shuffle(data_all)
        sp = int(len(data_all) * 0.8)

        self._train_data.set(
            [tf.convert_to_tensor([item[0] for item in data_all[:sp]], dtype=tf.int64)],
            [tf.convert_to_tensor([item[1] for item in data_all[:sp]], dtype=tf.int64)]
        )
        self._test_data.set(
            [tf.convert_to_tensor([item[0] for item in data_all[sp:]], dtype=tf.int64)],
            [tf.convert_to_tensor([item[1] for item in data_all[sp:]], dtype=tf.int64)]
        )

        print("train data:", len(self._train_data))
        print("test data:", len(self._test_data))

    def get_label(self, index):
        for key, value in self._label_dic.items():
            if value == index:
                return key
        return None

    def build_model(self):
        vocab_size = len(self._char_set)
        input = tf.keras.layers.Input([self._time_step])

        embb = tf.keras.layers.Embedding(vocab_size, self._emb)(input)

        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,
                                                                  return_sequences=True,
                                                                  recurrent_initializer='glorot_uniform'))(embb)

        output = tf.keras.layers.Flatten()(lstm)

        dense = tf.keras.layers.Dense(self._last_dim, activation=tf.keras.activations.softmax)(output)

        self.model = tf.keras.Model(inputs=input, outputs=dense)


class CharCollector:
    """
    Character Collector Class

    1. Collect character from data-set (documents)
    2. Return the index of character set
    """
    pass
