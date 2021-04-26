"""
library of character-based RNN
@Author Chanwoo Gwon, Yonsei Univ. Researcher, since 2020.05. ~
@Date 2020.10.22
"""

import tensorflow as tf
import os
import codecs
import numpy as np

from model.core import ModelCore, LOSS, Net


class CharRNN(ModelCore):
    """
    RNN Library Class

    1. Train and Predict using RNN based on Character
    2. Could be used like the encoder
    3. Can many-to-many or many-to-one RNN
    """
    def __init__(self, data_path, emb=100, last_dim=64, loss=LOSS.SPARSE_CATEGORICAL_CROSSENTROPY):

        self._text_set = None
        self.__emb = emb
        self._char_set = []
        self._char2idx = None
        self._idx2char = None
        self.__text_as_int = None
        self.__last_dim = last_dim

        super().__init__(data_path, loss=loss)

    def read_data(self):
        char_path = os.path.join(self._data_path, 'data.txt')

        self._text_set = open(char_path, 'rb').read().decode(encoding='utf-8')
        self._char_set = sorted(set(self._text_set))

        self._char2idx = {u: i for i, u in enumerate(self._char_set)}
        self._idx2char = np.array(self._char_set)

    def build_model(self):
        vocab_size = len(self._char_set)
        embb = tf.keras.layers.Embedding(vocab_size, self.__emb, batch_input_shape=[100, None])
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,
                                                                  return_sequences=True,
                                                                  stateful=True,
                                                                  recurrent_initializer='glorot_uniform'))(embb)
        dense = tf.keras.layers.Dense(self.__last_dim, activation='relu')(lstm)

        return tf.keras.Model(inputs=embb, outputs=dense)

    def to_vector(self, sequence):
        return [self._char2idx[c] for c in sequence]

    def to_text(self, sequence):
        return repr("".join(self._idx2char[sequence]))


class TypoClassifier(CharRNN):
    def __init__(self, data_path):
        super().__init__(data_path, last_dim=1, loss=LOSS.CATEGORICAL_CROSSENTROPY)

    def read_data(self):
        super().read_data()

        data_temp = []
        char_world = ''
        max_length = -1
        for line in self._text_set.split('\n'):
            split_data = line.strip().split('\t')

            typo = split_data[0]
            answer = split_data[1]
            data_temp.append([typo, int(answer)])
            char_world = "{0}{1}".format(char_world, split_data[0])
            max_length = max(max_length, len(typo))

        # make char set using typo only
        self._char_set = sorted(set(char_world))
        self._char2idx = {u: i for i, u in enumerate(self._char_set)}
        self._char2idx[' '] = len(self._char_set)  # add empty word
        self._idx2char = np.array(self._char_set)

        data_all = []
        for item in data_temp:
            item_one = item[0]
            if len(item_one) < max_length:
                item_one = "{0}{1}".format(item_one, ''.join([' '] * (max_length - len(item_one))))
            data_all.append([[self.to_vector(item_one)], [item[1]]])

        sp = int(len(data_all) * 0.8)

        self._train_data.set(
            [tf.convert_to_tensor(item[0], dtype=tf.int64) for item in data_all[:sp]],
            [tf.convert_to_tensor(item[1], dtype=tf.int64) for item in data_all[:sp]]
        )
        self._test_data.set(
            [tf.convert_to_tensor(item[0], dtype=tf.int64) for item in data_all[sp:]],
            [tf.convert_to_tensor(item[1], dtype=tf.int64) for item in data_all[sp:]]
        )

        print("train data:", len(self._train_data))
        print("test data:", len(self._test_data))


class CharCollector:
    """
    Character Collector Class

    1. Collect character from data-set (documents)
    2. Return the index of character set
    """
    pass
