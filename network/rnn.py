"""
library of character-based RNN
@Author Chanwoo Gwon, Yonsei Univ. Researcher, since 2020.05. ~
@Date 2020.10.22
"""

import tensorflow as tf
import os
import codecs
import numpy as np


class CharRNN:
    """
    RNN Library Class

    1. Train and Predict using RNN based on Character
    2. Could be used like the encoder
    3. Can many-to-many or many-to-one RNN
    """
    def __init__(self, base_path, emb=100, last_dim=64):
        self.__model = None
        self.__emb = emb
        self.__char_set = []
        self.__char2idx = None
        self.__idx2char = None
        self.__base_path = base_path
        self.__last_dim = last_dim

        self.__read_char()
        self.__make_vector()

    def __read_char(self):
        char_path = os.path.join(self.__base_path, '')
        with codecs.open(char_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                self.__char_set.append(line.strip())

    def __make_vector(self):
        self.__char2idx = {u: i for i, u in enumerate(self.__char_set)}
        self.__idx2char = np.array(self.__char_set)

    def __build_network(self):
        vocab_size = len(self.__char_set)
        embb = tf.keras.layers.Embedding(vocab_size, self.__emb, batch_input_shape=[100, None])
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'))(embb)
        dense = tf.keras.layers.Dense(self.__last_dim, activation='relu')(lstm)

        return tf.keras.Model(inputs=embb, outputs=dense)

    def to_vector(self, sequence):
        text_as_int = np.array([self.__char2idx[c] for c in sequence])
        return tf.data.Dataset.from_tensor_slices(text_as_int)

    def to_text(self, sequence):
        return repr("".join(self.__idx2char[sequence]))

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass


class CharCollector:
    """
    Character Collector Class

    1. Collect character from data-set (documents)
    2. Return the index of character set
    """
    pass
