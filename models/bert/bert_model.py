# coding=utf-8
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Input, Lambda, Dense
from keras import Model
from models.bert.bert_configure import BertConfigure
import numpy as np
import codecs
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir)))


class BertModel(object):
    def __init__(self):
        self.model_configure = BertConfigure()
        self.model = self._get_model()
        self.token_dict = self._read_token_dict()
        self.tokenizer = self.OurTokenizer(self.token_dict)
        self._init_model()

    class OurTokenizer(Tokenizer):
        def _tokenize(self, text):
            R = []
            for c in text:
                if c in self._token_dict:
                    R.append(c)
                elif self._is_space(c):
                    R.append('[unused1]')  # space类用未经训练的[unused1]表示
                else:
                    R.append('[UNK]')  # 剩余的字符是[UNK]
            return R

    def _read_token_dict(self):
        token_dict = {}

        with codecs.open(self.model_configure.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return token_dict

    def _get_model(self):
        # todo seq_len
        bert_model = load_trained_model_from_checkpoint(self.model_configure.config_path,
                                                        self.model_configure.checkpoint_path, seq_len=None)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)
        p = Dense(1, activation='sigmoid')(x)

        model = Model([x1_in, x2_in], p)
        return model

    def _init_model(self):
        sentence1 = '今天天气真好'
        sentence2 = '今天天气不好'
        self.model.load_weights(self.model_configure.restore_model_path)
        X1, X2 = self._data_preprocesser(sentence1, sentence2)
        self.model.predict([X1, X2])

    def _seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])

    def _data_preprocesser(self, sentence1, sentence2):
        X1, X2 = [], []
        for tmp_sent1, tmp_sent2 in zip(sentence1, sentence2):
            x1, x2 = self.tokenizer.encode(first=tmp_sent1[:self.model_configure.maxlen],
                                           second=tmp_sent2[:self.model_configure.maxlen])
            X1.append(x1)
            X2.append(x2)
        X1 = self._seq_padding(X1)
        X2 = self._seq_padding(X2)
        return X1, X2

    def predict(self, sentence1, sentence2):
        # self.model.load_weights(self.restore_model_path)
        X1, X2 = self._data_preprocesser(sentence1, sentence2)
        y_pred = self.model.predict([X1, X2], batch_size=1024)
        return y_pred
