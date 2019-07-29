# coding=utf-8
import os


class BertConfigure(object):
    def __init__(self):
        self.config_path = '/Data/public/Bert/chinese_L-12_H-768_A-12/bert_config.json'
        self.checkpoint_path = '/Data/public/Bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
        self.maxlen = 32
        self.vocab_path = '/Data/public/Bert/chinese_L-12_H-768_A-12/vocab.txt'
        self.restore_model_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)),
            'saved_models/bert_0726_0942.h5')

