import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from copy import deepcopy

import numpy as np
import json
import glob
import os
import sentencepiece as spm

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange

import math

from typing import Dict, Tuple

import torch.optim as optim
from torch.optim import AdamW as Adam
from torch.nn import LayerNorm

from abstract_structure.config.train_config import Config
class TokenizedCorpus(Dataset):
    def __init__(self, config, mode='train'):
        #
        self.config = config
        self.mode = mode
        self.saved_vocab = config.dataset_info['vocab_path']
        self.saved_model = config.dataset_info['model_path']
        #
        assert self.saved_model is not None, "No kowiki_small.model"
        assert self.saved_vocab is not None, "No kowiki_small.vocab"
        #
        self.corpus_path = config.dataset_info['corpus_path']
        self.corpus_file = sorted(glob.glob(self.corpus_path + '/*.json'))


    def __len__(self):
        assert len(self.corpus_file) != 0 , "corpus_file is empty"
        return len(self.corpus_file)

    def __getitem__(self, item):
        with open(self.corpus_file[item], 'r') as f:
            self.tokens = json.load(f)
        return (torch.tensor(self.tokens['tokens'][:-1]),
                torch.tensor(self.tokens['tokens'][1:]),
                torch.tensor(item))


    @staticmethod
    def collate_fn(data):
        inputs, outputs, item = list(zip(*data))
        #
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=0)
        #
        batch= [
            inputs,
            outputs,
            torch.stack(item, dim=0)
        ]
        return batch

if __name__ == '__main__':
    from abstract_structure.config.train_config import get_config_dict
    config = Config(get_config_dict())

    obj = TokenizedCorpus(config)
    #
    obj_loader = DataLoader(
        obj,
        batch_size=config.dataset_info['batch_train'],
        shuffle=True,
        collate_fn=TokenizedCorpus.collate_fn
    )
    for i, data in enumerate(obj_loader):
        print(i, data)