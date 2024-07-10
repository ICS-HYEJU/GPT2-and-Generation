import torch
import torch.nn as nn

import torch.nn.functional as F
from copy import deepcopy

import numpy as np
import json

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

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

def get_config_dict():
    dataset_info = dict(
    train_corpus =  "corpus.txt",
    eval_corpus = "corpus.txt",
    #
    name = "kowiki_small",
    path = 'kowiki_small.model',
    n_vocab = 8000,
    n_special_char = 7,
    #
    n_seq = 64,    # maximum sequence length
    n_layer = 6,   # number of transformer layers
    n_head = 4,    # number of multi-heads in attention layer
    d_hidn = 1024, # dimension of representation in each layer
    rate = 4,      # increase rate of dimensionality in bottleneck
    dropout = 0.1,
    pad_idx = 0,)
    #
    path = dict(
        save_base_path = 'runs'
    )
    #
    model = dict(
        name = 'gpt',
        generator = 'gpt_generation'
    )
    #
    solver = dict(
        name = 'Adam',
        base_lr = 1e-4,
        weight_decay = 1e-2,
        batch_train = 64,
        batch_eval = 64,
        epochs =100,
    )
    #
    scheduler = dict(
        name='LambdaLR',
        lr_lambda=lambda epoch: 0.95 ** epoch
    )
    #
    weight_info = dict(
        save_model_name = 'model.pth',
        save_checkpoint_path = 'checkpoint.pth',
        from_checkpoint = None.,
    )
    #
    device = dict(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    config = Config(dict(
        dataset_info = dataset_info,
        path = path,
        model = model,
        solver = solver,
        scheduler = scheduler,
        weight_info =  weight_info,
    ))

    return config