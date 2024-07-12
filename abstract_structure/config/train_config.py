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
        dataset_name="kowiki_small",
        train_corpus="corpus.txt",
        eval_corpus="corpus.txt",
        corpus_path="/storage/hjchoi/kowiki",
        #
        model_path = "/storage/hjchoi/kowiki/kowiki_small.model",
        vocab_path="/storage/hjchoi/kowiki/kowiki_small.vocab",
        n_vocab=8000,
        n_special_char=7,
        #
        n_seq=64,
        n_layer=6,
        n_head=4,
        d_hidn=1024,
        #
        rate=4,
        dropout=0.1,
        pad_idx=0,
        #
        batch_train=64,
        batch_eval=64,
        epoch=100,
    )

    path = dict(
        save_base_path='runs',
    )

    model = dict(
        name='GPT',
    )

    solver = dict(
        name='Adam',
        base_lr=1e-4,
        weight_decay=1e-2,
    )

    scheduler = dict(
        name='LambdaLR',
        lr_lambda=lambda epoch:0.95 ** epoch,
        max_epoch=20,
    )

    weight_info = dict(
        save_model_name='model.pth',              # save trained model weights to the file
        save_checkpoint_path='checkpoint.pth',    # save training state to the checkpoint file
        from_checkpoint=None,                     # load last training state from checkpoint file
    )

    device = dict(
        gpu_id=1,
    )

    # Merge all info into a dictionary variable
    config = dict(
        dataset_info=dataset_info,
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
        weight_info=weight_info,
        device=device
    )

    return Config(config)