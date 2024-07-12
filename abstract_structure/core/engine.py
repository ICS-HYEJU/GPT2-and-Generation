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

from abstract_structure.config.train_config import Config
class Trainer():
    def __init__(self, cfg, device):
        self.cfg = Config(cfg)
        self.device = self.cfg.device['gpu_id']

        # ===== Save Path =====
        self.save_path = self.make_save_path()

        # ===== TensorBoard =====
        #self.tblogger = SummaryWirtier(self.save_path)

        # ===== DataLoader =====
        self.train_loader = self.get_dataloader()

        # ===== Model =====
        self.model = self.build_model()

        # ===== Optimizer =====
        self.optimizer =self.build_optimizer()

        # ===== Scheduler =====
        self.scheduler =self.build_scheduler()

        # ===== Loss =====
        self.criterion = self.set_criterion()

        # ===== Parameters =====
        self.max_epoch = self.cfg.solver['max_epoch']
        self.max_stepnum = len(self.train_loader)

    def cal_loss(self, logits, labels):
        logits = logits.view(-1, logits.size(2)).to(self.device)
        labels = labels.view(-1).to(self.device)

    def set_criterion(self):
        nn.CrossEntropyLoss(ignore_index=self.cfg.dataset_info['pad_idx'], reduction='mean').to(self.device)

    def build_scheduler(self):
        from abstract_structure.solver.fn_scheduler import build_scheduler
        return build_scheduler(self.cfg, self.optimizer)

    def build_optimizer(self):
        from abstract_structure.solver.fn_optimizer import build_optimizer
        return build_optimizer(self.cfg, self.model)

    def build_model(self):
        name = self.cfg.model['name']
        if name == "GPT":
            from abstract_structure.model.gpt import Transformer
            model = Transformer(self.cfg)
        else:
            raise NotImplementedError(f'The required model is not implemented yet..')
        return model.to(self.device)

    def get_dataloader(self):
        save_pretrain = os.path.join(self.cfg.path['save_pretrain'],
                                     self.cfg.model['name'] + "_pretrain")
        os.makedirs(save_pretrain, exist_ok=True)

    def start_train(self):
