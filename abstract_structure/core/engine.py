import torch
import torch.nn as nn

import torch.nn.functional as F
from copy import deepcopy

import numpy as np
import json

import os
import sentencepiece as spm

from tqdm import tqdm
import time
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange

import math

from typing import Dict, Tuple

import torch.optim as optim
from torch.optim import AdamW as Adam
from torch.nn import LayerNorm

from abstract_structure.config.train_config import Config
class Engine():
    def __init__(self, cfg, mode, device):
        self.cfg = Config(cfg)
        self.mode = mode
        self.device = device

        # ===== Save Path =====
        self.save_path = self.make_save_path()

        # ===== TensorBoard =====
        #self.tblogger = SummaryWirtier(self.save_path)

        # ===== DataLoader =====
        self.dataloader = self.get_dataloader()

        # ===== Model =====
        self.model = self.build_model()
        # self.ge_model = self.build_ge_model()

        # ===== Optimizer =====
        self.optimizer =self.build_optimizer()

        # ===== Scheduler =====
        self.scheduler =self.build_scheduler()

        # ===== Loss =====
        self.criterion = self.set_criterion()

        # ===== Parameters =====
        self.max_epoch = self.cfg.solver['max_epoch']
        self.max_stepnum = len(self.dataloader)

    def cal_loss(self, logits, labels):
        logits = logits.view(-1, logits.size(2)).to(self.device)
        labels = labels.view(-1).to(self.device)
        return self.criterion(logits, labels)

    def set_criterion(self):
        return nn.CrossEntropyLoss(ignore_index=self.cfg.dataset_info['pad_idx'], reduction='mean').to(self.device)

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
        from abstract_structure.dataset import create_dataloader
        dataloader = create_dataloader(self.cfg, mode=self.mode)
        return dataloader

    def make_save_path(self):
        save_pretrain = os.path.join(self.cfg.path['save_base_path'],
                                     self.cfg.model['name'] + "_pretrain")
        os.makedirs(save_pretrain, exist_ok=True)
        return save_pretrain

    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        with tqdm(total=len(self.dataloader), desc=f"Train({epoch})", leave=True) as pbar:
            for step, data in enumerate(self.dataloader):
                # data[0] = inputs
                # data[1] = outputs
                # data[2] = item
                logits = self.model((data[0]).to(self.device))
                # Shape of logits :
                # Shape of logits.view(-1, logits.size(2)) :
                # shape of data['output'].view(-1)  :
                loss = self.cal_loss(logits, data[1].to(self.device))
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

                pbar.set_postfix_str(f"one epoch Loss: {loss:.3f} Avg Loss in batch: {np.mean(losses):.3f}")

        return np.mean(losses)

    def start_train(self):
        epoch = 0
        try:
            print(f"Training Start...")
            start_time = time.time()
            losses = []
            with tqdm(range(self.cfg.dataset_info['epoch']), desc="Epoch") as pbar:

                # Clear CUDA cache which is used for training.
                torch.cuda.empty_cache()

                # below loss means that mean value of losses for one epoch
                loss = self.train_one_epoch(epoch)
                losses.append(loss)

                # Clear CUDA cache which is used for evaluation
                torch.cuda.empty_cache()

                self.scheduler.step()

                # Save
                torch.save({'model:':self.model.state_dict()},
                               self.save_path + self.cfg.weight_info['save_model_name'])
                #

                pbar.set_postfix_str(f"Curr Loss: {loss:.3f}, Avg Loss: {np.mean(losses):.3f}")
                #
                epoch += 1
            print(f'\nTraining completed in {(time.time() - start_time) / 3600:.3f} hours.')

        except Exception as _:
            print('ERROR in training loop or eval/save model.')
            raise

    @torch.no_grad()
    def eval(self, config):
        self.model.eval()

        losses = []
        for step, data in enumerate(self.dataloader):
            logits, _ = self.model(data[0], past=None)
            loss = self.cal_loss(logits, data[1].to(self.device))

            losses.append(loss.item())

        return np.mean(losses)


if __name__ == '__main__':
    from abstract_structure.config.train_config import get_config_dict

    #
    cfg = get_config_dict()
    #
    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')
    #
    engine = Engine(cfg, mode='train',device=device)
    #
    engine.start_train()