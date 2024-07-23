# import torch
# import torch.nn as nn
#
# import torch.nn.functional as F
# from copy import deepcopy
#
# import numpy as np
# import json
#
# import os
# import sentencepiece as spm
#
# from tqdm import tqdm
# from tqdm.notebook import tqdm as tqdm_notebook
# from tqdm.notebook import trange
#
# import math
#
# from typing import Dict, Tuple
#
# import torch.optim as optim
# from torch.optim import AdamW as Adam
# from torch.nn import LayerNorm
#
# class Config(dict):
#     __getattr__ = dict.__getitem__
#     __setattr__ = dict.__setitem__
#
#     @classmethod
#     def load(cls, file):
#         with open(file, 'r') as f:
#             config = json.loads(f.read())
#             return Config(config)
#
# def get_generation_config():
#     config_info = dict(
#         dataset_name="kowiki_small",
#         vocab_path="/storage/hjchoi/kowiki/kowiki_small.model",
#         n_vocab=8000,
#         n_special_char=7,
#         #
#         n_seq=64,
#         n_layer=6,
#         n_head=4,
#         d_hidn=1024,
#         #
#         model_name="GPT_Generation",
#         #
#         rate=4,
#         dropout=0.1,
#         pad_idx=0,
#         nucleus_prob=0.75,
#         n_sample=5,
#         #
#         from_saved_model="model_weight.pth",     # load last training state from checkpoint file
#         device=1
#     )
#
#     config = Config(config_info)
#     return config