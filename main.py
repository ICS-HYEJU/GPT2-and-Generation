
import torch

from core.engine import Engine

from config.train_config import get_config_dict

from model.gptGeneration import GPT2Generation


if __name__ == '__main__':
    #
    cfg = get_config_dict()
    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')
    #
    engine = Engine(cfg, mode='train',device=device)
    engine.start_train()
    #

    generator = GPT2Generation(cfg=cfg, device=device)

    words = generator.generate('electronic engineering is')
    print(words)
