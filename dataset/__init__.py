from torch.utils.data import DataLoader

def create_dataloader(config, mode="train"):
    if config.dataset_info['dataset_name']:
        from abstract_structure.dataset.dataset import TokenizedCorpus
    else:
        raise ValueError("Invalid dataset name, currently supported [ kowiki_small ]")
    #
    if mode == 'train':
        print("train_mode")
        object = TokenizedCorpus(config)
        object_loader = DataLoader(
            object,
            batch_size=config.dataset_info['batch_train'],
            shuffle=True,
            collate_fn = TokenizedCorpus.collate_fn
        )
    elif mode == 'eval':
        print("eval_mode")
        object = TokenizedCorpus(config)
        object_loader = DataLoader(
            object,
            batch_size=config.dataset_info['batch_eval'],
            collate_fn=TokenizedCorpus.collate_fn
        )
    else:
        raise ValueError("Invalid mode")

    return object_loader