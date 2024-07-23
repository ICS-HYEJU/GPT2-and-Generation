import torch
from torch.utils.data import Dataset, DataLoader
import json
import glob
import os
import sentencepiece as spm

from config.train_config import Config


class TokenizedCorpus(Dataset):
    def __init__(self, config, mode='train'):
        #
        self.config = Config(config)
        self.mode = mode
        #
        self.saved_vocab = config.dataset_info['vocab_path']
        self.vocab = self.make_subtokenizer()
        #
        self.corpus_path = config.dataset_info['corpus_path']
        print("Data Loading...")
        self.corpus_file = glob.glob(self.corpus_path + '/*.json')
        print("Loading complete")

    def __len__(self):
        assert len(self.corpus_file) != 0, "corpus_file is empty"
        return len(self.corpus_file)

    def __getitem__(self, item):
        with open(self.corpus_file[item], 'r') as f:
            self.tokens = json.load(f)
            self.tokens['tokens'].append(0)
        return (torch.tensor(self.tokens['tokens'][:-1]),
                torch.tensor(self.tokens['tokens'][1:]),
                torch.tensor(item))

    def make_subtokenizer(self):
        if self.mode == "train":
            print("Loading subtokenizer-- training")
            if self.saved_vocab is not None and os.path.exists(self.saved_vocab):
                vocab = spm.SentencePieceProcessor()
                vocab.Load(self.saved_vocab)
                #
                assert vocab.vocab_size() - 7 == self.config.dataset_info[
                    'n_vocab'], "the size of vocabulary is not the same..."
                #
                print("[SubTokenizer] Loading Complete...")
            else:
                spm.SentencePieceTrainer.Train(
                    f"--input={self.config.dataset_info['train_corpus']} --model_prefix={config.dataset_info['dataset_name']} --vocab_size={config.dataset_info['n_vocab'] + 7}" +
                    " --model_type=bpe" +
                    " --max_sentence_length=999999" +
                    " --pad_id=0 --pad_piece=[PAD]" +
                    " --unk_id=1 --unk_piece=[UNK]" +
                    " --bos_id=2 --bos_piece=[BOS]" +
                    " --eos_id=3 --eos_piece=[EOS]" +
                    " --user_defined_symbols=[SEP],[CLS],[MASK]")
                #
                vocab = spm.SentencePieceProcessor()
                vocab.Load(config.dataset_info['dataset_name'] + ".model")
                #
                print("[SubTokenizer] Training Completed...")
        else:
            print("Loading subtokenizer--evaluating")
            assert self.saved_vocab is not None and os.path.exists(self.saved_vocab), "No kowiki_small.vocab"

            vocab = spm.SentencePieceProcessor()
            vocab.Load(self.saved_vocab)
            #
            assert self.vocab.vocab_size() - 7 == self.config.dataset_info['n_vocab'], \
                "the size of vocabulary is not the same..."
            #
            print("[SubTokenizer] Loading Completed....")

        return vocab

    @staticmethod
    def collate_fn(data):
        inputs, outputs, item = list(zip(*data))
        #
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=0)
        #
        batch = [
            inputs,
            outputs,
            torch.stack(item, dim=0)
        ]
        return batch


if __name__ == '__main__':
    from config.train_config import get_config_dict

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
