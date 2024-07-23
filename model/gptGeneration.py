import torch
import torch.nn as nn
import os
import sentencepiece as spm


class GPT2Generation():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.saved_vocab = self.cfg.dataset_info['vocab_path']
        #
        self.vocab = self.load_vocab_tokenizer()
        self.ge_model = self.construct_model()
        # self.state = self.load_state()

    def load_vocab_tokenizer(self):
        assert self.saved_vocab is not None and os.path.exists(self.saved_vocab), "No kowiki_small.vocab"
        #
        vocab = spm.SentencePieceProcessor()
        vocab.Load(self.saved_vocab)
        #
        assert vocab.vocab_size() - 7 == self.cfg.dataset_info['n_vocab'], \
            "the size of vocabulary is not the same..."
        #
        print("[SubTokenizer] Loading Completed....")

        return vocab

    def construct_model(self) -> nn.Module:
        from abstract_structure.model.gpt import ModelEngine
        ge_model = ModelEngine(config=self.cfg, dropout=0, training=False, bidirectional=False)
        print("[Transformer] Loading Completed...")

        return ge_model.to(self.device)

    def load_state(self):
        assert os.path.exists(
            self.cfg.generation_info['from_saved_model']), "There is no saved weight file...{}".format(
            self.cfg.generation_info['from_saved_model'])
        #
        state_dict = torch.load(self.cfg.generation_info['from_saved_model'], map_location=self.device)
        state = self.ge_model.load_state_dict(state_dict['model'])
        return state

    def encode_context(self, context: str):
        tokens = [self.vocab.PieceToId(t) for t in self.vocab.EncodeAsPieces(context)]
        tokens = [self.vocab.bos_id()] + tokens
        return tokens

    def decode_tokens(self, tokens) -> str:
        if self.vocab.eos_id() in tokens:
            tokens = tokens[:tokens.index(self.vocab.eos_id()) + 1]
        return self.vocab.DecodeIds(tokens)

    def generate(self, context: str) -> str:
        words = self.encode_context(context)

        current, past = words, None
        while (len(words) < self.cfg.dataset_info['n_seq']):
            current = [current]
            # Predict the Next word token from the given context
            probs, past = self._predict_probs(current, past)
            next_word = self._sample_from_top_p(probs)

            # Change the context to the predicted word.
            words.append(next_word)
            current = [next_word]
        return self.decode_tokens(words)

    @torch.no_grad()
    def _predict_probs(self, words, past):
        x = torch.tensor(words, dtype=torch.long).to(self.device)

        logits, past = self.ge_model(x, past)
        logits = logits.float().to(self.device)
        logits = logits[0]

        return logits[-1, :].softmax(dim=-1), past

    def _sample_from_top_p(self, probs: torch.Tensor) -> int:
        probs, indices = probs.sort(descending=True)

        # Ex) probs = [0.5, 0.2, 0.1, 0.1, 0.1]
        #     probs.cumsum(-1) = [0.5, 0.7, 0.8, 0.9, 1.0]
        mask = probs.cumsum(-1) > self.cfg.generation_info['nucleus_prob']
        # mask = [False, False, False, True, True]
        mask[0] = False
        # mask = [False, False, False, True, True]
        # ���� mask[0] = False �� ���� ������ probs = [0.95, 0.03, 0.02, 0, 0] �� ������ �������� ��.
        probs.masked_fill_(mask, 0)
        # probs = [0.5, 0.2, 0.1, 0.0, 0.0]

        # Sample from filtered distribution
        sampled_indices = indices[probs.multinomial(self.cfg.generation_info['n_sample'])[0]].item()

        return sampled_indices


if __name__ == '__main__':
    from config.train_config import get_config_dict

    cfg = get_config_dict()

    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')

    generator = GPT2Generation(cfg=cfg, device=device)

    words = generator.generate('electronic engineering is')
    print(words)
