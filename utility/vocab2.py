from collections import Counter
from itertools import chain
import json
import torch
from typing import List
from utils1 import read_corpus, pad_sents


class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()
        flat_corpus = []
        for i in corpus:
            flat_corpus.extend(i)
        word_freq = Counter(chain(*flat_corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab(object):
    def __init__(self, src_vocab: VocabEntry):
        self.src = src_vocab
    @staticmethod
    def build(src_sents, vocab_size, freq_cutoff) -> 'Vocab':
        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        return Vocab(src)

    def save(self, file_path):
        json.dump(dict(src_word2id=self.src.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        return Vocab(VocabEntry(src_word2id))

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


