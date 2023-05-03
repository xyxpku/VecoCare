import math

import numpy as np
import re


def pad_sents(sents, pad_token):
    sents_padded = []
    max_sent_count = max([len(sent) for sent in sents])
    for sent in sents:
        if len(sent) < max_sent_count:
            leftovers = [pad_token] * (max_sent_count - len(sent))
            sent.extend(leftovers)
        sents_padded.append(sent)

    return sents_padded

def seg_char(sent):
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]

    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars
    
def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        if source == 'src':
            sent = line.strip().split(' ')
        elif source == 'tgt':
            sent = seg_char(line.strip())
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
    return data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
