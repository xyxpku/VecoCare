import torch.nn as nn

class ModelEmbeddings(nn.Module):
    def __init__(self, embed_size, vocab):
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.source = None
        src_pad_token_idx = vocab.src['<pad>']
        self.source = nn.Embedding(
            len(vocab.src), self.embed_size, padding_idx=src_pad_token_idx)