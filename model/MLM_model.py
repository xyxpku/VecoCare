import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
import numpy as np

from functools import reduce

class text_visit_MLM():
    def __init__(
        self,
        text_mask_prob = 0.15,
        text_mask_keep_rand = [0.8,0.1,0.1],
        text_num_tokens = None,
        text_mask_token_id = 111,
        text_pad_token_id = 0,
        text_mask_ignore_token_ids = [],
        visit_mask_prob=0.15,
        visit_mask_keep_rand=[0.8, 0.1, 0.1],
        visit_num_tokens=None,
        visit_mask_token_id=111,
        visit_pad_token_id=0,
        visit_mask_ignore_token_ids=[]):

        self.text_mask_prob = text_mask_prob
        self.text_mask_keep_rand = torch.FloatTensor(text_mask_keep_rand)
        self.text_num_tokens = text_num_tokens
        self.text_mask_token_id = text_mask_token_id
        self.text_pad_token_id = text_pad_token_id
        self.text_mask_token_id = text_mask_token_id
        self.text_mask_ignore_token_ids = set([*text_mask_ignore_token_ids, text_pad_token_id])

        self.visit_mask_prob = visit_mask_prob
        self.visit_mask_keep_rand = torch.FloatTensor(visit_mask_keep_rand)
        self.visit_num_tokens = visit_num_tokens
        self.visit_mask_token_id = visit_mask_token_id
        self.visit_pad_token_id = visit_pad_token_id
        self.visit_mask_ignore_token_ids = set([*visit_mask_ignore_token_ids, visit_pad_token_id])

    def mask_with_tokens(self, x, token_ids):
        init_no_mask = torch.full_like(x, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (x == el), token_ids, init_no_mask)
        return mask

    def mask_text_out(self, x, device):
        mask_filter = ~self.mask_with_tokens(x, self.text_mask_ignore_token_ids)
        bs, slen = x.size()
        pred_mask = np.random.rand(bs, slen) <= self.text_mask_prob
        pred_mask = torch.from_numpy(pred_mask.astype(np.bool)).to(device)
        mask_final = mask_filter & pred_mask
        _x_real = x[mask_final]
        _x_rand = _x_real.clone().random_(self.text_num_tokens)
        _x_mask = _x_real.clone().fill_(self.text_mask_token_id)
        probs = torch.multinomial(self.text_mask_keep_rand, len(_x_real), replacement=True).to(device)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(mask_final, _x)
        assert x.size() == (bs, slen)
        assert mask_final.size() == (bs, slen)

        return x, _x_real, mask_final


    def mask_visit_out(self,x,device,codebook_len):
        bs,slen,clen = x.size()
        mask_filter = ~self.mask_with_tokens(x, self.visit_mask_ignore_token_ids)

        pred_mask = np.random.rand(bs, slen, clen) <= self.visit_mask_prob
        pred_mask = torch.from_numpy(pred_mask.astype(np.bool)).to(device)
        mask_final = mask_filter & pred_mask

        mask_visit = torch.sum(mask_final.long(),dim=-1) > 0
        mask_index = mask_final.nonzero(as_tuple=False)
        pa_vs_index = mask_index[:,:2]

        _x_real = x[mask_final]
        _x_rand = _x_real.clone().random_(self.visit_num_tokens)
        _x_mask = _x_real.clone().fill_(self.visit_mask_token_id)

        probs = torch.multinomial(self.visit_mask_keep_rand, len(_x_real), replacement=True).to(device)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(mask_final, _x)

        col_index = _x_real.unsqueeze(1)
        code_mask_index = torch.cat((pa_vs_index,col_index),dim=1)

        code_mask_label = torch.zeros((bs,slen,codebook_len)).to(device)
        code_mask_value = torch.ones((code_mask_index.size(0),)).to(device)

        code_mask_label.index_put_((code_mask_index[:,0],code_mask_index[:,1],code_mask_index[:,2]),code_mask_value)

        assert x.size() == (bs, slen, clen)
        assert mask_final.size() == (bs, slen, clen)

        return x, code_mask_label, mask_visit