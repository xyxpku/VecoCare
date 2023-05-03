import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import copy

from model.transformer_hita_attention import TransformerTime
from model.transformer_text import NMT_tran
from model.MLM_model import text_visit_MLM
from model.transformer_with_local import Transformer_local
import random


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim,
            dropout = 0.0
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def length_to_mask(length, max_len=None, dtype=None):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

class Model(nn.Module):
    def __init__(self, args,
                 max_visit_len, code_num, label_num, use_pretrain, embedding_weights, code2id, vocab, max_notes_len, train_patient_num):
        super(Model, self).__init__()
        self.device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
        self.max_visit_len = max_visit_len
        self.code_num = code_num
        self.label_num = label_num
        self.use_pretrain = use_pretrain
        self.max_notes_len = max_notes_len
        self.text_mask_token_id = len(vocab.src)
        self.visit_mask_token_id = len(code2id) + 2
        self.visit_mask_ignore_token_ids = [len(code2id) + 1]
        self.vocab = vocab
        self.code2id =code2id

        if self.use_pretrain:
            self.text_embed = embedding_weights.shape[1]
            V, D = embedding_weights.shape
            self.word_embedding = nn.Embedding(num_embeddings=V+1, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))
        else:
            self.text_embed = args.word_embedding_size
            src_pad_token_idx = vocab.src['<pad>']
            self.word_embedding = nn.Embedding(len(vocab.src)+1, self.text_embed, padding_idx=src_pad_token_idx)
            self.word_embedding.apply(init_weights)
        self.text_mask_ignore_token_ids = [vocab.src['[CLS]']]
        self.textEncoder_q = NMT_tran(hidden_size=self.text_embed,head_num=args.text_encoder_head_num,initial_embedding = self.word_embedding
                                    ,encoder_layers=args.text_encoder_layer,dropout_rate=args.text_dropout_prob)
        self.train_dropout_rate = args.hita_dropout_prob
        self.hita_input_size = args.hita_input_size
        self.hita_time_selection_layer_global = [args.hita_time_selection_layer_global_embed,args.global_query_size]
        self.hita_time_selection_layer_encoder = [args.hita_time_selection_layer_encoder_embed,self.hita_input_size]
        self.visitEncoder_q = TransformerTime(args,self.hita_input_size,self.max_visit_len,len(code2id),
                                                  self.hita_time_selection_layer_global, self.hita_time_selection_layer_encoder,dropout_rate=args.hita_dropout_prob)
        self.ehrcode_embed = nn.Embedding(self.code_num, self.hita_input_size, padding_idx=0)
        self.ehrcode_embed.apply(init_weights)

        assert self.hita_input_size == self.text_embed
        self.local_kernel_size = args.local_kernel_size
        self.fusingEncoder = Transformer_local(hidden_size=self.text_embed,head_num=args.fusing_encoder_head_num,initial_embedding = self.word_embedding
                                    ,encoder_layers=args.fusing_encoder_layer, local_kernel_size = self.local_kernel_size, dropout_rate=args.fusing_dropout_prob)
        self.token_type_embeddings = nn.Embedding(2, self.text_embed)
        self.token_type_embeddings.apply(init_weights)

        self.mlm_preprocesser = text_visit_MLM(args.text_mask_prob,eval(args.text_mask_keep_rand), len(vocab.src), self.text_mask_token_id, vocab.src['<pad>'], self.text_mask_ignore_token_ids,
                                               args.visit_mask_prob, eval(args.visit_mask_keep_rand), len(code2id) + 2, self.visit_mask_token_id, 0, self.visit_mask_ignore_token_ids)
        self.mlm_seqproj = nn.Linear(self.hita_input_size, len(code2id) + 2, bias=True)
        self.mlm_textproj = nn.Linear(self.text_embed, len(vocab.src) , bias=True)

        self.text_projection = ProjectionHead(embedding_dim=self.text_embed,projection_dim=args.projection_dim)
        self.visit_projection = ProjectionHead(embedding_dim=self.hita_input_size,projection_dim=args.projection_dim)

        self.ablation = args.ablation
        if "no_patient_memory" not in self.ablation:
            self.classifier_fc = nn.Linear(3 * self.text_embed + self.hita_input_size, self.label_num)
        else:
            self.classifier_fc = nn.Linear(self.text_embed + self.hita_input_size, self.label_num)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.train_dropout_rate = args.train_dropout_rate
        self.dropout = nn.Dropout(self.train_dropout_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.gwd_tau = args.gwd_tau

        self.register_buffer("patient_visits", torch.randn(self.hita_input_size, train_patient_num))
        self.register_buffer("patient_texts", torch.randn(self.text_embed, train_patient_num))

        self._init_weight()

    def _init_weight(self):
        self.textEncoder_q.apply(init_weights)
        self.visitEncoder_q.apply(init_weights)
        self.text_projection.apply(init_weights)
        self.visit_projection.apply(init_weights)
        self.classifier_fc.apply(init_weights)

        nn.init.xavier_uniform_(self.patient_visits)
        nn.init.xavier_uniform_(self.patient_texts)

    def calc_cost_gwd(self,x,y, tau=0.5):
        cos_dis = torch.bmm(x, torch.transpose(y, 1, 2))
        cos_dis = torch.exp(- cos_dis / tau)
        return cos_dis

    def calc_cl_loss(self, x_batch, s_batch, s_batch_dim2,
                        seq_time_batch, mask_mult, mask_final, notes_batch, notes_len_batch):
        sequence_embedding = torch.matmul(x_batch, self.ehrcode_embed.weight)
        sequence_contexts_q, sequence_embedding_q = self.visitEncoder_q(sequence_embedding,
                                                                                seq_time_batch, mask_mult,
                                                                                mask_final, s_batch,
                                                                                self.device)
        sequence_mask = length_to_mask(s_batch, self.max_visit_len).unsqueeze(1)

        text_mask = length_to_mask(notes_len_batch, self.max_notes_len).unsqueeze(1)
        text_contexts_q, text_embedding_q = self.textEncoder_q("calc_only_text", notes_batch, text_mask)

        sequence_contexts_q= self.visit_projection(sequence_contexts_q)
        text_contexts_q= self.text_projection(text_contexts_q)
        sequence_embedding_q = self.visit_projection(sequence_embedding_q)
        text_embedding_q = self.text_projection(text_embedding_q)
        sequence_embedding_q = nn.functional.normalize(sequence_embedding_q, dim=1)
        text_embedding_q = nn.functional.normalize(text_embedding_q, dim=1)

        batch_size = sequence_contexts_q.shape[0]
        seq_contexts_sample = sequence_contexts_q
        text_contexts_sample = text_contexts_q
        sample_size = batch_size

        sequence_contexts_q_trans = seq_contexts_sample.repeat(1,sample_size,1).view(sample_size*sample_size,seq_contexts_sample.shape[1],-1)
        text_contexts_q_trans = text_contexts_sample.repeat(sample_size,1,1)
        sequence_contexts_q_retrans = seq_contexts_sample.repeat(sample_size,1,1)
        text_contexts_q_retrans = text_contexts_sample.repeat(1,sample_size,1).view(sample_size*sample_size,text_contexts_sample.shape[1],-1)

        gw_v2t = self.gwd(sequence_contexts_q_trans.transpose(2,1), text_contexts_q_trans.transpose(2,1), self.gwd_tau)
        logits_gw_v2t = torch.exp(-gw_v2t / 0.1).view(sample_size,sample_size)
        gw_t2v = self.gwd(text_contexts_q_retrans.transpose(2,1), sequence_contexts_q_retrans.transpose(2,1), self.gwd_tau)
        logits_gw_t2v = torch.exp(-gw_t2v / 0.1).view(sample_size,sample_size)

        labels_gw = torch.arange(sample_size, device=self.device, dtype=torch.long)
        loss_gw_v2t =  F.cross_entropy(logits_gw_v2t, labels_gw)
        loss_gw_t2v = F.cross_entropy(logits_gw_t2v, labels_gw)

        return loss_gw_v2t, loss_gw_t2v

    def gwd(self, X, Y, tau, lamda=1e-1, iteration=5, OT_iteration=20):
        m = X.size(2)
        n = Y.size(2)
        bs = X.size(0)
        p = (torch.ones(bs, m, 1)/m).to(self.device)
        q = (torch.ones(bs, n, 1)/n).to(self.device)
        return self.GW_distance(X, Y, p, q, tau, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

    def GW_distance(self, X, Y, p, q, tau, lamda=0.5, iteration=5, OT_iteration=20):
        Cs = self.cos_batch(X, X, tau).float().to(self.device)
        Ct = self.cos_batch(Y, Y, tau).float().to(self.device)
        bs = Cs.size(0)
        m = Ct.size(2)
        n = Cs.size(2)
        T, Cst = self.GW_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
        temp = torch.bmm(torch.transpose(Cst,1,2), T)
        distance = self.batch_trace(temp, m, bs)
        return distance

    def GW_batch(self, Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
        one_m = torch.ones(bs, m, 1).float().to(self.device)
        one_n = torch.ones(bs, n, 1).float().to(self.device)

        Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
            torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2)))
        gamma = torch.bmm(p, q.transpose(2,1))
        for i in range(iteration):
            C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
            gamma = self.OT_batch(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
        Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        return gamma.detach(), Cgamma

    def OT_batch(self, C, bs, n, m, beta=0.5, iteration=50):
        sigma = torch.ones(bs, int(m), 1).to(self.device)/float(m)
        T = torch.ones(bs, n, m).to(self.device)
        A = torch.exp(-C/beta).float().to(self.device)
        for t in range(iteration):
            Q = A * T
            for k in range(1):
                delta = 1 / (n * torch.bmm(Q, sigma))
                a = torch.bmm(torch.transpose(Q,1,2), delta)
                sigma = 1 / (float(m) * a)
            T = delta * Q * sigma.transpose(2,1)
        return T

    def batch_trace(self, input_matrix, n, bs):
        a = torch.eye(n).to(self.device).unsqueeze(0).repeat(bs, 1, 1)
        b = a * input_matrix
        return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)

    def cos_batch(self, x, y, tau):
        bs = x.size(0)
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)
        cos_dis = torch.exp(- cos_dis / tau).transpose(1, 2)

        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        return torch.nn.functional.relu(res.transpose(2, 1))

    def calc_mlm_loss(self, x_batch, s_batch, s_batch_dim2,
                        seq_time_batch, mask_mult, mask_final, notes_batch, notes_len_batch, vs_code_index_batch):
        x_seq, y_seq, pred_mask_seq = self.mlm_preprocesser.mask_visit_out(vs_code_index_batch,self.device,len(self.code2id) + 2)
        x_text, y_text, pred_mask_text = self.mlm_preprocesser.mask_text_out(notes_batch,self.device)


        sequence_embedding = torch.sum(self.ehrcode_embed(x_seq),dim=2)
        sequence_contexts, sequence_embedding = self.visitEncoder_q(sequence_embedding,
                                                                                seq_time_batch, mask_mult,
                                                                                mask_final, s_batch,
                                                                                self.device)
        sequence_mask = length_to_mask(s_batch, self.max_visit_len).unsqueeze(1)

        text_mask = length_to_mask(notes_len_batch, self.max_notes_len).unsqueeze(1)
        text_contexts, text_embedding = self.textEncoder_q("calc_only_text", x_text, text_mask)

        sequence_contexts = sequence_contexts + \
                            self.token_type_embeddings(torch.zeros_like(sequence_mask.permute(0,2,1).squeeze(-1)).to(self.device).long())
        text_contexts = text_contexts + \
                            self.token_type_embeddings(torch.ones_like(text_mask.permute(0,2,1).squeeze(-1)).to(self.device).long())

        all_contexts = torch.cat([sequence_contexts, text_contexts], dim=1)
        all_mask = torch.cat([sequence_mask.int(), text_mask.int()], dim=-1).int()

        result_contexts = self.fusingEncoder("calc_text_and_visit",all_contexts,all_mask)
        sequence_contexts, text_contexts = torch.split(result_contexts,[sequence_contexts.shape[1],text_contexts.shape[1]],dim=1)

        y_seq = y_seq[pred_mask_seq].long().view(-1, len(self.code2id) + 2)
        masked_tensor_sequence = sequence_contexts[pred_mask_seq].view(-1, self.hita_input_size)
        masked_tensor_text = text_contexts[pred_mask_text.unsqueeze(-1).expand_as(text_contexts)].view(-1, self.text_embed)

        scores_seq = self.mlm_seqproj(masked_tensor_sequence).view(-1, len(self.code2id) + 2)
        loss_seq = F.binary_cross_entropy_with_logits(scores_seq, y_seq.float(), reduction='mean')

        scores_text = self.mlm_textproj(masked_tensor_text).view(-1, len(self.vocab.src))
        loss_text = F.cross_entropy(scores_text, y_text, reduction='mean')

        return loss_seq, loss_text

    def memoryUpdate(self, sequence_embedding_train_all,text_embedding_train_all):
        self.patient_visits[:, :] = sequence_embedding_train_all.T
        self.patient_texts[:, :] = text_embedding_train_all.T

    def get_similar_patient(self,sequence_embedding):
        patient_inner = torch.einsum('nc,ck->nk', [sequence_embedding, self.patient_visits.clone().detach()])
        patient_attn_pos = F.softmax(patient_inner, dim=-1)
        patient_attn_neg = -F.softmax(-patient_inner, dim=-1)
        patient_emb_pos = torch.mm(patient_attn_pos,self.patient_texts.T.clone().detach())
        patient_emb_neg = torch.mm(patient_attn_neg, self.patient_texts.T.clone().detach())
        return patient_emb_pos,patient_emb_neg

    def calc_logit(self, x_batch, s_batch, s_batch_dim2,
                   seq_time_batch, mask_mult, mask_final, notes_batch, notes_len_batch):
        sequence_embedding = torch.matmul(x_batch, self.ehrcode_embed.weight)
        sequence_contexts, sequence_embedding = self.visitEncoder_q(sequence_embedding,
                                                                                seq_time_batch, mask_mult,
                                                                                mask_final, s_batch,
                                                                                self.device)
        sequence_mask = length_to_mask(s_batch, self.max_visit_len).unsqueeze(1)

        text_mask = length_to_mask(notes_len_batch, self.max_notes_len).unsqueeze(1)
        text_contexts, text_embedding = self.textEncoder_q("calc_only_text", notes_batch, text_mask)

        sequence_contexts = sequence_contexts + \
                            self.token_type_embeddings(torch.zeros_like(sequence_mask.permute(0,2,1).squeeze(-1)).to(self.device).long())
        text_contexts = text_contexts + \
                            self.token_type_embeddings(torch.ones_like(text_mask.permute(0,2,1).squeeze(-1)).to(self.device).long())

        all_contexts = torch.cat([sequence_contexts, text_contexts], dim=1)
        all_mask = torch.cat([sequence_mask.int(), text_mask.int()], dim=-1).int()

        result_contexts = self.fusingEncoder("calc_text_and_visit",all_contexts,all_mask)
        sequence_contexts, text_contexts = torch.split(result_contexts,[sequence_contexts.shape[1],text_contexts.shape[1]],dim=1)
        sequence_embedding = sequence_contexts * mask_final
        sequence_embedding = sequence_embedding.sum(1, keepdim=False)
        text_embedding = text_contexts[:,0,:]

        if "no_patient_memory" not in self.ablation:
            patient_emb_pos, patient_emb_neg = self.get_similar_patient(sequence_embedding)
            all_embedding = torch.cat((sequence_embedding, text_embedding, patient_emb_pos, patient_emb_neg), dim=1)
        else:
            all_embedding = torch.cat((sequence_embedding, text_embedding), dim=1)

        patient_embedding_final = self.dropout(all_embedding)
        logits = self.classifier_fc(patient_embedding_final)
        return logits, sequence_embedding, text_embedding

    def forward(self, mode, *input):
        if mode == 'calc_cl_loss':
            return self.calc_cl_loss(*input)
        elif mode == 'calc_mlm_loss':
            return self.calc_mlm_loss(*input)
        elif mode == 'calc_logit':
            return self.calc_logit(*input)




