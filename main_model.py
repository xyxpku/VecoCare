import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import json
import random
import logging
import argparse
from time import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model.Model import Model
from utility.log_helper import *
from utility.parser_model import *
from utility.loader_model import DataLoader
from utility.metrics import get_metrics

def evaluate(args,device,model,dataLoader,result_path, flag,epoch):
    model.eval()
    y_true_evaluate = []
    y_pred_evaluate = []
    with torch.no_grad():
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch,
        batch_mask, batch_mask_final, seq_time_batch, notes_batch, notes_len_batch) in enumerate(dataLoader.sequence_batch_iter(flag=flag,args=args)):
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            seq_time_batch = torch.Tensor(seq_time_batch).to(device).unsqueeze(2) / args.hita_time_scale
            mask_mult = torch.BoolTensor(1 - batch_mask).to(device).unsqueeze(2)
            mask_final = torch.Tensor(batch_mask_final).to(device).unsqueeze(2)
            notes_batch = torch.LongTensor(notes_batch).to(device)
            notes_len_batch = torch.LongTensor(notes_len_batch).to(device)

            logits, sequence_embedding, text_embedding = model('calc_logit', x_batch, s_batch, s_batch_dim2,
                                                      seq_time_batch, mask_mult, mask_final, notes_batch, notes_len_batch)
            real_logits = torch.sigmoid(logits)

            logits_cpu = real_logits.data.cpu().numpy()
            labels_cpu = y_batch.data.cpu().numpy()
            y_true_evaluate.append(labels_cpu)
            y_pred_evaluate.append(logits_cpu)

    y_true_evaluate = np.vstack(y_true_evaluate)
    y_pred_evaluate = np.vstack(y_pred_evaluate)
    prauc_dict,auc_dict = get_metrics(y_pred_evaluate, y_true_evaluate, flag, epoch, result_path)

    return prauc_dict["micro"],prauc_dict["macro"],auc_dict["micro"],auc_dict["macro"]


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    dataLoader = DataLoader(args, logging)
    model = Model(args, dataLoader.max_visit_len, dataLoader.code_num, dataLoader.label_num,
                        args.use_pretrain, dataLoader.embedding_weights, dataLoader.code2id, dataLoader.vocab,
                        dataLoader.max_notes_len, dataLoader.train_patient_num)

    model.to(device)
    logging.info(model)
    with open(args.save_dir+"params.json",mode = "w") as f:
        json.dump(args.__dict__,f,indent=4)

    grouped_parameters_t = [
        {'params': [p for n, p in model.named_parameters() if ('visitEncoder' in n)],
          'lr': args.visit_encoder_lr, 'weight_decay': args.train_weight_decay},
        {'params': [p for n, p in model.named_parameters() if ('textEncoder' in n)],
          'lr': args.text_encoder_lr, 'weight_decay': args.train_weight_decay},
        {'params': [p for n, p in model.named_parameters() if (not (('visitEncoder' in n) or ('textEncoder' in n)))],
         'lr': args.base_lr, 'weight_decay': args.train_weight_decay}
    ]

    optimizer_cl = optim.Adam(model.parameters(), lr= args.pretrain_lr, weight_decay=args.pretrain_weight_decay)
    optimizer_mlm = optim.Adam(model.parameters(), lr= args.pretrain_lr, weight_decay=args.pretrain_weight_decay)
    optimizer_seq = optim.Adam(grouped_parameters_t)
    model_path =os.path.join(args.save_dir, 'model.pt')
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    result_path = args.save_dir + "metrics.txt"
    best_dev_epoch = 0
    best_dev_auc, final_micro_prauc, final_macro_prauc = 0.0, 0.0, 0.0
    final_micro_auroc, final_macro_auroc = 0.0, 0.0

    for epoch in range(1, args.n_epoch_pretrain_cl + 1):
        time0 = time()
        model.train()
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch,
        batch_mask, batch_mask_final, seq_time_batch, notes_batch, notes_len_batch, vs_code_index_batch) in enumerate(dataLoader.sequence_pretrain_iter(flag="train", args=args)):
            optimizer_cl.zero_grad()
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            seq_time_batch = torch.Tensor(seq_time_batch).to(device).unsqueeze(2) / args.hita_time_scale
            mask_mult = torch.BoolTensor(1 - batch_mask).to(device).unsqueeze(2)
            mask_final = torch.Tensor(batch_mask_final).to(device).unsqueeze(2)
            notes_batch = torch.LongTensor(notes_batch).to(device)
            notes_len_batch = torch.LongTensor(notes_len_batch).to(device)
            loss_gw_v2t, loss_gw_t2v = model('calc_cl_loss', x_batch, s_batch, s_batch_dim2,
                                                      seq_time_batch, mask_mult, mask_final, notes_batch, notes_len_batch)
            cl_loss = loss_gw_v2t + loss_gw_t2v
            cl_loss.backward()
            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer_cl.step()
        logging.info(
                'Pre Training For CL: Epoch {:04d} | Total Time {:.1f}s'.format(epoch,
                                                                                  time() - time0))
    mlm_weight = {
        "visit_loss": 1 / 2.0,
        "text_loss": 1 / 2.0,
    }
    init_loss_visit = None
    init_loss_text = None
    mlm_beta = 1
    mlm_weight_learn_rate = 0.025

    for epoch in range(1, args.n_epoch_pretrain_mlm + 1):
        time0 = time()
        model.train()
        mlm_total_loss = 0.0
        mlm_visit_loss = 0.0
        mlm_text_loss = 0.0
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch,
        batch_mask, batch_mask_final, seq_time_batch, notes_batch, notes_len_batch, vs_code_index_batch) in enumerate(dataLoader.sequence_pretrain_iter(flag="train", args=args)):
            optimizer_mlm.zero_grad()
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)
            seq_time_batch = torch.Tensor(seq_time_batch).to(device).unsqueeze(2) / args.hita_time_scale
            mask_mult = torch.BoolTensor(1 - batch_mask).to(device).unsqueeze(2)
            mask_final = torch.Tensor(batch_mask_final).to(device).unsqueeze(2)
            notes_batch = torch.LongTensor(notes_batch).to(device)
            notes_len_batch = torch.LongTensor(notes_len_batch).to(device)
            vs_code_index_batch = torch.LongTensor(vs_code_index_batch).to(device)
            loss_seq, loss_text = model('calc_mlm_loss', x_batch, s_batch, s_batch_dim2,
                                                      seq_time_batch, mask_mult, mask_final, notes_batch, notes_len_batch, vs_code_index_batch)
            if init_loss_visit == None:
                init_loss_visit = loss_seq.item()
            if init_loss_text == None:
                init_loss_text = loss_text.item()
            relative_visit = loss_seq.item() / init_loss_visit
            relative_text = loss_text.item() / init_loss_text
            inv_rate_visit = relative_visit ** mlm_beta
            inv_rate_text = relative_text ** mlm_beta
            mlm_weight["visit_loss"] = mlm_weight["visit_loss"] - mlm_weight_learn_rate * (
                        mlm_weight["visit_loss"] - inv_rate_visit / (inv_rate_text + inv_rate_visit))
            mlm_weight["text_loss"] = 1 - mlm_weight["visit_loss"]

            mlm_loss = mlm_weight["visit_loss"]*loss_seq + mlm_weight["text_loss"]*loss_text
            mlm_loss.backward()
            mlm_total_loss += mlm_loss.item()
            mlm_visit_loss += loss_seq.item()
            mlm_text_loss += loss_text.item()
            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer_mlm.step()
        logging.info(
                'Pre Training For MLM: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.4f} | Visit Loss {:.4f}  Visit Loss Weight {:.4f} | Text Loss {:.4f}  Text Loss Weight {:.4f}'
                    .format(epoch,time() - time0, mlm_total_loss, mlm_visit_loss, mlm_weight["visit_loss"], mlm_text_loss, mlm_weight["text_loss"]))

    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()
        sequence_total_loss = 0.0
        y_true_train = []
        y_pred_train = []
        sequence_embedding_train = []
        text_embedding_train = []
        for step, (
        x_batch, s_batch, s_batch_dim2, y_batch,
        batch_mask, batch_mask_final, seq_time_batch, notes_batch, notes_len_batch) in enumerate(dataLoader.sequence_batch_iter(flag="train", args=args)):
            optimizer_seq.zero_grad()
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            seq_time_batch = torch.Tensor(seq_time_batch).to(device).unsqueeze(2) / args.hita_time_scale
            mask_mult = torch.BoolTensor(1 - batch_mask).to(device).unsqueeze(2)
            mask_final = torch.Tensor(batch_mask_final).to(device).unsqueeze(2)
            notes_batch = torch.LongTensor(notes_batch).to(device)
            notes_len_batch = torch.LongTensor(notes_len_batch).to(device)

            logits, sequence_embedding, text_embedding = model('calc_logit', x_batch, s_batch, s_batch_dim2,
                                                      seq_time_batch, mask_mult, mask_final, notes_batch, notes_len_batch)
            sequence_embedding_train.append(sequence_embedding.clone().detach())
            text_embedding_train.append(text_embedding.clone().detach())

            real_logits = torch.sigmoid(logits)
            sequence_loss = loss_func(logits,y_batch.float())
            sequence_loss.backward()
            sequence_total_loss += sequence_loss.item()

            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer_seq.step()

            logits_cpu = real_logits.data.cpu().numpy()
            labels_cpu = y_batch.data.cpu().numpy()
            y_true_train.append(labels_cpu)
            y_pred_train.append(logits_cpu)

        sequence_embedding_train_all = torch.cat(sequence_embedding_train,dim=0)
        text_embedding_train_all = torch.cat(text_embedding_train, dim=0)

        if "no_patient_memory" not in args.ablation:
                model.memoryUpdate(sequence_embedding_train_all, text_embedding_train_all)

        y_true_train = np.vstack(y_true_train)
        y_pred_train = np.vstack(y_pred_train)
        get_metrics(y_pred_train, y_true_train, "train", epoch, result_path)
        logging.info(
                'Training: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch,time() - time0,sequence_total_loss))


        time1 = time()
        dev_micro_prauc,dev_macro_prauc,dev_micro_auroc,dev_macro_auroc = evaluate(args, device, model, dataLoader, result_path,  "val",
                                                 epoch)
        logging.info(
            'Val Evaluation: Epoch {:04d} | Total Time {:.1f}s'.format(
                epoch, time() - time1))

        time1 = time()
        test_micro_prauc,test_macro_prauc,test_micro_auroc,test_macro_auroc = evaluate(args, device, model, dataLoader, result_path, "test",
                                                   epoch)
        logging.info(
            'Test Evaluation: Epoch {:04d} | Total Time {:.1f}s'.format(
                epoch, time() - time1))

        if dev_macro_prauc >= best_dev_auc:
            best_dev_auc = dev_macro_prauc
            best_dev_epoch = epoch
            final_micro_prauc = test_micro_prauc
            final_micro_auroc = test_micro_auroc
            final_macro_auroc = test_macro_auroc
            final_macro_prauc = test_macro_prauc
            torch.save(model.state_dict(), model_path)
            print(f'model saved to {model_path}')

        logging.info("Epoch: {}".format(epoch))
        logging.info('best test micro prauc: {:.4f}'.format(final_micro_prauc))
        logging.info('best test macro prauc: {:.4f}'.format(final_macro_prauc))
        logging.info('best test micro auroc: {:.4f}'.format(final_micro_auroc))
        logging.info('best test macro auroc: {:.4f}'.format(final_macro_auroc))

        if epoch > args.unfreeze_epoch and epoch - best_dev_epoch >= args.max_epochs_before_stop:
            break

    logging.info('best test auc: {:.4f} (at epoch {})'.format(final_macro_prauc, best_dev_epoch))
    logging.info('final test micro prauc: {:.4f}'.format(final_micro_prauc))
    logging.info('final test macro prauc: {:.4f}'.format(final_macro_prauc))
    logging.info('final test micro auroc: {:.4f}'.format(final_micro_auroc))
    logging.info('final test macro auroc: {:.4f}'.format(final_macro_auroc))
    logging.info('{:.4f},{:.4f},{:.4f},{:.4f}'.format(final_micro_prauc, final_macro_prauc, final_micro_auroc,
                                                      final_macro_auroc))


if __name__ == "__main__":
    args = parse_args()
    main(args)