import os
import random
import collections
import pickle

import torch
import numpy as np
import pandas as pd
import math


class DataLoader(object):
    def __init__(self, args, logging):
        self.model_input_data_dir = args.model_input_data_dir
        self.sequence_batch_size = args.sequence_batch_size
        self.pretrain_batch_size = args.pretrain_batch_size

        self.train_sequence_file_path = self.model_input_data_dir + str(args.seed) + "/" + "train.pkl"
        self.val_sequence_file_path = self.model_input_data_dir + str(args.seed) + "/" + "val.pkl"
        self.test_sequence_file_path = self.model_input_data_dir + str(args.seed) + "/" + "test.pkl"

        self.use_pretrain  = args.use_pretrain
        self.pretrain_embedding_path = self.model_input_data_dir + args.pretrain_embedding_path
        self.dict_path = self.model_input_data_dir + "code_note_dict.pkl"

        self.cuda_choice = args.cuda_choice
        self.embedding_weights = None

        self._load_input_data()

    def _load_input_data(self):
        with open (self.train_sequence_file_path,"rb") as fin:
            data_dict_train = pickle.load(fin)
        with open (self.val_sequence_file_path,"rb") as fin:
            data_dict_val = pickle.load(fin)
        with open (self.test_sequence_file_path,"rb") as fin:
            data_dict_test = pickle.load(fin)
        with open(self.dict_path, "rb") as fin:
            self.code2id = pickle.load(fin)
            self.vocab = pickle.load(fin)
        if self.use_pretrain == 1:
            self.embedding_weights = np.load(self.pretrain_embedding_path)

        self.max_visit_len = data_dict_train["sequence_all_array"][0].get_shape()[0]
        self.code_num = data_dict_train["sequence_all_array"][0].get_shape()[1]
        self.label_num = data_dict_train["label_array"].shape[1]
        self.max_notes_len = data_dict_train["notes"].shape[1]

        self.train_patient_num = data_dict_train["label_array"].shape[0]

        self.sequence_train_array = [data_dict_train["sequence_all_array"],data_dict_train["sequence_len_array"],
                                     data_dict_train["sequence_len_dim2_array"],data_dict_train["label_array"],data_dict_train["mask"],data_dict_train["mask_final"],data_dict_train["seq_time"]
                                     ,data_dict_train["notes"],data_dict_train["notes_len"],data_dict_train["vs_code_index"]]
        self.sequence_val_array = [data_dict_val["sequence_all_array"], data_dict_val["sequence_len_array"],
                                     data_dict_val["sequence_len_dim2_array"], data_dict_val["label_array"],data_dict_val["mask"],data_dict_val["mask_final"],data_dict_val["seq_time"]
                                   ,data_dict_val["notes"],data_dict_val["notes_len"],data_dict_val["vs_code_index"]]
        self.sequence_test_array = [data_dict_test["sequence_all_array"], data_dict_test["sequence_len_array"],
                                     data_dict_test["sequence_len_dim2_array"], data_dict_test["label_array"],data_dict_test["mask"],data_dict_test["mask_final"],data_dict_test["seq_time"]
                                    ,data_dict_test["notes"],data_dict_test["notes_len"],data_dict_test["vs_code_index"]]
        self.sequence_array_dict ={}
        self.sequence_array_dict["train"] = self.sequence_train_array
        self.sequence_array_dict["val"] =self.sequence_val_array
        self.sequence_array_dict["test"] = self.sequence_test_array


    def sequence_pretrain_iter(self,flag,args,shuffle = True):
        sequence_all_array = self.sequence_array_dict[flag]
        patient_num  = len(sequence_all_array[0])
        batch_num = math.ceil(patient_num / self.pretrain_batch_size)
        index_array = list(range(patient_num))
        if flag == "train" and shuffle:
            np.random.shuffle(index_array)
            sequence_all_array[0] = [sequence_all_array[0][index] for index in list(index_array)]
        for i in range(batch_num):
            indices = index_array[i * self.pretrain_batch_size: (i + 1) * self.pretrain_batch_size]
            x_batch_origin = sequence_all_array[0][i * self.pretrain_batch_size: (i + 1) * self.pretrain_batch_size]
            x_batch = np.array([sparse_matrix.toarray() for sparse_matrix in x_batch_origin])
            s_batch = sequence_all_array[1][indices]
            s_batch_dim2 = sequence_all_array[2][indices]
            y_batch = sequence_all_array[3][indices]
            batch_mask = sequence_all_array[4][indices]
            batch_mask_final = sequence_all_array[5][indices]
            seq_time_batch = sequence_all_array[6][indices]
            notes_batch = sequence_all_array[7][indices]
            notes_len_batch = sequence_all_array[8][indices]
            vs_code_index_batch = sequence_all_array[9][indices]

            yield x_batch, s_batch, s_batch_dim2, y_batch, batch_mask, batch_mask_final, seq_time_batch, notes_batch, notes_len_batch, vs_code_index_batch

    def sequence_batch_iter(self,flag,args,shuffle = True):
        sequence_all_array = self.sequence_array_dict[flag]
        patient_num  = len(sequence_all_array[0])
        batch_num = math.ceil(patient_num / self.sequence_batch_size)
        index_array = list(range(patient_num))
        if flag == "train" and shuffle:
            np.random.shuffle(index_array)
            sequence_all_array[0] = [sequence_all_array[0][index] for index in list(index_array)]
        for i in range(batch_num):
            indices = index_array[i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
            x_batch_origin = sequence_all_array[0][i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
            x_batch = np.array([sparse_matrix.toarray() for sparse_matrix in x_batch_origin])
            s_batch = sequence_all_array[1][indices]
            s_batch_dim2 = sequence_all_array[2][indices]
            y_batch = sequence_all_array[3][indices]
            batch_mask = sequence_all_array[4][indices]
            batch_mask_final = sequence_all_array[5][indices]
            seq_time_batch = sequence_all_array[6][indices]
            notes_batch = sequence_all_array[7][indices]
            notes_len_batch = sequence_all_array[8][indices]
            yield x_batch, s_batch, s_batch_dim2, y_batch, batch_mask, batch_mask_final, seq_time_batch, notes_batch, notes_len_batch