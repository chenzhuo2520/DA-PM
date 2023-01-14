import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer, XLNetTokenizer
from .cmu_mosi import CmuMosiDataset, CmuMoseiDataset
import pickle
import numpy as np



def get_my_roberta_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained("roberta_base/")
    # tokenizer = BertTokenizer.from_pretrained('xlnet-base-cased')
    # tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')

    return tokenizer


def fetch_mosi_datasets(config, pos_vocab):
    train_df = pd.read_csv('data/MOSI/mosi_train_df_pos_id.csv')
    val_df = pd.read_csv('data/MOSI/mosi_val_df_pos_id.csv')
    test_df = pd.read_csv('data/MOSI/mosi_test_df_pos_id.csv')

    tokenizer = get_my_roberta_tokenizer()
    pickle_filename = 'data/MOSI/unaligned_50.pkl'
    with open(pickle_filename, 'rb') as f:
        d = pickle.load(f)
    train_split_noalign = d['train']
    dev_split_noalign = d['valid']
    test_split_noalign = d['test']
    vis = np.concatenate((train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']),
                       axis=0)
    auc = np.concatenate((train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']),
                         axis=0)
    all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)
    all_id_list = list(map(lambda x: x.replace('$', ''), all_id.tolist()))
    df_all_id = pd.DataFrame(all_id_list, columns=['id'])
    train_dataset = CmuMosiDataset(config, train_df, tokenizer, pos_vocab, auc, vis,  df_all_id)
    val_dataset = CmuMosiDataset(config, val_df, tokenizer, pos_vocab, auc, vis, df_all_id)
    test_dataset = CmuMosiDataset(config, test_df, tokenizer, pos_vocab, auc, vis, df_all_id)

    return train_dataset, val_dataset, test_dataset



def fetch_mosei_datasets(config, pos_vocab):
    df_all = pd.read_csv('data/MOSEI/mosei_val_df_pos_id.csv')
    train_df = df_all[df_all['mode'] == 'train']
    val_df = df_all[df_all['mode'] == 'valid']
    test_df = df_all[df_all['mode'] == 'test']

    tokenizer = get_my_roberta_tokenizer()
    pickle_filename = 'data/MOSEI/unaligned_50.pkl'
    with open(pickle_filename, 'rb') as f:
        d = pickle.load(f)
    train_split_noalign = d['train']
    dev_split_noalign = d['valid']
    test_split_noalign = d['test']
    vis = np.concatenate((train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']),
                       axis=0)
    auc = np.concatenate((train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']),
                         axis=0)
    all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)
    all_id_list = list(map(lambda x: x.replace('$', ''), all_id.tolist()))
    df_all_id = pd.DataFrame(all_id_list, columns=['id'])
    train_dataset = CmuMoseiDataset(config, train_df, tokenizer, pos_vocab, auc, vis, df_all_id)
    val_dataset = CmuMoseiDataset(config, val_df, tokenizer, pos_vocab, auc, vis, df_all_id)
    test_dataset = CmuMoseiDataset(config, test_df, tokenizer, pos_vocab, auc, vis, df_all_id)

    return train_dataset, val_dataset, test_dataset

