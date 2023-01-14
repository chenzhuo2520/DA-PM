import torch
import numpy as np

def pad_sequence_pos(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
    x = (np.zeros(maxlen) + pad_id).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class CmuMosiDataset(torch.utils.data.Dataset):
    def __init__(self, config, df, tokenizer, pos_vocab, auc, vis, df_all_id):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.pos_vocab = pos_vocab
        self.auc = auc
        self.vis = vis
        self.df_all_id = df_all_id

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        tokenizer_out = self.tokenizer(self.df.iloc[idx]['text'], return_attention_mask=True, return_tensors="pt")
        input_ids = tokenizer_out['input_ids']

        if (self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1]) > 0:
            words_attention_mask = torch.cat((tokenizer_out['attention_mask'][0],
                                                 torch.zeros((self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1]),
                                                             dtype=torch.int64)))

            input_ids = torch.cat((input_ids[0], torch.full((self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1],),
                                                               self.tokenizer.pad_token_id, dtype=torch.int64)))

        else:
            input_ids = input_ids[0, :self.config['MAX_TEXT_LENGTH']]
            words_attention_mask = tokenizer_out['attention_mask'][0, :self.config['MAX_TEXT_LENGTH']]
        pos = [self.pos_vocab.stoi.get(t, self.pos_vocab.unk_index) for t in self.df.iloc[idx]['text_pos'].split()]
        pos = pad_sequence_pos(pos, pad_id=0, maxlen=self.config['MAX_TEXT_LENGTH'], dtype='int64', padding='post', truncating='post')

        score = torch.tensor(self.df.iloc[idx]['score'], dtype=torch.float32)


        aa = self.df.iloc[idx]['id']
        index = self.df_all_id[self.df_all_id.id == aa].index.tolist()

        auc_embedding = torch.from_numpy(self.auc[index])#[375,5]
        auc_embedding = auc_embedding.to(torch.float32)

        vis_embedding = torch.from_numpy(self.vis[index])#[375,5]
        vis_embedding = vis_embedding.to(torch.float32)

        return auc_embedding, vis_embedding, input_ids, words_attention_mask, score, pos

class CmuMoseiDataset(torch.utils.data.Dataset):
    def __init__(self, config, df, tokenizer, pos_vocab, auc, vis, df_all_id):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.pos_vocab = pos_vocab
        self.auc = auc
        self.vis = vis
        self.df_all_id = df_all_id

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        tokenizer_out = self.tokenizer(self.df.iloc[idx]['text'], return_attention_mask=True, return_tensors="pt")
        input_ids = tokenizer_out['input_ids']

        if (self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1]) > 0:
            words_attention_mask = torch.cat((tokenizer_out['attention_mask'][0],
                                                 torch.zeros((self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1]),
                                                             dtype=torch.int64)))

            input_ids = torch.cat((input_ids[0], torch.full((self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1],),
                                                               self.tokenizer.pad_token_id, dtype=torch.int64)))

        else:
            input_ids = input_ids[0, :self.config['MAX_TEXT_LENGTH']]
            words_attention_mask = tokenizer_out['attention_mask'][0, :self.config['MAX_TEXT_LENGTH']]
        pos = [self.pos_vocab.stoi.get(t, self.pos_vocab.unk_index) for t in self.df.iloc[idx]['text_pos'].split()]
        pos = pad_sequence_pos(pos, pad_id=0, maxlen=self.config['MAX_TEXT_LENGTH'], dtype='int64', padding='post', truncating='post')

        score = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)

        aa = self.df.iloc[idx]['id']
        index = self.df_all_id[self.df_all_id.id == aa].index.tolist()
        index = index[0]
        auc_embedding = torch.from_numpy(self.auc[index])#[375,5]
        auc_embedding = auc_embedding.to(torch.float32)

        vis_embedding = torch.from_numpy(self.vis[index])#[375,5]
        vis_embedding = vis_embedding.to(torch.float32)

        return auc_embedding, vis_embedding, input_ids, words_attention_mask, score, pos

