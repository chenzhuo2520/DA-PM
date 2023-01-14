import torch
from .laa import GRU_aucdio, GRU_visual
from .text_embedding import LanguageEmbeddingLayer
from torch import nn
from .encoders import My_ClassificationHead, Full_classificationHead
import torch.nn.functional as F


class My_Model(nn.Module):
    def __init__(self, device='cpu'):
        """Construct MultiMoldal InfoMax model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.device = device

        self.text_enc = LanguageEmbeddingLayer()
        self.laa_a = GRU_aucdio(audio_output_size=768)
        self.laa_v = GRU_visual(audio_output_size=768)

        self.my_classifier = My_ClassificationHead()
        self.agg_1 = torch.nn.Linear(in_features=768, out_features=768, bias=True)
        self.agg_2 = torch.nn.Linear(in_features=768, out_features=1, bias=True)
        self.bert_drop = nn.Dropout(0.2)

        self.affine1 = nn.Parameter(torch.Tensor(768, 768))
        self.alpha_audio_1 = torch.nn.Linear(80, 80, bias=True)
        self.alpha_audio_2 = torch.nn.Linear(80, 80, bias=True)

        self.affine2 = nn.Parameter(torch.Tensor(768, 768))
        self.alpha_vis_1 = torch.nn.Linear(80, 80, bias=True)
        self.alpha_vis_2 = torch.nn.Linear(80, 80, bias=True)

        self.v_text = torch.nn.Linear(768, 768, bias=True)
        self.a_text = torch.nn.Linear(768, 768, bias=True)

        # mosi 38, mosei 50
        self.pos_emb = nn.Embedding(38, 768, padding_idx=0)
        self.full_linear = Full_classificationHead()


    def forward(self, audio_embed, vis_embed, input_ids, attention_mask, pos):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        text_embedding = self.text_enc(input_ids, attention_mask)
        audio_embed = audio_embed.squeeze(dim=1)
        vis_embed = vis_embed.squeeze(dim=1)
        audio_embedding = self.laa_a(audio_embed)
        vis_embedding = self.laa_v(vis_embed)
        pos_embedding = self.pos_emb(pos)
        audio_text_embedding = text_embedding.clone()
        vis_text_embedding = text_embedding.clone()

        # 音频映射
        hat = text_embedding.clone()
        A1 = F.softmax(
            torch.bmm(torch.matmul(audio_embedding, self.affine1), torch.transpose(text_embedding, 1, 2)),
            dim=-1)
        acu_text = torch.bmm(A1, text_embedding)  # [32,80,768]
        alpha_audio_1 = self.alpha_audio_1(audio_embedding[:, :, 0])
        alpha_audio_2 = torch.tanh(alpha_audio_1)
        alpha_audio = self.alpha_audio_2(alpha_audio_2)
        for i in range(text_embedding.shape[1]):
            alpha_audio_w = alpha_audio[:, i].unsqueeze(dim=-1)
            hat[:, i, :] = audio_text_embedding[:, i, :] + alpha_audio_w * acu_text[:, i, :]
        # 音频映射

        # 视频映射
        hvt = text_embedding.clone()
        A2 = F.softmax(
            torch.bmm(torch.matmul(vis_embedding, self.affine2), torch.transpose(text_embedding, 1, 2)),
            dim=-1)
        vis_text = torch.bmm(A2, text_embedding)  # [32,80,768]
        alpha_vis_1 = self.alpha_vis_1(vis_embedding[:, :, 0])
        alpha_vis_2 = torch.tanh(alpha_vis_1)
        alpha_vis = self.alpha_vis_2(alpha_vis_2)
        for i in range(text_embedding.shape[1]):
            alpha_vis_w = alpha_vis[:, i].unsqueeze(dim=-1)
            hvt[:, i, :] = vis_text_embedding[:, i, :] + alpha_vis_w * vis_text[:, i, :]
        # 视频映射

        # pos+text
        text_embedding = (self.a_text(hat) + self.v_text(hvt)) / 2
        text_embedding = self.bert_drop(text_embedding)
        seq_weight = torch.sigmoid(self.agg_1(pos_embedding))
        alph_pos = torch.softmax(self.agg_2(seq_weight), 1)  # [32,80,1]
        text_pos = torch.bmm(alph_pos.transpose(1, 2), text_embedding)  # [32,1,768]
        text_pos = text_pos[:, 0, :]  # [32,768]
        logits = self.my_classifier(text_pos)
        # pos+text

        return logits
