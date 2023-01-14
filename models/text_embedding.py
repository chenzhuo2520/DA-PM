from torch import nn
from transformers import RobertaModel, RobertaConfig, BertConfig, BertModel, XLNetConfig, XLNetModel

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self):
        super(LanguageEmbeddingLayer, self).__init__()
        bertconfig = RobertaConfig.from_pretrained("roberta_base/")
        self.bertmodel = RobertaModel.from_pretrained("roberta_base/", config=bertconfig)

    def forward(self, bert_sent, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask)
        bert_output = bert_output[0]
        return bert_output   # return head (sequence representation)
#
# class LanguageEmbeddingLayer(nn.Module):
#     """Embed input text with "glove" or "Bert"
#     """
#     def __init__(self):
#         super(LanguageEmbeddingLayer, self).__init__()
#         bertconfig = XLNetConfig.from_pretrained("xlnet-base-cased/")
#         self.bertmodel = XLNetModel.from_pretrained("xlnet-base-cased/", config=bertconfig)
#
#     def forward(self, bert_sent, bert_sent_mask):
#         bert_output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask)
#         bert_output = bert_output.last_hidden_state
#         return bert_output

# class LanguageEmbeddingLayer(nn.Module):
#     """Embed input text with "glove" or "Bert"
#     """
#     def __init__(self):
#         super(LanguageEmbeddingLayer, self).__init__()
#         bertconfig = BertConfig.from_pretrained("bert_base_uncased/")
#         self.bertmodel = BertModel.from_pretrained("bert_base_uncased/", config=bertconfig)
#
#     def forward(self, bert_sent, bert_sent_mask):
#         bert_output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask)
#         bert_output = bert_output[0]
#         return bert_output   # return head (sequence representation)