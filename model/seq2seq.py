import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import math
import random
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, dropout=0.2):
        super(Bert, self).__init__()
        self.encoder=BertModel.from_pretrained('bert-base-uncased')
        self.Linear=nn.Linear(768, 7)
        self.drop=nn.Dropout(p=dropout)
    def forward(self, src):
        self.encoder.eval()
        with torch.no_grad():
            embedding=self.encoder(src)[0]
        embedding=self.drop(embedding)
        logits=self.Linear(embedding)
        return logits


class Transformer(nn.Module):
    def __init__(self, words_number=30521, layers=4, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
        super(Transformer,self).__init__()
        self.embedding = nn.Embedding(words_number, d_model)
        self.tgt_embedding=nn.Embedding(words_number, d_model)
        self.position_embedding=PositionEncoding(d_model)

        self.transformer=torch.nn.Transformer(d_model=d_model, nhead=8, num_encoder_layers=layers, \
            num_decoder_layers=layers, dim_feedforward=2048, dropout=dropout, activation='relu')

        self.output_layer = nn.Linear(d_model, words_number)
    def generate_mask(self, sz):
        mask=torch.triu(torch.ones(sz,sz),1)
        mask=mask.masked_fill(mask==1,float('-inf'))
    def make_len_mask(self, inp):
        return (inp==0).transpose(0,1)
    def forward(self, src, tgt):
        tgt_mask=self.generate_mask(len(tgt))
        src_pad_mask=self.make_len_mask(src)
        tgt_pad_mask=self.make_len_mask(tgt)
        embedding_src=self.embedding(src)
        embedding_tgt=self.tgt_embedding(tgt)
        embedding_src=self.position_embedding(embedding_src)
        embedding_tgt=self.position_embedding(embedding_tgt)
        output=self.transformer(src=embedding_src, tgt=embedding_tgt,
               tgt_mask=tgt_mask,
               src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        output, _ = torch.max(output, 0)
        output = self.output_layer(output)
        return F.softmax(output, dim = -1)

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len=100):
        super(PositionEncoding, self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len, d_model)
        position=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2]=torch.sin(position*div_term)
        pe[:, 1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x=x+self.pe[:x.size(0),:]
        return self.dropout(x)
