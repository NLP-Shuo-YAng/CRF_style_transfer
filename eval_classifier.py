import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer

class TextMLP(nn.Module):
    def __init__(self, vocab_size, p=0.2, dim=768, use_attention=True, label_num=2):
        super(TextMLP, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dropout=nn.Dropout(p)
        self.use_attention=use_attention
        self.leaky_relu=nn.LeakyReLU(negative_slope=0.005, inplace=False)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.linear4 = nn.Linear(dim, dim)
        self.linear5 = nn.Linear(dim, dim)
        self.fc=nn.Linear(dim, label_num)
        
        if self.use_attention:
            self.attn = nn.MultiheadAttention(dim, num_heads=8)
    
    def forward(self, x):
        x = self.embedding(x) 
        if self.attn:
            attn_input=x.view(x.size(1), x.size(0), -1)
            attn_out, attn_weight=self.attn(attn_input, attn_input, attn_input)
            attn_out=attn_out.view(x.size(0), x.size(1), -1)
            x=x*attn_out
        x=self.dropout(x)
        x, _=torch.max(x, 1)
        hidden=self.leaky_relu(self.linear1(x))
        hidden=self.leaky_relu(self.linear2(x))
        hidden=F.tanh(self.linear3(x))
        hidden=self.leaky_relu(self.linear4(x))
        hidden=F.tanh(self.linear5(x))
        logits = self.fc(hidden)
        x=self.dropout(logits)
        return logits

