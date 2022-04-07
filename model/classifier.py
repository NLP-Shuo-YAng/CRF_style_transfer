import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as Data
import sys
sys.path.append('..')
from eval_classifier import TextMLP
class TextCNN(nn.Module):
    def __init__(self, vocab_size, p=0.1, use_BN=False, use_attention=False, dim=512, label_num=2):
        super(TextCNN, self).__init__()
        filter_num = 64 
        filter_sizes='2,3,4'
        self.use_BN=use_BN
        filter_sizes = [int(fsz) for fsz in filter_sizes.split(',')]
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dropout=nn.Dropout(p)
        
        self.leaky_relu=nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.convs_bn = nn.BatchNorm2d(num_features=filter_num)
        self.linear = nn.Linear(len(filter_sizes)*filter_num, len(filter_sizes)*filter_num)
        self.fc=nn.Linear(len(filter_sizes)*filter_num, label_num)
        
        if use_attention:
            self.attn = nn.MultiheadAttention(dim, num_heads=8)
        else:
            self.attn=False
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (fsz, dim)) for fsz in filter_sizes])
    def forward(self, x):
        x = self.embedding(x) 
        if self.attn:
            attn_input=x.view(x.size(1), x.size(0), -1)
            attn_out, attn_weight=self.attn(attn_input, attn_input, attn_input)
            attn_out=attn_out.view(x.size(0), x.size(1), -1)
            x=attn_out*x
        x = x.unsqueeze(1)
        x=self.dropout(x)
        if self.use_BN:
            x = [self.convs_bn(self.leaky_relu(conv(x))).squeeze(3) for conv in self.convs] 
        else:
            x = [self.leaky_relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
        x = torch.cat(x, 1)
        logits = self.leaky_relu(self.linear(x))
        logits=self.fc(x)
        return logits


def pretrain_classifier(src, tgt, src_test, tgt_test, vocab_size, show_step=20, epoch_num=5, eval_step=500, batch_size=64):
    # src, tgt is tensor list
    device=torch.device('cuda')
    model=TextMLP(vocab_size).to(device)
    model.train()
    loss_f=nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    data=torch.cat([src, tgt], 0)
    data_test=torch.cat([src_test, tgt_test], 0)
    labels=[]
    labels_test=[]
    for _ in range(src.size(0)):
        labels.append(0)
    for _ in range(tgt.size(0)):
        labels.append(1)
    for _ in range(src_test.size(0)):
        labels_test.append(0)
    for _ in range(tgt_test.size(0)):
        labels_test.append(1)
    labels=torch.tensor(labels).unsqueeze(0).t()
    labels_test=torch.tensor(labels_test).unsqueeze(0).t()
    dataset = Data.TensorDataset(data, labels)
    dataset_test = Data.TensorDataset(data_test, labels_test)
    data_iter=torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=4) 
    data_iter_test=torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                    shuffle=True, num_workers=4) 
    for epoch in range(epoch_num):
        print('----------- Epoch: ',epoch,'-------------')
        count=0
        print_loss=0
        for src, tgt in data_iter:
            count+=1
            optimizer.zero_grad()
            output=model(src.to(device))
            loss=loss_f(output, tgt.view(-1).to(device))
            loss.backward()
            print_loss+=float(loss)
            _=nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()
            loss=0
            if count%show_step==0:
                print('Loss: ', round(print_loss/show_step, 3))
                print_loss=0
            if count%eval_step==0:
                classifier_eval(model, data_iter_test)
    
    return model
    
def classifier_eval(model, data_iter):
    device=torch.device('cuda')
    acc=0
    count=0
    model.eval()
    for src, tgt in data_iter:
        output=model(src.to(device)).argmax(-1).view(-1)
        tgt=tgt.view(-1)
        for i in range(output.size(-1)):
            if output[i]==tgt[i]:
                acc+=1
            count+=1
    print('------------- Accuracy: ', round(acc/count*100, 3), ' %')
    model.train()
    return
