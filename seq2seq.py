import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch.utils.data as Data
import numpy
import math
import random
def main():
    pth='parallel/yelp_pos2neg.txt'
    BATCH_SIZE=64
    MAX_LENGTH=20
    print('Data Preprocessing...')
    data_iter=pre_processed(pth, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    print('Start Training...')
    train(data_iter,lr=0.00003, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, vocab=None)
    print('Finished')
    return

def pre_processed(pth, batch_size=64, max_length=20, simple_vocab=False):
    f=open(pth, 'r')
    lines=f.readlines()
    f.close()
    lines_src=[]
    lines_tgt=[]
    for line in lines:
        lines_src.append(line.split('\t')[0])
        lines_tgt.append(line.split('\t')[1].strip('\n'))
    tensors_src=lines2tensors(lines_src, max_length)#[:,1:]
    tensors_tgt=lines2tensors(lines_tgt, max_length)
    
    vocab={0:0}
    words_number=1
    simple_vocab_tensors_src=[]
    simple_vocab_tensors_tgt=[]
    for tensor_src in tensors_src:
        simple_vocab_tensor_src=[]
        for w in tensor_src:
            if int(w) not in vocab:
                vocab[int(w)]=words_number
                words_number+=1
            simple_vocab_tensor_src.append(vocab[int(w)])
        simple_vocab_tensors_src.append(torch.tensor(simple_vocab_tensor_src).view(1,-1))
    for tensor_tgt in tensors_tgt:
        simple_vocab_tensor_tgt=[]
        for w in tensor_tgt:
            if int(w) not in vocab:
                vocab[int(w)]=words_number
                words_number+=1
            simple_vocab_tensor_tgt.append(vocab[int(w)])
        simple_vocab_tensors_tgt.append(torch.tensor(simple_vocab_tensor_tgt).view(1,-1))
    if simple_vocab:
        dataset=Data.TensorDataset(torch.cat(simple_vocab_tensors_src, 0), torch.cat(simple_vocab_tensors_tgt, 0)) 
    else:
        dataset = Data.TensorDataset(tensors_src, tensors_tgt)
    data_iter=torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, 
                            collate_fn=None, pin_memory=False, 
                            drop_last=False, timeout=0, 
                            worker_init_fn=None, 
                            multiprocessing_context=None)
    if simple_vocab:
        return data_iter, vocab
    else:
        return data_iter

def batch_len_sort(src, tgt, pad=0):
    new_src=[]
    new_tgt=[]
    lengths=[]
    for line in src:
        length=0
        for w in line:
            if w !=torch.tensor(0):
                length+=1
            else: 
                break
        lengths.append(length)
    lengths=torch.tensor(lengths)    
    for _ in range(src.size(0)):
        pointer=lengths.argmax(-1)
        new_src.append(src[pointer].unsqueeze(0).clone())
        new_tgt.append(tgt[pointer].unsqueeze(0).clone())
        lengths[pointer]=-1
    return torch.cat(new_src, 0), torch.cat(new_tgt, 0)            
                                
def lines2tensors(lines, max_length):
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    tensors=[]
    input_ids_list=[]
    for line in lines:
        sen=tokenizer.tokenize(line.strip('\n'))
        input_ids_list.append(tokenizer.encode(sen))
    for input_ids in input_ids_list:
        if len(input_ids)>max_length:
            input_ids=input_ids[:max_length]
        else:
            for _ in range(max_length-len(input_ids)):
                input_ids.append(0)
        tensors.append(torch.tensor(input_ids).unsqueeze(0))
    return torch.cat(tensors, 0)

def maskNLLLoss(source, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(source, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()

def mask_batch(batch):
    mask = binaryMatrix(batch)
    mask = torch.BoolTensor(mask)
    return mask

def binaryMatrix(l, PAD_token=0):
    m=[]
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def remove_pad(batch, pad=0):
    batch=batch.t()
    for i in range(batch.size(0)):
        flag=True
        line=batch[i]
        for w in line:
            if int(w)!=0:
                flag=False
        if flag:
            break
    return batch[:i,:].t()

class Transformer(nn.Module):
    def __init__(self, words_number=30521, layers=4, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super(Transformer,self).__init__()
        self.embedding = nn.Embedding(words_number, d_model)
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
        embedding_tgt=self.embedding(tgt)
        embedding_src=self.position_embedding(embedding_src)
        embedding_tgt=self.position_embedding(embedding_tgt)
        '''
        memory=self.encoder(embedding_src)
        output=self.decoder(embedding_tgt, memory)
        '''
        output=self.transformer(src=embedding_src, tgt=embedding_tgt, \
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



def train(data_iter, lr=0.0001, batch_size=64, max_length=20, clip=50, epoch_num=50, \
          eval_step=500, teacher_forcing=1.0, vocab=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Bert_model= BertModel.from_pretrained('bert-base-uncased')
    if vocab:
        transformer=Transformer(len(vocab))
    else:
        transformer=Transformer()
    #loss_f=nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

    #Bert_model=Bert_model.to(device)
    #Bert_model.eval()    
    transformer=transformer.to(device)
    transformer.train()    
    iter_count=0
    for epoch in range(epoch_num):
        print('--------- Epoch: ', epoch, ' ---------')
        for src, tgt in data_iter:
            src, tgt=batch_len_sort(src, tgt)
            src=remove_pad(src)
            tgt=remove_pad(tgt)
            max_length=tgt.size(-1)
            optimizer.zero_grad()
            iter_count+=1
            loss=0
            print_losses=[]
            n_Totals=0
            #with torch.no_grad():
                #embedding_src=Bert_model(src.to(device))[0].reshape(max_length, -1, 768)
            mask=mask_batch(tgt.t())
            mask=mask.to(device)
            for i in range(max_length-1):
                #with torch.no_grad():
                    #embedding_tgt=Bert_model(tgt.to(device)[:,:i+1])[0].reshape(i+1, -1, 768)
                if random.random()<teacher_forcing or i==0:
                    decoder_input=tgt[:,:i+1].t()
                else:
                    decoder_input=torch.cat((tgt[:,:i].t().to(device), words.unsqueeze(0).to(device)), 0)
                output=transformer(src.t().to(device), decoder_input.to(device))
                words=output.argmax(-1)
                mask_loss, n_Total=maskNLLLoss(output, tgt.t()[i+1].to(device), mask[i+1])
                
                #print('--------------')
                
                #mask_loss=loss_f(output, tgt.t()[i+1].to(device))
                #n_Total=output.size(0)

                n_Totals+=n_Total
                loss+=mask_loss
                print_losses.append(mask_loss.item() * n_Total)
            loss.backward()
            _ = nn.utils.clip_grad_norm_(transformer.parameters(), clip)
            optimizer.step()
            print('Training Loss:',round(float(sum(print_losses) / n_Totals), 3))
            if iter_count%eval_step==0:
                model_eval(src, transformer, max_length, device, vocab)
                transformer.train()
    torch.save(transformer.state_dict(), './parallel/transformers.pth')

def model_eval(src, transformer, max_length, device, simple_vocab=None):
    transformer.eval()
    tokenizer=BertTokenizer.from_pretrained('./bert-base-uncased')
    #Bert_model=BertModel.from_pretrained('bert-base-uncased').to(device)
    text=[]
    if simple_vocab:
        SOS=simple_vocab[101]
        translate_vocab={}
        for w in simple_vocab:
            translate_vocab[simple_vocab[w]]=w
    else:
        SOS=101
    input_words=torch.tensor([SOS for _ in range(src.size(0))]).view(1,-1).to(device)
    words=input_words
    for i in range(max_length-1):
        with torch.no_grad():
            #embedding_tgt=Bert_model(input_words.view(-1,i+1))[0].reshape(i+1, -1, 768).to(device)
            output=transformer(src.t().to(device), input_words.view(i+1,-1).to(device))
            words=output.argmax(-1).view(1,-1)
            input_words=torch.cat([input_words, words], 0)
            if simple_vocab:
                translated_words=[]
                for w in words.clone().view(-1):
                    translated_words.append(int(translate_vocab[int(w)]))
                text.append(torch.tensor(translated_words).clone().view(1,-1))
            else:
                text.append(words.clone())
    text=torch.cat(text, 0).t()
    if simple_vocab:
        new_src_text=[]
        for words in src:
            translated_words=[]
            for w in words.clone().view(-1):
                translated_words.append(int(translate_vocab[int(w)]))
            new_src_text.append(torch.tensor(translated_words).clone().view(1,-1))
        new_src_text=torch.cat(new_src_text, 0)
    for i in range(text.size(0)):
        print('-----------------------------')
        if simple_vocab:
            print(tokenizer.decode(new_src_text[i].cpu().numpy().tolist()))
        else:
            print(tokenizer.decode(src[i].cpu().numpy().tolist()))
        print(tokenizer.decode(text[i].cpu().numpy().tolist()))
    transformer.train()
    return

if __name__=='__main__':
    main()


