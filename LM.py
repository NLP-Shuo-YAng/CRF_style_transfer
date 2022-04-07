import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.utils.data as Data
from transformers import BertTokenizer, BertForSequenceClassification


def pretrain_LM(tensors, eopch_num, save_pth, batch_size=64, show_step=50, eval_step=500):
    device=torch.device('cuda')
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    model=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=tokenizer.vocab_size)
    model.train()
    model=model.to(device)
    loss_f=nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    count=0
    print_loss=0
    tensors_backup=tensors.copy()
    for epoch in range(eopch_num):
        print('-----------','Epoch: ',epoch,'------------------')
        tensors_new=tensors_backup.copy()
        data_iter=mask_LM(tensors_new, batch_size=batch_size)
        for src, tgt in data_iter:
            count+=1
            optimizer.zero_grad()
            output=model(src.to(device))[0]
            loss=loss_f(output, tgt.to(device))
            loss.backward()
            optimizer.step()
            print_loss+=float(loss)
            if count%show_step==0:
                print('LM Training Loss: ', round(print_loss/show_step, 3))
                print_loss=0
            if count%eval_step==0:
                model.eval()
                out=model(src.to(device))[0].argmax(-1)
                print('src: ', tokenizer.decode(src[0].cpu().numpy().tolist()))
                print('tgt: ', tokenizer.decode([tgt[0].cpu().numpy().tolist()]))
                print('out: ', tokenizer.decode([out[0].cpu().numpy().tolist()]))
                model.train()
    model.eval()
    torch.save(model.state_dict(), save_pth)
    return

def data_processor(pth, max_len):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    f=open(pth)
    lines=f.readlines()
    f.close()
    tensors=[]
    for line in lines:
        tensor=tokenizer.encode(line)[:max_len]
        for _ in range(max_len-len(tensor)):
            tensor.append(0)
        tensors.append(tensor)
    return tensors # list

def mask_LM(tensors, batch_size=64):
    src=[]
    tgt=[]
    for tensor in tensors:
        length=0
        for i in tensor:
            if i !=0:
                length+=1
            else:
                break
        mask_position=random.randint(1, length-2)
        tgt.append(torch.tensor(tensor[mask_position]).unsqueeze(0))
        new_tensor=tensor.copy()
        new_tensor[mask_position]=103     # [mask]
        src.append(torch.tensor(new_tensor.copy()).unsqueeze(0))
    src=torch.cat(src, 0)
    tgt=torch.cat(tgt, 0)
    dataset = Data.TensorDataset(src, tgt)
    data_iter=torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=4) 
    return data_iter

def main():
    pth='./Data/GYAFC/Family_Relationships/train/informal'
    BATCH_SIZE=32
    MAX_LENGTH=30
    SAVE_LM_PTH='./GYAFC_for2in_LM.pth'
    LM_EPOCH=10
    
    print('Data creating.')
    tensors=data_processor(pth=pth, max_len=MAX_LENGTH)
    print('Training.')
    pretrain_LM(tensors, eopch_num=LM_EPOCH, batch_size=BATCH_SIZE, save_pth=SAVE_LM_PTH, show_step=20)

if __name__ == '__main__':
    main()
